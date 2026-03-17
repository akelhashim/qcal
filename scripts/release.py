#!/usr/bin/env python3

"""Release helper for qcal.

This script automates the mechanical steps of cutting a new release using the
repository's Keep-a-Changelog formatted ``CHANGELOG.md`` and the package version
declared in ``qcal/__init__.py``.

It performs the following operations:

- Promotes the contents of ``## [Unreleased]`` into a new version section
  ``## [X.Y.Z] - YYYY-MM-DD``.
- Resets ``[Unreleased]`` to an empty template of subsections.
- Updates the reference links at the bottom of the changelog:
  - ``[Unreleased]: .../compare/v<new>...HEAD``
  - Adds ``[<new>]: .../releases/tag/v<new>``
- Bumps ``__version__`` in ``qcal/__init__.py`` (``pyproject.toml`` pulls this
  dynamically via setuptools).

Use ``--dry-run`` to preview changes as unified diffs.
"""
import argparse
import datetime as _dt
import difflib
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

CHANGELOG_DEFAULT = pathlib.Path("CHANGELOG.md")
INIT_DEFAULT = pathlib.Path("qcal") / "__init__.py"


_UNRELEASED_HEADER_RE = re.compile(r"^## \[Unreleased\]\s*$")
_VERSION_HEADER_RE = re.compile(
    r"^## \[(?P<version>[^\]]+)\]\s*-\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$"
)
_ANY_SECTION_HEADER_RE = re.compile(r"^## \[(?P<label>[^\]]+)\].*$")
_SUBSECTION_RE = re.compile(r"^###\s+(?P<title>.+?)\s*$")


def _read_text(path: pathlib.Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def _write_text(path: pathlib.Path, content: str) -> None:
    """Write a UTF-8 text file."""
    path.write_text(content, encoding="utf-8")


def _unified_diff(old: str, new: str, filename: str) -> str:
    """Return a unified diff between two strings with stable file labels."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )


def _find_unreleased_block(lines: List[str]) -> Tuple[int, int]:
    """Locate the ``[Unreleased]`` section block in a changelog.

    Args:
        lines: Full changelog split into lines (including newline characters).

    Returns:
        A ``(start, end)`` tuple of indices such that ``lines[start]`` is the
        ``## [Unreleased]`` header and ``lines[start + 1 : end]`` is the body.
        ``end`` is the index of the next ``## [...]`` header after
        ``[Unreleased]``.

    Raises:
        ValueError: If the ``## [Unreleased]`` header or its end boundary cannot
            be found.
    """
    unreleased_idx = None
    for i, line in enumerate(lines):
        if _UNRELEASED_HEADER_RE.match(line.rstrip("\n")):
            unreleased_idx = i
            break

    if unreleased_idx is None:
        raise ValueError("Could not find '## [Unreleased]' header in changelog")

    end_idx = None
    for j in range(unreleased_idx + 1, len(lines)):
        is_section = _ANY_SECTION_HEADER_RE.match(lines[j].rstrip("\n"))
        is_unreleased = _UNRELEASED_HEADER_RE.match(lines[j].rstrip("\n"))
        if is_section and not is_unreleased:
            end_idx = j
            break

    if end_idx is None:
        raise ValueError("Could not find the end of the [Unreleased] section")

    return unreleased_idx, end_idx


def _split_subsections(
    body_lines: List[str]
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Parse Keep-a-Changelog subsections (``### Title``) from a section body.

    Args:
        body_lines: Lines in a section body.

    Returns:
        A pair ``(order, content)``.

        - ``order`` is a list of subsection titles in the order they appear.
        - ``content`` maps subsection title to its raw body lines.
    """
    order: List[str] = []
    content: Dict[str, List[str]] = {}

    current: Optional[str] = None
    for line in body_lines:
        m = _SUBSECTION_RE.match(line.rstrip("\n"))
        if m:
            current = m.group("title")
            if current not in content:
                order.append(current)
                content[current] = []
            continue

        if current is None:
            continue

        content[current].append(line)

    return order, content


def _is_subsection_content_nonempty(lines: List[str]) -> bool:
    """Return True if a subsection contains any non-empty, non-header content.
    """
    for ln in lines:
        if ln.strip() and not ln.lstrip().startswith("#"):
            return True
    return False


def _build_unreleased_empty(order: List[str]) -> List[str]:
    """Build an empty ``[Unreleased]`` body using the observed subsection order.
    """
    out: List[str] = []
    if not order:
        order = [
            "Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"
        ]

    for title in order:
        out.append(f"### {title}\n")
        out.append("\n")

    return out


def _build_release_body(
    order: List[str], subsection_content: Dict[str, List[str]]
) -> List[str]:
    """Build the new release section body from non-empty subsections."""
    out: List[str] = []
    for title in order:
        lines = subsection_content.get(title, [])
        if not _is_subsection_content_nonempty(lines):
            continue

        trimmed_lines = list(lines)
        while trimmed_lines and trimmed_lines[0].strip() == "":
            trimmed_lines.pop(0)

        out.append(f"### {title}\n")
        out.append("\n")
        out.extend(trimmed_lines)
        if len(out) and (not out[-1].endswith("\n")):
            out[-1] = out[-1] + "\n"
        if len(out) and out[-1].strip() != "":
            out.append("\n")

    return out


def _update_reference_links(
    changelog_text: str, new_version: str, previous_version: str
) -> str:
    """Update the bottom reference links in the changelog.

    This updates the ``[Unreleased]`` compare URL to point at the new version
    and inserts a new tag URL line for ``new_version``.

    The current implementation expects the existing ``[Unreleased]`` link to be
    of the form ``.../compare/v<prev>...HEAD``.

    Args:
        changelog_text: Full changelog text.
        new_version: The version being released.
        previous_version: The version currently referenced by ``[Unreleased]``.

    Returns:
        Updated changelog text.

    Raises:
        ValueError: If the existing reference links do not match the expected
            Keep-a-Changelog reference format.
    """
    lines = changelog_text.splitlines(keepends=True)

    unreleased_line_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("[Unreleased]:"):
            unreleased_line_idx = i
            break

    if unreleased_line_idx is None:
        raise ValueError(
            "Could not find '[Unreleased]:' reference link at bottom of "
            "changelog"
        )

    unreleased_line = lines[unreleased_line_idx]
    m = re.match(
        (
            r"^\[Unreleased\]:\s+(?P<repo>https?://[^\s]+)/compare/v"
            r"(?P<prev>[^.\s]+(?:\.[^.\s]+)*)\.\.\.HEAD\s*$"
        ),
        unreleased_line.strip(),
    )
    if not m:
        raise ValueError(
            "Unreleased compare link did not match expected format: "
            "'[Unreleased]: .../compare/vX.Y.Z...HEAD'"
        )

    repo = m.group("repo")

    new_unreleased_line = (
        f"[Unreleased]: {repo}/compare/v{new_version}...HEAD\n"
    )
    lines[unreleased_line_idx] = new_unreleased_line

    new_tag_line = f"[{new_version}]: {repo}/releases/tag/v{new_version}\n"

    version_ref_re = re.compile(r"^\[(?P<ver>\d+\.\d+\.\d+)\]:\s+.*$")

    insert_at = unreleased_line_idx + 1
    for i in range(unreleased_line_idx + 1, len(lines)):
        if version_ref_re.match(lines[i].strip()):
            insert_at = i
            break

    existing_versions = {
        m.group("ver")
        for ln in lines
        for m in [version_ref_re.match(ln.strip())]
        if m
    }
    if new_version in existing_versions:
        raise ValueError(
            f"Changelog already contains a reference link for version "
            f"{new_version}"
        )

    lines.insert(insert_at, new_tag_line)

    return "".join(lines)


def _extract_current_version_from_init(init_text: str) -> str:
    """Extract the current package version from ``qcal/__init__.py`` contents.
    """
    m = re.search(
        r"^__version__\s*=\s*\"(?P<ver>[^\"]+)\"\s*$",
        init_text,
        flags=re.M,
    )
    if not m:
        raise ValueError(
            "Could not find __version__ assignment in qcal/__init__.py"
        )
    return m.group("ver")


def _bump_version_in_init(init_text: str, new_version: str) -> str:
    """Return updated ``qcal/__init__.py`` contents with ``__version__`` set."""
    new_text, n = re.subn(
        r"^(__version__\s*=\s*\")[^\"]+(\"\s*)$",
        rf"\g<1>{new_version}\2",
        init_text,
        flags=re.M,
    )
    if n != 1:
        raise ValueError("Failed to update __version__ in qcal/__init__.py")
    return new_text


def _bump_patch_version(version: str) -> str:
    m = re.fullmatch(
        r"(?P<maj>\d+)\.(?P<min>\d+)\.(?P<pat>\d+)", version.strip()
    )
    if not m:
        raise ValueError(
            (
                f"Version '{version}' is not in expected X.Y.Z format; please "
                "pass an explicit version"
            )
        )
    maj = int(m.group("maj"))
    min_ = int(m.group("min"))
    pat = int(m.group("pat"))
    return f"{maj}.{min_}.{pat + 1}"


def release(
    *,
    version: str,
    date: str,
    changelog_path: pathlib.Path,
    init_path: pathlib.Path,
) -> Tuple[str, str]:
    """Compute updated changelog and init contents for a new release.

    This does not write to disk; it returns the new file contents.

    Args:
        version: New version to release.
        date: Release date in ``YYYY-MM-DD`` format.
        changelog_path: Path to ``CHANGELOG.md``.
        init_path: Path to ``qcal/__init__.py``.

    Returns:
        A pair ``(new_changelog_text, new_init_text)``.

    Raises:
        ValueError: If the changelog structure is not as expected or
            ``[Unreleased]`` contains no releasable content.
    """
    changelog_text = _read_text(changelog_path)
    init_text = _read_text(init_path)

    previous_version = _extract_current_version_from_init(init_text)

    lines = changelog_text.splitlines(keepends=True)
    unreleased_start, unreleased_end = _find_unreleased_block(lines)

    unreleased_body = lines[unreleased_start + 1 : unreleased_end]
    subsection_order, subsection_content = _split_subsections(unreleased_body)

    release_body = _build_release_body(subsection_order, subsection_content)
    if not release_body:
        raise ValueError(
            "No content found under [Unreleased]; nothing to release"
        )

    empty_unreleased_body = _build_unreleased_empty(subsection_order)

    new_version_header = f"## [{version}] - {date}\n"

    new_lines: List[str] = []
    new_lines.extend(lines[: unreleased_start + 1])
    if len(new_lines) and new_lines[-1].strip() != "":
        new_lines.append("\n")
    new_lines.extend(empty_unreleased_body)

    if new_lines and new_lines[-1].strip() != "":
        new_lines.append("\n")

    new_lines.append(new_version_header)
    new_lines.append("\n")
    new_lines.extend(release_body)

    if new_lines and new_lines[-1].strip() != "":
        new_lines.append("\n")

    new_lines.extend(lines[unreleased_end:])

    new_changelog_text = "".join(new_lines)
    new_changelog_text = _update_reference_links(
        new_changelog_text,
        version,
        previous_version,
    )

    new_init_text = _bump_version_in_init(init_text, version)

    return new_changelog_text, new_init_text


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a new qcal release by promoting CHANGELOG [Unreleased]."
        )
    )
    parser.add_argument(
        "version",
        nargs="?",
        default=None,
        help=(
            "New version to release (e.g. 0.0.4). If omitted, bumps patch by 1."
        ),
    )
    parser.add_argument(
        "--date",
        default=_dt.date.today().isoformat(),
        help="Release date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--changelog",
        default=str(CHANGELOG_DEFAULT),
        help="Path to CHANGELOG.md",
    )
    parser.add_argument(
        "--init",
        dest="init_path",
        default=str(INIT_DEFAULT),
        help="Path to qcal/__init__.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diffs, do not write files",
    )

    args = parser.parse_args(argv)

    try:
        changelog_path = pathlib.Path(args.changelog)
        init_path = pathlib.Path(args.init_path)

        old_changelog = _read_text(changelog_path)
        old_init = _read_text(init_path)

        version = args.version
        if version is None:
            current_version = _extract_current_version_from_init(old_init)
            version = _bump_patch_version(current_version)

        new_changelog, new_init = release(
            version=version,
            date=args.date,
            changelog_path=changelog_path,
            init_path=init_path,
        )

        if args.dry_run:
            sys.stdout.write(
                _unified_diff(old_changelog, new_changelog, str(changelog_path))
            )
            sys.stdout.write(_unified_diff(old_init, new_init, str(init_path)))
            return 0

        _write_text(changelog_path, new_changelog)
        _write_text(init_path, new_init)
        return 0

    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
