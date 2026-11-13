"""Sphinx configuration for the qcal documentation.

Build locally with::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html
"""

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'qcal'
author = 'Akel Hashim'
copyright = (
    '2025, The Regents of the University of California, through Lawrence '
    'Berkeley National Laboratory'
)
try:
    release = _pkg_version('quantum-calibration')
except PackageNotFoundError:  # pragma: no cover - docs built from source
    import qcal

    release = qcal.__version__
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'myst_nb',  # Markdown (MyST) pages + Jupyter notebooks
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Napoleon (Google-style docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_rtype = False
napoleon_use_ivar = True
# Resolve short type names in docstrings that exist in several modules.
napoleon_preprocess_types = True
napoleon_type_aliases = {
    'Transpiler': '~qcal.transpilation.transpiler.Transpiler',
}

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
# Members are listed explicitly in the autosummary templates
# (docs/_templates/autosummary/), so module pages only show the module
# docstring and each class/function gets its own page.
autodoc_default_options = {
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autoclass_content = 'class'

# Optional / hardware-specific dependencies that are not installed on
# Read the Docs. Autodoc replaces these with mock objects on import.
autodoc_mock_imports = [
    'qubic',
    'qubitconfig',
    'distproc',
    'trueq',
    'qiskit_superstaq',
    'cirq_superstaq',
    'general_superstaq',
]

# -- MyST / MyST-NB ----------------------------------------------------------

myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
    'amsmath',
    'deflist',
]
myst_heading_anchors = 3
nb_execution_mode = 'off'  # tutorials talk to hardware; never execute

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_title = f'qcal {release}'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}


# -- Hooks -------------------------------------------------------------------


def _skip_third_party_members(app, what, name, obj, skip, options):
    """Skip members whose docstrings are inherited from matplotlib.

    Matplotlib generates a ``set`` method on every ``Artist`` subclass (e.g.
    ``qcal.plotting.bloch_sphere.Arrow3D``) with a docstring that uses
    matplotlib-specific Sphinx roles, which break the qcal build.
    """
    doc = getattr(obj, '__doc__', None) or ''
    if ':mpltype:' in doc:
        return True
    return skip


def setup(app):
    """Register Sphinx event hooks."""
    app.connect('autodoc-skip-member', _skip_third_party_members)
