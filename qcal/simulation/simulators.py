"""Submodule for simulation of quantum circuits.

"""
from __future__ import annotations

import logging
from typing import Any, List

import jax.numpy as jnp
import numpy as np
import quax

from qcal.circuit import Circuit, CircuitSet

logger = logging.getLogger(__name__)


__all__ = ('StateVectorSimulator',)


class StateVectorSimulator:
    """Noiseless statevector simulator backed by rigetti-quax.

    Each circuit is evolved from |0...0⟩ by applying the ideal
    unitary of every non-measurement gate. Measurement outcomes are
    sampled from the resulting probability distribution.

    Basic example usage::

        sim = StateVectorSimulator()
        sim.run(circuit)
        print(circuit.results)
    """

    def __init__(self, n_shots: int | None = None) -> None:
        """Initialize a StateVectorSimulator.

        Args:
            n_shots (int | None, optional): default number of shots
                per circuit. If ``None``, the exact probability
                distribution is returned instead of sampled counts.
                Defaults to ``None``.
        """
        self._n_shots = n_shots
        self._circuits = None
        self._states = None

    @property
    def states(self) -> list | None:
        """Final state vectors from the last call to run().

        Returns:
            list | None: one ``quax.StateVector`` per circuit, in the
                same order as ``self.circuits``, or ``None`` if
                ``run()`` has not been called.
        """
        return self._states

    @property
    def circuits(self) -> CircuitSet | None:
        """Circuits from the last call to run().

        Returns:
            CircuitSet | None: all circuits from the last run.
        """
        return self._circuits

    @property
    def n_shots(self) -> int | None:
        """Default number of shots per circuit.

        Returns:
            int | None: number of shots, or ``None`` for exact
                probabilities.
        """
        return self._n_shots

    def _simulate(self, circuit: Circuit) -> tuple:
        """Simulate a single circuit and return counts and final state.

        Args:
            circuit (Circuit): qcal Circuit to simulate.

        Returns:
            tuple: (counts dict, quax.StateVector) where counts maps
                bitstrings to counts (int) or probabilities (float).
        """
        qudits = sorted(circuit.qudits)
        n_qudits = len(qudits)
        qudit_to_idx = {q: i for i, q in enumerate(qudits)}

        qudit_dim = dict.fromkeys(qudits, 2)
        for cycle in circuit:
            if cycle.is_barrier:
                continue
            for gate in cycle:
                if gate.is_measurement:
                    continue
                n_g = len(gate.qudits)
                d = round(gate.unitary.shape[0] ** (1 / n_g))
                for q in gate.qudits:
                    qudit_dim[q] = max(qudit_dim[q], d)

        all_dims = tuple(qudit_dim[q] for q in qudits)
        state = quax.zero_state_vector(dims=all_dims)

        meas_qudits: list = []
        for cycle in circuit:
            if cycle.is_barrier:
                continue
            for gate in cycle:
                if gate.is_measurement:
                    meas_qudits.extend(gate.qudits)
                    continue
                n_g = len(gate.qudits)
                d = round(gate.unitary.shape[0] ** (1 / n_g))
                U = quax.Unitary.from_matrix(
                    jnp.array(gate.unitary, dtype=complex),
                    dims=((d,) * n_g, (d,) * n_g),
                )
                subsystem = tuple(
                    qudit_to_idx[q] for q in gate.qudits
                )
                state = quax.targeted_apply_unitary(
                    U, state, subsystem
                )

        if meas_qudits:
            meas_idx = [
                qudit_to_idx[q]
                for q in sorted(set(meas_qudits))
            ]
        else:
            meas_idx = list(range(n_qudits))

        strides = []
        for i in range(n_qudits):
            s = 1
            for j in range(i + 1, n_qudits):
                s *= all_dims[j]
            strides.append(s)

        probs = np.asarray(quax.probabilities(state), dtype=float)
        probs /= probs.sum()
        counts: dict = {}

        if self._n_shots is None:
            for raw in range(len(probs)):
                if probs[raw] == 0.0:
                    continue
                bits = ''.join(
                    str((raw // strides[m]) % all_dims[m])
                    for m in meas_idx
                )
                counts[bits] = (
                    counts.get(bits, 0.0) + float(probs[raw])
                )
        else:
            sampled = np.random.choice(
                len(probs), size=self._n_shots, p=probs
            )
            for raw in sampled:
                bits = ''.join(
                    str((int(raw) // strides[m]) % all_dims[m])
                    for m in meas_idx
                )
                counts[bits] = counts.get(bits, 0) + 1

        return counts, state

    def run(
        self,
        circuits: Any | Circuit | List[Any],
        n_shots:  int | None = None,
    ) -> None:
        """Simulate circuits and attach Results to each circuit.

        After calling this method, each circuit's ``.results``
        attribute holds a Results object with sampled bitstring
        counts.

        Args:
            circuits (Any | Circuit | List[Any]): circuit(s)
                to simulate.
            n_shots (int | None, optional): shots per circuit.
                ``None`` returns exact probabilities. Overrides
                the instance default when given. Defaults to
                ``None``.
        """
        if n_shots is not None:
            self._n_shots = n_shots

        if isinstance(circuits, CircuitSet):
            self._circuits = circuits
        else:
            if not isinstance(circuits, list):
                circuits = [circuits]
            self._circuits = CircuitSet(circuits=circuits)

        self._states = []
        for circuit in self._circuits:
            counts, state = self._simulate(circuit)
            circuit.results = counts
            self._states.append(state)
