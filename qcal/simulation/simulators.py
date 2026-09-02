"""Submodule for simulation of quantum circuits.

Simulation is backed by `rigetti-quax
<https://github.com/rigetti/quax>`_, a JAX-based quantum circuit
simulator. Because it is built on JAX, gate applications are
JIT-compiled on the first call and run efficiently on CPU or GPU.

Three classes are provided:

:class:`Simulator`
    Abstract base class. Handles common initialization, properties
    (``n_shots``, ``circuits``, ``states``), and the ``run()`` loop.
    Subclasses must implement :meth:`_simulate`.

:class:`StateVectorSimulator`
    Noiseless statevector simulation. Evolves an initial |0...0⟩
    state through every non-measurement gate and records the exact
    probability distribution (or sampled results).

:class:`DensityMatrixSimulator`
    Noisy density-matrix simulation. Evolves an initial
    |0...0⟩⟨0...0| state and, after each gate, applies an optional
    noise channel (``quax.KrausMap`` or ``quax.SuperOp``) specified
    by a *noise_model* dict keyed on gate-class name.

Both qubits (d=2) and qutrits (d=3) are supported; the simulator
detects the required per-qudit dimension from gate unitaries
automatically.

Workflow
--------
1. Construct a :class:`~qcal.circuit.Circuit` from
   :class:`~qcal.circuit.Cycle` objects using gates from
   ``qcal.gates``.
2. Create a simulator (optionally with a default ``n_shots``).
3. Call ``.run()`` — this attaches a
   :class:`~qcal.results.Results` object to each circuit.

When ``n_shots=None`` (the default), the exact probability
distribution is stored rather than sampled results.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

import jax.numpy as jnp
import numpy as np
import quax

from qcal.circuit import Circuit, CircuitSet
from qcal.gates.single_qubit import MCM
from qcal.simulation.error_models import ErrorModel

logger = logging.getLogger(__name__)


__all__ = ('Simulator', 'StateVectorSimulator', 'DensityMatrixSimulator')


class Simulator(ABC):
    """Abstract base class for quantum circuit simulators.

    Handles common initialization, shared properties, and the
    :meth:`run` loop. Subclasses must implement :meth:`_simulate`.
    """

    def __init__(self, n_shots: int | None = None) -> None:
        """Initialize a Simulator.

        Args:
            n_shots (int | None, optional): default number of shots per circuit.
                If ``None``, the exact probability distribution is returned
                instead of sampled results. Defaults to ``None``.
        """
        self._n_shots = n_shots
        self._circuits = None
        self._results = None
        self._states = None

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
            int | None: number of shots, or ``None`` for exact probabilities.
        """
        return self._n_shots

    @property
    def results(self) -> list[dict] | None:
        """Results from the last call to run().

        Returns:
            list[dict] | None: one results dict per circuit, or ``None`` if
                ``run()`` has not been called.
        """
        return self._results

    @property
    def states(self) -> list | None:
        """Final quantum states from the last call to run().

        Returns:
            list | None: one state object per circuit (type depends on the
                subclass), or ``None`` if ``run()`` has not been called.
        """
        return self._states

    @abstractmethod
    def _simulate(
        self, circuit: Circuit, n_shots: int | None
    ) -> tuple:
        """Simulate a single circuit and return results and final state.

        Args:
            circuit (Circuit): qcal Circuit to simulate.
            n_shots (int | None): shots to sample, or ``None`` for exact
                probabilities.

        Returns:
            tuple: (results dict, state) where results maps bitstrings
                to results (int) or probabilities (float).
        """

    def run(
        self,
        circuits: Circuit | CircuitSet | List[Circuit],
        n_shots:  int | None = None,
    ) -> None:
        """Simulate circuits and attach Results to each circuit.

        After calling this method, each circuit's ``.results`` attribute holds a
        Results object with sampled ditstring results (or exact probabilities
        when ``n_shots`` is ``None``).

        Args:
            circuits (Circuit | CircuitSet | List[Circuit]): circuit(s)
                to simulate.
            n_shots (int | None, optional): shots per circuit. ``None`` returns
                exact probabilities. Overrides the instance default when given.
                Defaults to ``None``.
        """
        _n_shots = n_shots if n_shots is not None else self._n_shots

        if isinstance(circuits, CircuitSet):
            self._circuits = circuits
        else:
            if not isinstance(circuits, list):
                circuits = [circuits]
            self._circuits = CircuitSet(circuits=circuits)

        self._results = []
        self._states = []
        for circuit in self._circuits:
            results, state = self._simulate(circuit, _n_shots)
            circuit.results = results
            self._results.append(results)
            self._states.append(state)


class StateVectorSimulator(Simulator):
    """Noiseless statevector simulator backed by rigetti-quax.

    Each circuit is evolved from |0...0⟩ by applying the ideal
    unitary of every non-measurement gate. Measurement outcomes are
    sampled from the resulting probability distribution.

    Basic example usage::

            sim = StateVectorSimulator()
            sim.run(circuit)
            print(circuit.results)

    To simulate non-ideal evolution, overwrite a gate's
    ``.unitary`` attribute with a custom matrix before calling
    :meth:`run`. The simulator reads each gate's unitary at
    simulation time, so any replacement is used as-is::

        import numpy as np
        from qcal.circuit import Circuit, Cycle
        from qcal.gates.single_qubit import X

        # Build a circuit with an X gate on qubit 0
        gate = X(0)

        # Replace with a slightly over-rotated unitary
        theta = np.pi + 0.1          # pi-pulse with 0.1 rad error
        gate.unitary = np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2),  np.cos(theta / 2)],
        ])

        circuit = Circuit([Cycle({gate})])

        sim = StateVectorSimulator()
        sim.run(circuit)
        print(circuit.results)
    """

    @property
    def states(self) -> list[quax.StateVector] | None:
        """Final state vectors from the last call to run().

        Returns:
            list[quax.StateVector] | None: one ``quax.StateVector`` per circuit,
                in the same order as ``self.circuits``, or ``None`` if ``run()``
                has not been called.
        """
        return self._states

    def _simulate(
        self, circuit: Circuit, n_shots: int | None
    ) -> tuple:
        """Simulate a single circuit and return results and final state.

        Args:
            circuit (Circuit): qcal Circuit to simulate.
            n_shots (int | None): shots to sample, or ``None`` for
                exact probabilities.

        Returns:
            tuple: (results dict, quax.StateVector) where results maps
                bitstrings to results (int) or probabilities (float).
        """
        qudits = sorted(circuit.qudits)
        n_qudits = len(qudits)
        # Map qudit label → position index used by quax subsystem args
        qudit_to_idx = {q: i for i, q in enumerate(qudits)}

        # Determine each qudit's Hilbert-space dimension (default qubit).
        # A gate acting on n_g qudits with a d^n_g × d^n_g unitary implies
        # local dimension d for each of its qudits.
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

        # Initialize |0...0⟩ in the joint Hilbert space
        all_dims = tuple(qudit_dim[q] for q in qudits)
        state = quax.zero_state_vector(dims=all_dims)

        # Apply gates cycle by cycle; collect measured qudits for readout
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

        # Which qudit positions contribute to the output bitstring
        if meas_qudits:
            meas_idx = [
                qudit_to_idx[q]
                for q in sorted(set(meas_qudits))
            ]
        else:
            meas_idx = list(range(n_qudits))

        # strides[i] is the place value of qudit i in the flat probability
        # array index. quax lays out the joint state as a big-endian mixed-
        # radix integer: index = d0*d1*...*d_{n-1} with qudit 0 most
        # significant. For a 3-qudit system with dims (d0, d1, d2):
        #   strides = [d1*d2, d2, 1]
        # so the digit for qudit m is (raw // strides[m]) % all_dims[m].
        strides = []
        for i in range(n_qudits):
            s = 1
            for j in range(i + 1, n_qudits):
                s *= all_dims[j]
            strides.append(s)

        # Normalize to guard against floating-point drift
        probs = np.asarray(quax.probabilities(state), dtype=float)
        probs /= probs.sum()
        results: dict = {}

        if n_shots is None:
            # Return the exact probability distribution
            for raw in range(len(probs)):
                if probs[raw] == 0.0:
                    continue
                bits = ''.join(
                    str((raw // strides[m]) % all_dims[m])
                    for m in meas_idx
                )
                results[bits] = (
                    results.get(bits, 0.0) + float(probs[raw])
                )
        else:
            # Sample shot outcomes from the distribution
            sampled = np.random.choice(
                len(probs), size=n_shots, p=probs
            )
            for raw in sampled:
                bits = ''.join(
                    str((int(raw) // strides[m]) % all_dims[m])
                    for m in meas_idx
                )
                results[bits] = results.get(bits, 0) + 1

        return results, state


class DensityMatrixSimulator(Simulator):
    """Noisy density-matrix simulator backed by rigetti-quax.

    Each circuit is evolved from |0...0⟩⟨0...0| by applying the
    ideal unitary of every non-measurement gate. After each gate,
    an optional noise channel from *noise_model* is applied to the
    same subsystem. Measurement outcomes are sampled from the
    diagonal of the resulting density matrix.

    The *noise_model* may be either:

    * A :class:`~qcal.simulation.error_models.ErrorModel` instance
      (e.g. :class:`~qcal.simulation.error_models.DepolarizingNoise`)
      that assigns channels by broad gate category
      (``'single_qubit'``, ``'two_qubit'``, etc.)::

          from qcal.simulation.error_models import DepolarizingNoise
          noise = DepolarizingNoise(single_qubit=0.001, two_qubit=0.01)
          sim = DensityMatrixSimulator(noise_model=noise)
          sim.run(circuit)

    * A plain ``dict`` mapping gate-class names to ``quax.KrausMap``
      or ``quax.SuperOp`` channels for fine-grained control::

          import quax
          noise_model = {
              'X':  quax.channels.depolarizing(0.01),
              'CZ': quax.channels.depolarizing(0.05),
          }
          sim = DensityMatrixSimulator(noise_model=noise_model)
          sim.run(circuit)

    Mid-circuit measurement (MCM) note:
        MCM outcomes (``circuit.mcm_results``) and terminal outcomes
        (``circuit.results``) are each sampled from their correct
        marginal distributions, but the two samplings are independent.
        Shot-by-shot cross-correlations — e.g. the joint distribution
        P(terminal=0 | MCM=1) — are not preserved. A per-shot
        conditional (quantum-trajectory) simulation is required to
        capture that correlation.
    """

    def __init__(
        self,
        noise_model: ErrorModel | dict | None = None,
        n_shots:     int | None = None,
    ) -> None:
        """Initialize a DensityMatrixSimulator.

        Args:
            noise_model (ErrorModel | dict | None, optional): either a
                :class:`~qcal.simulation.error_models.ErrorModel`
                instance (category-based) or a ``dict`` mapping
                gate-class names to ``quax.KrausMap`` / ``quax.SuperOp``
                channels. The channel is applied immediately after each
                gate's unitary. Defaults to ``None`` (no noise).
            n_shots (int | None, optional): default number of shots
                per circuit. ``None`` returns exact probabilities.
                Defaults to ``None``.
        """
        super().__init__(n_shots)
        self._noise_model = noise_model if noise_model is not None else {}

    @property
    def noise_model(self) -> ErrorModel | dict:
        """The active noise model.

        Returns:
            ErrorModel | dict: a
                :class:`~qcal.simulation.error_models.ErrorModel`
                instance or a gate-name → channel dict.
        """
        return self._noise_model

    @property
    def states(self) -> list[quax.DensityMatrix] | None:
        """Final density matrices from the last call to run().

        Returns:
            list[quax.DensityMatrix] | None: one ``quax.DensityMatrix`` per
                circuit, or ``None`` if ``run()`` has not been called.
        """
        return self._states

    def _apply_channel(
        self,
        channel:   quax.KrausMap | quax.SuperOp,
        rho:       quax.DensityMatrix,
        subsystem: tuple,
    ) -> quax.DensityMatrix:
        """Apply a KrausMap or SuperOp to rho at subsystem.

        If the channel acts on a single qudit but the gate subsystem
        has multiple qudits, the channel is applied independently to
        each qudit (independent per-qudit noise). To apply a joint
        multi-qudit channel, provide a channel whose number of
        subsystems matches the gate's qudit count exactly.

        Args:
            channel: ``quax.KrausMap`` or ``quax.SuperOp``.
            rho: ``quax.DensityMatrix`` to update.
            subsystem (tuple): qudit index positions.

        Returns:
            quax.DensityMatrix: updated density matrix.

        Raises:
            TypeError: if *channel* is neither ``KrausMap`` nor
                ``SuperOp``.
            ValueError: if the channel qudit count is > 1 and does
                not match the gate's subsystem size.
        """
        if not isinstance(channel, (quax.KrausMap, quax.SuperOp)):
            raise TypeError(
                f'Expected quax.KrausMap or quax.SuperOp, '
                f'got {type(channel).__name__}'
            )
        ch_n = len(channel.dims[0])
        if ch_n == len(subsystem):
            targets = [subsystem]
        elif ch_n == 1:
            # Apply single-qudit channel independently per qudit
            targets = [(idx,) for idx in subsystem]
        else:
            raise ValueError(
                f'Channel acts on {ch_n} qudits but gate subsystem '
                f'has {len(subsystem)} qudits.'
            )

        for target in targets:
            if isinstance(channel, quax.KrausMap):
                rho = quax.targeted_apply_kraus_map(
                    channel, rho, target
                )
            else:
                rho = quax.targeted_apply_superop(
                    channel, rho, target
                )

        return rho

    def _simulate(
        self, circuit: Circuit, n_shots: int | None
    ) -> tuple:
        """Simulate a single circuit and return results and final state.

        Args:
            circuit (Circuit): qcal Circuit to simulate.
            n_shots (int | None): shots to sample, or ``None`` for
                exact probabilities.

        Returns:
            tuple: (results dict, quax.DensityMatrix).
        """
        qudits = sorted(circuit.qudits)
        n_qudits = len(qudits)
        qudit_to_idx = {q: i for i, q in enumerate(qudits)}

        # Determine per-qudit Hilbert-space dimension (default qubit)
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
        rho = quax.zero_state_matrix(dims=all_dims)

        # Apply gates cycle by cycle; collect measured qudits
        meas_qudits: list = []
        # Ordered records: (cycle_mcm_qudits, joint_flat) — one per cycle
        # with MCM gates. joint_flat is the pre-instrument joint probability
        # distribution over all MCM qudits in that cycle (sorted order).
        mcm_gate_records: list = []
        # integer qudit labels measured mid-circuit
        mcm_qudits: set = set()
        for cycle in circuit:
            if cycle.is_barrier:
                continue

            # Snapshot joint distribution over this cycle's MCM qudits
            # BEFORE applying any instruments (so outcomes are correlated).
            cycle_mcm_gates = [g for g in cycle if isinstance(g, MCM)]
            if cycle_mcm_gates:
                cycle_mcm_qudits = sorted(
                    q for g in cycle_mcm_gates for q in g.qudits
                )
                cycle_mcm_indices = [
                    qudit_to_idx[q] for q in cycle_mcm_qudits
                ]
                snap = np.asarray(quax.probabilities(rho), dtype=float)
                snap /= snap.sum()
                probs_nd = snap.reshape(all_dims)
                non_mcm_axes = tuple(
                    ax for ax in range(len(all_dims))
                    if ax not in cycle_mcm_indices
                )
                joint = (
                    probs_nd.sum(axis=non_mcm_axes)
                    if non_mcm_axes else probs_nd
                )
                joint_flat = joint.flatten()
                joint_flat /= joint_flat.sum()
                mcm_gate_records.append((cycle_mcm_qudits, joint_flat))

            for gate in cycle:
                if gate.is_measurement:
                    if isinstance(gate, MCM):
                        for q in gate.qudits:
                            idx = qudit_to_idx[q]
                            d = qudit_dim[q]
                            instrument = (
                                self._noise_model.instrument_for(
                                    f'Q{q}', d,
                                )
                                if isinstance(
                                    self._noise_model, ErrorModel
                                )
                                else None
                            )
                            if instrument is None:
                                instrument = (
                                    quax.instrument_from_confusion_and_transition(
                                        jnp.eye(d, dtype=float),
                                        jnp.eye(d, dtype=float),
                                        dims=(d,),
                                    )
                                )
                            rho_outs, _ = (
                                quax.targeted_apply_instrument_to_density_matrix(
                                    instrument, rho,
                                    subsystem=(idx,),
                                )
                            )
                            mcm_qudits.add(q)
                            # NOTE: Unconditional post-MCM state: average
                            # over all measurement outcomes. This gives
                            # the correct marginal state for subsequent
                            # gates, but discards conditioning on the
                            # specific MCM outcome, so MCM outcomes and
                            # terminal outcomes are sampled from
                            # independent marginals rather than a joint
                            # distribution. See class docstring.
                            rho = quax.DensityMatrix.from_matrix(
                                jnp.sum(rho_outs.matrix, axis=0),
                                all_dims,
                            )
                    else:
                        meas_qudits.extend(gate.qudits)
                    continue
                n_g = len(gate.qudits)
                d = round(gate.unitary.shape[0] ** (1 / n_g))
                subsystem = tuple(
                    qudit_to_idx[q] for q in gate.qudits
                )
                U = quax.Unitary.from_matrix(
                    jnp.array(gate.unitary, dtype=complex),
                    dims=((d,) * n_g, (d,) * n_g),
                )
                rho = quax.targeted_apply_unitary_to_density_matrix(
                    U, rho, subsystem
                )
                gate_name = type(gate).__name__
                if isinstance(self._noise_model, ErrorModel):
                    channel = self._noise_model.channel_for_gate(gate)
                    if channel is None:
                        channel = self._noise_model.channel_for(
                            gate_name
                        )
                else:
                    channel = self._noise_model.get(gate_name)
                if channel is not None:
                    rho = self._apply_channel(
                        channel, rho, subsystem
                    )

        if mcm_qudits and n_shots is None:
            raise ValueError(
                'n_shots must be specified when the circuit contains '
                'MCM gates; exact probability mode is not supported '
                'for mid-circuit measurements.'
            )

        # Terminal Meas qudits define circuit.results bitstring positions.
        # MCM intermediate outcomes go to circuit.mcm_results, not here.
        all_meas: list = sorted(
            set(meas_qudits) if meas_qudits
            else set(range(n_qudits))
        )

        # Same stride/bitstring logic as StateVectorSimulator
        strides = []
        for i in range(n_qudits):
            s = 1
            for j in range(i + 1, n_qudits):
                s *= all_dims[j]
            strides.append(s)

        # Diagonal of ρ gives computational-basis probabilities
        probs = np.asarray(quax.probabilities(rho), dtype=float)
        probs /= probs.sum()

        # Classical per-qudit readout smearing for terminal Meas gates
        if isinstance(self._noise_model, ErrorModel):
            for q in sorted(set(meas_qudits)):
                cmat = self._noise_model.confusion_matrix_for(f'Q{q}')
                if cmat is None:
                    continue
                m = qudit_to_idx[q]
                d = all_dims[m]
                s = strides[m]
                probs_new = np.zeros_like(probs)
                for raw in range(len(probs)):
                    j = (raw // s) % d       # actual digit of qudit m
                    for i in range(d):       # reported digit
                        probs_new[raw + (i - j) * s] += (
                            cmat[i, j] * probs[raw]
                        )
                probs = probs_new / probs_new.sum()

        results: dict = {}

        if n_shots is None:
            # No MCM gates (guard above ensures this)
            for raw in range(len(probs)):
                if probs[raw] == 0.0:
                    continue
                bits = ''.join(
                    str((raw // strides[qudit_to_idx[q]]) % all_dims[qudit_to_idx[q]])
                    for q in all_meas
                )
                results[bits] = (
                    results.get(bits, 0.0) + float(probs[raw])
                )
        else:
            sampled = np.random.choice(
                len(probs), size=n_shots, p=probs
            )
            for shot in range(n_shots):
                raw = int(sampled[shot])
                bits = ''.join(
                    str(
                        (raw // strides[qudit_to_idx[q]])
                        % all_dims[qudit_to_idx[q]]
                    )
                    for q in all_meas
                )
                results[bits] = results.get(bits, 0) + 1

            # Populate circuit.mcm_results: one Results entry per cycle
            # with MCM gates, sampled jointly from the pre-instrument
            # joint probability distribution over that cycle's MCM qudits.
            # Note: these samples are drawn independently of the terminal
            # measurement samples above, so shot index i in mcm_results
            # does not correspond to the same trajectory as results[i].
            # See class docstring for details.
            if mcm_gate_records:
                mcm_results_list = []
                for cycle_mcm_qudits, joint_flat in mcm_gate_records:
                    cycle_mcm_dims = tuple(
                        all_dims[qudit_to_idx[q]]
                        for q in cycle_mcm_qudits
                    )
                    cycle_strides: list = []
                    for k in range(len(cycle_mcm_dims)):
                        s = 1
                        for j in range(k + 1, len(cycle_mcm_dims)):
                            s *= cycle_mcm_dims[j]
                        cycle_strides.append(s)
                    cycle_outcomes = np.random.choice(
                        len(joint_flat), size=n_shots, p=joint_flat
                    )
                    gate_results: dict = {}
                    for shot in range(n_shots):
                        raw = int(cycle_outcomes[shot])
                        bits = ''.join(
                            str(
                                (raw // cycle_strides[k])
                                % cycle_mcm_dims[k]
                            )
                            for k in range(len(cycle_mcm_qudits))
                        )
                        gate_results[bits] = (
                            gate_results.get(bits, 0) + 1
                        )
                    mcm_results_list.append(gate_results)
                circuit._mcm_results = []
                circuit.mcm_results = mcm_results_list

        return results, rho
