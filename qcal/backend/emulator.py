"""Emulator backend for qcal.

Provides a drop-in QPU substitute that runs a density-matrix
simulation in place of real hardware. Useful for testing calibration,
characterization, and benchmarking workflows end-to-end without a
physical backend.

Default noise model:

- Depolarizing gate noise: 0.0005 on single-qubit gates, 0.005 on
  two-qubit gates.
- Readout confusion matrix: diag(0.995, 0.98) applied to every qubit
  defined in the config.

Basic usage::

    from qcal.backend.emulator import Emulator
    from qcal.config import Config

    config = Config('examples/config/config.yaml')
    qpu = Emulator(config)
    qpu.run(circuits)

Or load the bundled example config automatically::

    qpu = Emulator()
    qpu.run(circuits)
"""
import logging
import pathlib

import numpy as np

from qcal.circuit import Circuit, CircuitSet
from qcal.config import Config
from qcal.qpu.qpu import QPU
from qcal.simulation.error_models import DepolarizingNoise
from qcal.simulation.simulators import (
    DensityMatrixSimulator,
    Simulator,
)

logger = logging.getLogger(__name__)

__all__ = ('Emulator',)

# Path to the example config shipped with the repository
_EXAMPLE_CONFIG = (
    pathlib.Path(__file__).parent.parent.parent
    / 'examples' / 'config' / 'config.yaml'
)


class Emulator(QPU):
    """Emulator QPU backed by a quantum circuit simulator.

    Inherits from :class:`~qcal.qpu.qpu.QPU` and replaces the
    hardware acquisition layer with a
    :class:`~qcal.simulation.simulators.Simulator` instance. Any
    :class:`~qcal.simulation.simulators.Simulator` subclass is
    accepted via the *simulator* argument, including
    :class:`~qcal.simulation.simulators.StateVectorSimulator` for
    noiseless simulation.

    When no *simulator* is provided, a
    :class:`~qcal.simulation.simulators.DensityMatrixSimulator` is
    constructed with the following default noise model:

    - **Gate noise** — depolarizing channel: 0.0005 on single-qubit
      gates, 0.005 on two-qubit gates.
    - **Readout noise** — confusion matrix diag(0.995, 0.98) applied
      to every qubit defined in the config.

    Basic example usage::

        from qcal.backend.emulator import Emulator

        qpu = Emulator()          # uses example config + default noise
        qpu.run(circuit)
        print(circuit.results)
    """

    def __init__(
        self,
        config:          Config | None = None,
        n_shots:         int = 1024,
        n_batches:       int = 1,
        n_circs_per_seq: int = 1,
        n_levels:        int = 2,
        simulator:       Simulator | None = None,
        **kwargs
    ) -> None:
        """Initialize an Emulator QPU.

        Args:
            config (Config | None, optional): qcal Config object. When
                ``None``, the example config bundled under
                ``examples/config/config.yaml`` is loaded. Defaults to
                ``None``.
            n_shots (int, optional): number of shots per circuit.
                Defaults to 1024.
            n_batches (int, optional): number of measurement batches.
                Defaults to 1.
            n_circs_per_seq (int, optional): maximum number of circuits
                per sequence batch. Defaults to 1.
            n_levels (int, optional): number of energy levels to
                simulate. Defaults to 2.
            simulator (Simulator | None, optional): a pre-configured
                :class:`~qcal.simulation.simulators.Simulator` instance
                (e.g. a custom
                :class:`~qcal.simulation.simulators.DensityMatrixSimulator`
                or
                :class:`~qcal.simulation.simulators.StateVectorSimulator`)
                to use in place of the default depolarizing model.
                Defaults to ``None``.
        """
        if config is None:
            config = Config(str(_EXAMPLE_CONFIG))

        QPU.__init__(
            self,
            config=config,
            n_shots=n_shots,
            n_batches=n_batches,
            n_circs_per_seq=n_circs_per_seq,
            n_levels=n_levels,
            jit=False,
            **kwargs
        )

        if simulator is not None:
            self._simulator = simulator
        else:
            noise = DepolarizingNoise(
                single_qubit=0.0005,
                two_qubit=0.005,
            )

            # Confusion matrix in qcal's row-stochastic convention:
            # C[prep, meas] = P(measure meas | prepared prep).
            # diag(p0, p1) => C = [[p0, 1-p0], [1-p1, p1]]
            p0, p1 = (0.995, 0.98)
            cmat = np.array([
                [p0,       1.0 - p0],
                [1.0 - p1, p1      ],
            ])
            for q in config.qubits:
                noise.add_readout_noise(q, cmat)

            self._simulator = DensityMatrixSimulator(
                noise_model=noise,
                n_shots=n_shots,
            )

        self._measurements = None

    @property
    def simulator(self) -> Simulator:
        """Underlying simulator.

        Returns:
            Simulator: active simulator instance.
        """
        return self._simulator

    def _initialize(
        self,
        circuits,
        n_shots:   int | None = None,
        n_batches: int | None = None,
    ) -> None:
        """Initialize the experiment and reset measurement storage.

        Calls the parent :meth:`~qcal.qpu.qpu.QPU._initialize` then
        resets ``self._measurements`` to the dict format expected by
        :meth:`acquire` and :meth:`process`.

        Args:
            circuits: circuits to measure.
            n_shots (int | None, optional): number of shots per batch.
                Defaults to ``None``.
            n_batches (int | None, optional): number of batches of
                shots. Defaults to ``None``.
        """
        super()._initialize(circuits, n_shots, n_batches)
        self._measurements = {'results': [], 'states': []}

    def acquire(self) -> None:
        """Simulate the current circuit batch.

        Runs the simulator on ``self._exp_circuits`` and attaches a
        :class:`~qcal.results.Results` object to each circuit.
        Appends per-circuit results dicts and states to
        ``self._measurements``.
        """
        self._simulator.run(
            self._exp_circuits, n_shots=self._n_shots
        )
        self._measurements['results'].extend(self._simulator.results)
        self._measurements['states'].extend(self._simulator.states)

    def process(self) -> None:
        """Write simulation results and states to all circuit sets.

        Reads the results and states accumulated by :meth:`acquire`
        and writes them to every non-empty circuit set
        (``self._circuits``, ``self._transpiled_circuits``,
        ``self._compiled_circuits``).

        For each circuit set:

        - A ``'results'`` column is added (one results dict per
          circuit).
        - A ``'state'`` column is added (one
          ``state.pretty_print()`` string per circuit).
        - For qcal :class:`~qcal.circuit.Circuit` objects the results
          dict is also written to the circuit's ``.results``
          attribute directly.
        """
        all_results = self._measurements['results']
        all_states = [
            state.pretty_print() for state in self._measurements['states']
        ]

        for circuit_set in (
            self._circuits,
            self._transpiled_circuits,
            self._compiled_circuits,
        ):
            if not (
                isinstance(circuit_set, CircuitSet)
                and len(circuit_set) > 0
            ):
                continue
            circuit_set['results'] = all_results
            circuit_set['state'] = all_states
            for circuit, results in zip(
                circuit_set, all_results, strict=True
            ):
                if isinstance(circuit, Circuit):
                    circuit.results = results

