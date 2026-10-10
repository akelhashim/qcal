"""Submodule for readout characterization.

"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from typing import Callable

from IPython.display import clear_output

import qcal.settings as settings
from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle
from qcal.compilation.decompositions import decompose_cycle
from qcal.config import Config
from qcal.gates.single_qubit import Id, Meas
from qcal.qpu.qpu import QPU
from qcal.results import Results

logger = logging.getLogger(__name__)


__all__ = ('RandomizedReadout',)


def RandomizedReadout(
    qpu:    QPU,
    config: Config,
    qubits: Sequence[int],
    **kwargs
) -> Callable:
    """Characterize the probability of bit-flips under randomized readout.

    This method requires that the `qpu` object accepts `randomized_readout` as
    a valid runtime kwarg.

    Basic example useage:
    ```
    ro = RandomizedReadout(CustomQPU, config, [0, 1, 2])
    ro.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubits (Sequence[int]): qubits to measure.

    Returns:
        Callable: RandomizedReadout class.
    """


    class RandomizedReadout(qpu):
        """RandomizedReadout class."""

        def __init__(
            self,
            config: Config,
            qubits: Sequence[int],
            **kwargs
        ) -> None:
            """Initialize the RandomizedReadout class within the function."""
            qpu_params = inspect.signature(qpu.__init__).parameters
            accepts_randomized_readout = (
                'randomized_readout' in qpu_params
                or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in qpu_params.values()
                )
            )
            if not accepts_randomized_readout:
                raise ValueError(
                    f"{qpu.__name__} does not accept 'randomized_readout' "
                    "as a valid kwarg. RandomizedReadout requires a QPU "
                    "backend whose __init__ supports randomized readout."
                )

            qpu.__init__(self,
                config=config,
                **kwargs
            )
            self._qubits = sorted(qubits)
            self._probabilities = None

        @property
        def results(self) -> Results:
            """Results object for the RandomizedReadout experiment.

            Returns:
                Results: qcal Results object.
            """
            if self._circuits[0]:
                return self._circuits[0].results
            return Results()

        def generate_circuits(self):
            """Generate the readout calibration circuits."""
            logger.info(' Generating circuits...')

            circuit = Circuit()
            circuit.extend(
                decompose_cycle(Cycle({Id(q) for q in self._qubits}))
            )
            circuit.append(Barrier(self._qubits))
            circuit.append(Cycle({Meas(q) for q in self._qubits}))

            self._circuits = CircuitSet([circuit])

        def analyze(self):
            """Analyze the data."""
            pass

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            qstring = "".join("Q"+str(q) for q in self._qubits)
            self._data_manager._exp_id += (f'_randomized_readout_{qstring}')
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    self._circuits[0].results.probabilities, 'probabilities'
                )

        def plot(self):
            """Plot the results distribution of the experiment."""
            self.results.plot()

        def final(self):
            """Final calibration method."""
            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return RandomizedReadout(
        config=config,
        qubits=qubits,
        **kwargs
    )
