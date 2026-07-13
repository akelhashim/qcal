"""Submodule for estimating the ZZ phase (i.e. rate) between qubits.

"""
import logging
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output
from lmfit import Parameters
from numpy.typing import NDArray
from plotly.subplots import make_subplots

import qcal.settings as settings
from qcal.characterization.characterize import Characterize
from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle
from qcal.config import Config
from qcal.fitting.fit import FitDecayingCosine
from qcal.fitting.utils import est_freq_fft
from qcal.gate.gate import Gate
from qcal.gate.single_qubit import X90, Y90, Idle, Rz
from qcal.math.utils import round_to_order_error, uncertainty_of_sum
from qcal.plotting.utils import calculate_nrows_ncols
from qcal.qpu.qpu import QPU
from qcal.units import MHz, kHz, us
from qcal.utils import flatten, save_init

logger = logging.getLogger(__name__)


# Maps conditional_phase -> (conditional_subspace, target_subspace)
_PHASE_TO_SUBSPACES = {
    '11': ('GE', 'GE'),
    '21': ('GF', 'GE'),
    '12': ('GE', 'GF'),
    '22': ('GF', 'GF'),
}
_SEQ_LABELS = {'GE': ('C0', 'C1'), 'GF': ('C0', 'C2'), 'EF': ('C1', 'C2')}


def make_JAZZ_circuits(
    qubit_pairs:          List[Tuple[int, int]],
    times:                Dict,
    detuning:             float,
    target_subspace:      str = 'GE',
    conditional_subspace: str = 'GE',
    mq_gate:              Callable | None = None,
) -> CircuitSet:
    """Build JAZZ circuits for a single subspace combination.

    Args:
        qubit_pairs (List[Tuple[int, int]]): list of (control, target) qubit
            pairs.
        times (Dict): dict mapping qubit pair -> time array.
        detuning (float): virtual detuning frequency in Hz.
        target_subspace (str): subspace of the target qubit Ramsey ('GE',
            'GF', or 'EF').
        conditional_subspace (str): subspace used to define the two
            conditional control states ('GE', 'GF', or 'EF').
        mq_gate (Callable | None): uninstantiated two-qubit gate class (e.g.
            ``CZ``). When provided it replaces the BIRD echo — the gate is
            instantiated as ``mq_gate(qp)`` for each qubit pair. Defaults to
            ``None`` (use BIRD echo).

    Returns:
        CircuitSet: circuits with 'sequence' and 'phase' metadata columns.
    """
    all_qubits = sorted(flatten(qubit_pairs))

    # --- State-prep pulses per subspace ----------------------------------

    # Control state preparation cycles (each entry = one Cycle)
    # low state: |0> for GE/GF, |1> for EF
    # high state: |1> for GE, |2> for GF/EF
    ctrl_low = {
        'GE': [], # |0>
        'GF': [], # |0>
        'EF': [   # |1> (GE pi)
            Cycle({X90(p[0], subspace='GE') for p in qubit_pairs}),
            Cycle({X90(p[0], subspace='GE') for p in qubit_pairs}),
        ],
    }[conditional_subspace]

    ctrl_high = {
        'GE': [  # |1> (GE pi)
            Cycle({X90(p[0], subspace='GE') for p in qubit_pairs}),
            Cycle({X90(p[0], subspace='GE') for p in qubit_pairs}),
        ],
        'GF': [  # |2> (GE pi then GF pi)
            Cycle({X90(p[0], subspace='GE') for p in qubit_pairs}),
            Cycle({X90(p[0], subspace='GE') for p in qubit_pairs}),
            Cycle({X90(p[0], subspace='EF') for p in qubit_pairs}),
            Cycle({X90(p[0], subspace='EF') for p in qubit_pairs}),
        ],
        'EF': [  # |2> (GE pi then EF pi)
            Cycle({X90(p[0], subspace='GE') for p in qubit_pairs}),
            Cycle({X90(p[0], subspace='GE') for p in qubit_pairs}),
            Cycle({X90(p[0], subspace='EF') for p in qubit_pairs}),
            Cycle({X90(p[0], subspace='EF') for p in qubit_pairs}),
        ],
    }[conditional_subspace]

    # Target state preparation (creates superposition in target_subspace)
    target_prep = {
        'GE': [  # GE pi/2 → |0>+|1> superposition
            Cycle({X90(p[1], subspace='GE') for p in qubit_pairs}),
        ],
        'GF': [  # GE pi/2 then EF pi → |0>+|2> superposition
            Cycle({X90(p[1], subspace='GE') for p in qubit_pairs}),
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
        ],
        'EF': [  # GE pi then EF pi/2 → |1>+|2> superposition
            Cycle({X90(p[1], subspace='GE') for p in qubit_pairs}),
            Cycle({X90(p[1], subspace='GE') for p in qubit_pairs}),
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
        ],
    }[target_subspace]

    # Measurement preparation (closes the Ramsey).
    # X90 closing pulse → in-phase (I); Y90 closing pulse → out-of-phase (Q).
    target_meas_x = {
        'GE': [
            Cycle({X90(p[1], subspace='GE') for p in qubit_pairs}),
        ],
        'GF': [
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
            Cycle({X90(p[1], subspace='GE') for p in qubit_pairs}),
        ],
        'EF': [
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
        ],
    }[target_subspace]
    target_meas_y = {
        'GE': [
            Cycle({Y90(p[1], subspace='GE') for p in qubit_pairs}),
        ],
        'GF': [
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
            Cycle({Y90(p[1], subspace='GE') for p in qubit_pairs}),
        ],
        'EF': [
            Cycle({Y90(p[1], subspace='EF') for p in qubit_pairs}),
        ],
    }[target_subspace]

    # Subspace for the virtual-detuning Rz on the target qubit
    rz_subspace = 'EF' if target_subspace in ('GF', 'EF') else 'GE'

    # --- Echo cycles (BIRD-style: refocuses 1-body Z, preserves ZZ) ------
    qp = qubit_pairs
    ts, cs = target_subspace, conditional_subspace
    if  cs == 'GE' and ts == 'GE':
        echo = [
            Cycle({X90(q, 'GE') for q in all_qubits}),
            Cycle({X90(q, 'GE') for q in all_qubits}),
        ]
    elif cs == 'GE' and ts == 'GF':
        echo = [
            Cycle({X90(q, 'GE') for q in all_qubits}),
            Cycle({X90(q, 'GE') for q in all_qubits}),
            Cycle({X90(p[1], 'EF') for p in qp}),
            Cycle({X90(p[1], 'EF') for p in qp}),
            Cycle({X90(p[1], 'GE') for p in qp}),
            Cycle({X90(p[1], 'GE') for p in qp}),
        ]
    elif cs == 'GE' and ts == 'EF':
        echo = [
            Cycle({X90(p[0], 'GE') for p in qp} | {X90(p[1], 'EF') for p in qp}),
            Cycle({X90(p[0], 'GE') for p in qp} | {X90(p[1], 'EF') for p in qp}),
        ]
    elif cs == 'GF' and ts == 'GE':
        echo = [
            Cycle({X90(q, 'GE') for q in all_qubits}),
            Cycle({X90(q, 'GE') for q in all_qubits}),
            Cycle({X90(p[0], 'EF') for p in qp}),
            Cycle({X90(p[0], 'EF') for p in qp}),
            Cycle({X90(p[0], 'GE') for p in qp}),
            Cycle({X90(p[0], 'GE') for p in qp}),
        ]
    elif cs == 'GF' and ts == 'GF':
        echo = [
            Cycle({X90(q, 'GE') for q in all_qubits}),
            Cycle({X90(q, 'GE') for q in all_qubits}),
            Cycle({X90(q, 'EF') for q in all_qubits}),
            Cycle({X90(q, 'EF') for q in all_qubits}),
            Cycle({X90(q, 'GE') for q in all_qubits}),
            Cycle({X90(q, 'GE') for q in all_qubits}),
        ]
    elif cs == 'GF' and ts == 'EF':
        echo = [
            Cycle({X90(p[0], 'GE') for p in qp} | {X90(p[1], 'EF') for p in qp}),
            Cycle({X90(p[0], 'GE') for p in qp} | {X90(p[1], 'EF') for p in qp}),
            Cycle({X90(p[0], 'EF') for p in qp}),
            Cycle({X90(p[0], 'EF') for p in qp}),
            Cycle({X90(p[0], 'GE') for p in qp}),
            Cycle({X90(p[0], 'GE') for p in qp}),
        ]
    elif cs == 'EF' and ts == 'GE':
        echo = [
            Cycle({X90(p[0], 'EF') for p in qp} | {X90(p[1], 'GE') for p in qp}),
            Cycle({X90(p[0], 'EF') for p in qp} | {X90(p[1], 'GE') for p in qp}),
        ]
    elif cs == 'EF' and ts == 'GF':
        echo = [
            Cycle({X90(p[1], 'GE') for p in qp} | {X90(p[0], 'EF') for p in qp}),
            Cycle({X90(p[1], 'GE') for p in qp} | {X90(p[0], 'EF') for p in qp}),
            Cycle({X90(p[1], 'EF') for p in qp}),
            Cycle({X90(p[1], 'EF') for p in qp}),
            Cycle({X90(p[1], 'GE') for p in qp}),
            Cycle({X90(p[1], 'GE') for p in qp}),
        ]
    elif cs == 'EF' and ts == 'EF':
        echo = [
            Cycle({X90(q, 'EF') for q in all_qubits}),
            Cycle({X90(q, 'EF') for q in all_qubits}),
        ]

    if mq_gate is not None:
        echo = [Cycle({mq_gate(qp) for qp in qubit_pairs})]

    # --- Build circuits --------------------------------------------------
    seq_low, seq_high = _SEQ_LABELS[conditional_subspace]
    circuits = CircuitSet()
    sequence = []
    phases = []

    for t in times[qubit_pairs[0]]:
        phase = 2. * np.pi * detuning * t

        for ctrl_cycles, seq_label in (
            (ctrl_low, seq_low), (ctrl_high, seq_high)
        ):
            for tmeas, quad_suffix in (
                (target_meas_x, ' X90'), (target_meas_y, ' Y90')
            ):
                circuit = Circuit()

                if ctrl_cycles:
                    circuit.extend(ctrl_cycles)
                    # circuit.append(Barrier(all_qubits))
                    circuit.join(Circuit(target_prep))
                else:
                    circuit.extend(target_prep)
                circuit.append(Barrier(all_qubits))

                circuit.extend([
                    Cycle({Idle(q, duration=t/2) for q in all_qubits}),
                    Barrier(all_qubits),
                ])

                if mq_gate is not None:
                    circuit.extend([
                        Cycle({mq_gate(qp) for qp in qubit_pairs}),
                        Barrier(all_qubits),
                    ])

                circuit.extend(echo)
                circuit.append(Barrier(all_qubits))

                if mq_gate is not None:
                    circuit.extend([
                        Cycle({mq_gate(qp) for qp in qubit_pairs}),
                        Barrier(all_qubits),
                    ])

                circuit.extend([
                    Cycle({Idle(q, duration=t/2) for q in all_qubits}),
                    Cycle({
                        Rz(p[1], phase, subspace=rz_subspace)
                        for p in qubit_pairs
                    }),
                    Barrier(all_qubits),
                ])
                circuit.extend(tmeas)
                circuit.measure()

                sequence.append(f'{seq_label}{quad_suffix}')
                phases.append(phase)
                circuits.append(circuit)

    circuits['sequence'] = sequence
    circuits['phase'] = phases
    return circuits


def JAZZ(
        qpu:               QPU,
        config:            Config,
        qubit_pairs:       List[Tuple[int, int]],
        t_max:             float = 5 * us,
        detuning:          float = 1 * MHz,
        conditional_phase: str = '11',
        n_elements:        int = 50,
        mq_gate:           Gate | None = None,
        mq_gate_params:    Dict[Tuple[int, int], str] | str | None = None,
        params:            Dict[Tuple[int, int], str] | str | None = None,
        **kwargs
    ) -> Callable:
    """Joint Amplification of ZZ (JAZZ).

    This characterization measures the ZZ rotation angle/phase between qubits.
    To do so, we perform a Ramsey on the target qubit when the control qubit is
    in |0> and |1>, while echoing both qubits with a pi pulse in the middle. The
    difference in frequency between the two Ramsey experiments corresponds to
    the ZZ coupling strength between the two qubits.

    JAZZ is based on Bilinear Rotational Decoupling (BIRD) sequences:
    https://www.sciencedirect.com/science/article/pii/0009261482832296

    The ``conditional_phase`` argument selects which ZZ coupling to measure:

    * ``'11'`` — conditional GE, target GE (control in |0⟩ vs |1⟩)
    * ``'21'`` — conditional GF, target GE (control in |0⟩ vs |2⟩)
    * ``'12'`` — conditional GE, target GF (control in |0⟩ vs |1⟩)
    * ``'22'`` — conditional GF, target GF (control in |0⟩ vs |2⟩)

    Basic example usage:
    ```
    cal = JAZZ(
        CustomQPU,
        config,
        qubit_pairs=[(0, 1), (2, 3)])
    cal.run()
    ```

    Args:
        qpu (QPU): custom QPU object.
        config (Config): qcal Config object.
        qubit_pairs (List[Tuple[int, int]]): pairs of qubit labels for the
            ZZ characterization (e.g. [(0, 1), (2, 3)]).
        t_max (float): maximum free-evolution time. Defaults to 5 µs.
        detuning (float): virtual detuning frequency. Defaults to 1 MHz.
        conditional_phase (str): which ZZ coupling to measure. Must be one of
            '11', '12', '21', '22'. Defaults to '11'.
        n_elements (int): number of time points. Defaults to 50.
        mq_gate (Callable | None): uninstantiated two-qubit gate class (e.g.
            ``CZ``) to use in place of the BIRD echo. Instantiated as
            ``mq_gate(qp)`` per qubit pair. Defaults to None.
        mq_gate_params (Dict[Tuple[int, int], str] | str | None): if using
            ``mq_gate``, the config parameter paths for the two-qubit gate time.
            Defaults to None.
        params (Dict[Tuple[int, int], str] | str | None): config parameter paths
            to write the ZZ results to. Defaults to None, which uses
            'two_qubit/{qp}/ZZ{conditional_phase}'.

    Returns:
        Callable: JAZZ characterization class.
    """

    class JAZZ(qpu, Characterize):
        """JAZZ characterization class.

        This class inherits a custom QPU from the JAZZ characterization
        function.
        """

        @save_init
        def __init__(
            self,
            config:            Config,
            qubit_pairs:       List[Tuple[int, int]],
            t_max:             float = 5 * us,
            detuning:          float = 1 * MHz,
            conditional_phase: str = '11',
            n_elements:        int = 50,
            mq_gate:           Gate | None = None,
            mq_gate_params:    Dict[Tuple[int, int], str] | str | None = None,
            params:            Dict[Tuple[int, int], str] | str | None = None,
            **kwargs
        ) -> None:
            """Initialize the JAZZ class within the function."""
            qpu.__init__(self, config=config, **kwargs)
            Characterize.__init__(self, config)

            if conditional_phase not in _PHASE_TO_SUBSPACES:
                raise ValueError(
                    f"'conditional_phase' must be one of "
                    f"{list(_PHASE_TO_SUBSPACES)}!"
                )

            self._qubits = qubit_pairs
            self._detuning = detuning
            self._conditional_phase = conditional_phase
            self._mq_gate = mq_gate

            self._cs, self._ts = _PHASE_TO_SUBSPACES[conditional_phase]
            self._seq_low, self._seq_high = _SEQ_LABELS[self._cs]
            self._seq_low_x  = f'{self._seq_low} X90'
            self._seq_low_y  = f'{self._seq_low} Y90'
            self._seq_high_x = f'{self._seq_high} X90'
            self._seq_high_y = f'{self._seq_high} Y90'

            self._times = {
                q: np.linspace(0., t_max, n_elements) for q in qubit_pairs
            }
            self._param_sweep = self._times

            if params:
                self._params = params
            else:
                self._params = {
                    qp: f'two_qubit/{qp}/ZZ{conditional_phase}'
                    for qp in qubit_pairs
                }

            if mq_gate and mq_gate_params:
                self._mq_gate_params = mq_gate_params
            elif mq_gate and not mq_gate_params:
                self._mq_gate_params = {
                    qp: [
                        f'two_qubit/{qp}/pulse/0/time',
                        f'two_qubit/{qp}/pulse/1/time'
                    ]
                    for qp in qubit_pairs
                }

            self._fit = {
                qp: {
                    self._seq_low_x: FitDecayingCosine(),
                    self._seq_high_x: FitDecayingCosine(),
                } for qp in qubit_pairs
            }
            self._freq_low = dict.fromkeys(qubit_pairs, False)
            self._freq_high = dict.fromkeys(qubit_pairs, False)

            self._circuits = None

        @property
        def loss(self) -> Dict[Tuple[int, int], float]:
            """Loss for each qubit pair.

            This property can be used for parameter optimization.

            Returns:
                Dict[Tuple[int, int], float]: loss for each qubit pair.
            """
            return {
                qp: [self._char_values[qp]['val']]
                for qp in self._qubits
                if self._char_values[qp]
            }

        @property
        def qubit_pairs(self) -> Sequence[Tuple[int, int]]:
            """Qubit pair labels.

            Returns:
                Sequence[Tuple[int, int]]: qubit pairs.
            """
            return self._qubits

        @property
        def times(self) -> Dict[Tuple[int, int], NDArray]:
            """Time arrays for each qubit pair.

            Returns:
                Dict[Tuple[int, int], NDArray]: time arrays keyed by qubit pair.
            """
            return self._times

        def generate_circuits(self) -> None:
            """Generate all JAZZ circuits."""
            logger.info(' Generating circuits...')

            self._circuits = make_JAZZ_circuits(
                self._qubits, self._times, self._detuning,
                target_subspace=self._ts,
                conditional_subspace=self._cs,
                mq_gate=self._mq_gate,
            )

            if self._mq_gate and self._mq_gate_params:
                for qp in self._qubits:
                    for param in self._mq_gate_params[qp]:
                        self._circuits[f'param: {param}'] = np.repeat(
                            self._times[qp], 4
                        ) / 2 # divide by 2 since there are 2 gates per circuit

        def _fit_pair(
            self,
            times: NDArray,
            prob: List[float],
            freq_hint: float | None = None,
        ) -> Parameters:
            """Build initial Parameters from prob for a decaying-cosine fit."""
            est_freq = (
                freq_hint if freq_hint is not None
                else est_freq_fft(times, prob)
            )
            e = np.array(prob).min()
            a = np.array(prob).max() - e
            b = np.mean(np.diff(prob) / np.diff(times)) / (a if a else 1.)
            params = Parameters()
            params.add('a', value=a)
            params.add('b', value=b)
            params.add('c', value=est_freq)
            params.add('d', value=0.)
            params.add('e', value=e)
            return params

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            qubits = sorted(flatten(self._qubits))
            seq_low_x,  seq_low_y  = self._seq_low_x,  self._seq_low_y
            seq_high_x, seq_high_y = self._seq_high_x, self._seq_high_y

            for qp in self._qubits:
                t_idx = qubits.index(qp[1])

                I_low = [
                    c.results.marginalize(t_idx).populations['0']
                    for c in self._circuits.subset(sequence=seq_low_x).circuit
                ]
                Q_low = [
                    c.results.marginalize(t_idx).populations['0']
                    for c in self._circuits.subset(sequence=seq_low_y).circuit
                ]
                I_high = [
                    c.results.marginalize(t_idx).populations['0']
                    for c in self._circuits.subset(sequence=seq_high_x).circuit
                ]
                Q_high = [
                    c.results.marginalize(t_idx).populations['0']
                    for c in self._circuits.subset(sequence=seq_high_y).circuit
                ]

                self._results[qp] = {
                    seq_low_x:  I_low,  seq_low_y:  Q_low,
                    seq_high_x: I_high, seq_high_y: Q_high,
                }

                # Use complex quadrature signal to resolve frequency sign.
                z_low  = np.array(I_low)  - 1j * np.array(Q_low)
                z_high = np.array(I_high) - 1j * np.array(Q_high)
                signed_freq_low  = est_freq_fft(self._times[qp], z_low)
                signed_freq_high = est_freq_fft(self._times[qp], z_high)

                weights = np.full(
                    len(self._times[qp]),
                    1.0 / np.sqrt(self._n_shots),
                )
                params_low = self._fit_pair(
                    self._times[qp], I_low, freq_hint=signed_freq_low
                )
                self._fit[qp][seq_low_x].fit(
                    self._times[qp], I_low, params=params_low, weights=weights
                )
                params_high = self._fit_pair(
                    self._times[qp], I_high, freq_hint=signed_freq_high
                )
                self._fit[qp][seq_high_x].fit(
                    self._times[qp], I_high, params=params_high, weights=weights
                )

                if self._fit[qp][seq_low_x].fit_success:
                    fit_freq = self._fit[qp][seq_low_x].fit_params['c'].value
                    self._freq_low[qp] = (
                        np.sign(signed_freq_low) * abs(fit_freq)
                    )
                else:
                    self._freq_low[qp] = signed_freq_low

                if self._fit[qp][seq_high_x].fit_success:
                    fit_freq = self._fit[qp][seq_high_x].fit_params['c'].value
                    self._freq_high[qp] = (
                        np.sign(signed_freq_high) * abs(fit_freq)
                    )
                else:
                    self._freq_high[qp] = signed_freq_high

                if (
                    self._freq_low[qp] is not False
                    and self._freq_high[qp] is not False
                ):
                    val = self._freq_high[qp] - self._freq_low[qp]
                    err = uncertainty_of_sum([
                        self._fit[qp][seq_low_x].fit_params['c'].stderr or 0.,
                        self._fit[qp][seq_high_x].fit_params['c'].stderr or 0.,
                    ])
                    val, err = round_to_order_error(val, err)
                    self._char_values[qp] = {'val': val, 'err': err}

        def save(self):
            """Save all circuits and data."""
            clear_output(wait=True)
            self._data_manager._exp_id += (
                f'_JAZZ_{"".join("Q"+str(q) for q in self._qubits)}'
            )
            if settings.Settings.save_data:
                qpu.save(self)
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._results]), 'sweep_results'
                )
                self._data_manager.save_to_csv(
                    pd.DataFrame([self._char_values]), 'characterized_values'
                )

        def _draw_ax(self, ax, qp) -> None:
            """Draw a single axes panel."""
            seq_low_x, seq_high_x = self._seq_low_x, self._seq_high_x
            unit, unit_str = (MHz, 'MHz') if self._mq_gate else (kHz, 'kHz')

            ax.set_xlabel(r'Time ($\mu$s)', fontsize=15)
            ax.set_ylabel(r'$|0\rangle$ Population', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True)

            state_label_low  = self._seq_low.replace('C', '|')  + '⟩'
            state_label_high = self._seq_high.replace('C', '|') + '⟩'
            ax.plot(
                self._times[qp] / us, self._results[qp][seq_low_x],
                'o', c='blue', label=rf'Q{qp[0]} {state_label_low} (I)'
            )
            ax.plot(
                self._times[qp] / us, self._results[qp][seq_high_x],
                'o', c='red', label=rf'Q{qp[0]} {state_label_high} (I)'
            )

            for seq, color in ((seq_low_x, 'b'), (seq_high_x, 'r')):
                if self._fit[qp][seq].fit_success:
                    freq = self._fit[qp][seq].fit_params['c'].value
                    x = np.linspace(
                        self._times[qp][0], self._times[qp][-1], 100
                    )
                    ax.plot(
                        x / us, self._fit[qp][seq].predict(x),
                        f'{color}-', label=f'{freq / unit:.1f} {unit_str}'
                    )

            title = f'{qp} ZZ{self._conditional_phase}'
            if self._char_values[qp]:
                val = self._char_values[qp]['val']
                err = self._char_values[qp]['err']
                title += f': {val / unit:.1f} ({err / unit:.1f}) {unit_str}'
            ax.set_title(title)
            ax.legend(loc=0, fontsize=12)

        def plot(self) -> None:
            """Plot the frequency sweep and fit results."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubits))
            unit, unit_str = (MHz, 'MHz') if self._mq_gate else (kHz, 'kHz')

            seq_low_x, seq_high_x = self._seq_low_x, self._seq_high_x
            state_label_low  = self._seq_low.replace('C', '|') + '⟩'
            state_label_high = self._seq_high.replace('C', '|') + '⟩'

            # --- subplot titles ---
            subplot_titles = []
            for qp in self._qubits:
                title = f'{qp} ZZ{self._conditional_phase}'
                if self._char_values[qp]:
                    val = self._char_values[qp]['val']
                    err = self._char_values[qp]['err']
                    title += f' = {val/unit:.1f} ({err/unit:.1f}) {unit_str}'
                subplot_titles.append(title)

            pfig_height = 350 * nrows
            pfig_width  = 350 * ncols + 50
            pfig_gap_px = 80
            pfig_margin = {'t': 60, 'b': 60, 'l': 60, 'r': 30}
            vertical_spacing = (
                0.0 if nrows <= 1
                else min(0.2, pfig_gap_px / pfig_height)
            )
            horizontal_spacing = (
                0.0 if ncols <= 1
                else min(0.2, pfig_gap_px / pfig_width)
            )

            pfig = make_subplots(
                rows=nrows, cols=ncols,
                subplot_titles=subplot_titles,
                vertical_spacing=vertical_spacing,
                horizontal_spacing=horizontal_spacing,
            )
            pfig.update_annotations(font_size=12, yshift=5)

            _leg_x = pfig.layout.xaxis.domain[0] + 0.01
            _leg_y = pfig.layout.yaxis.domain[1] - 0.01

            blue, red = '#1f77b4', '#d62728'

            for k, qp in enumerate(self._qubits):
                row = (k // ncols) + 1
                col = (k % ncols) + 1
                times_us = self._times[qp] / us

                pfig.add_trace(
                    go.Scatter(
                        x=times_us,
                        y=self._results[qp][seq_low_x],
                        mode='markers',
                        marker={'size': 6, 'color': blue},
                        name=f'Q{qp[0]} {state_label_low} (I)',
                        showlegend=(k == 0),
                    ),
                    row=row, col=col,
                )
                pfig.add_trace(
                    go.Scatter(
                        x=times_us,
                        y=self._results[qp][seq_high_x],
                        mode='markers',
                        marker={'size': 6, 'color': red},
                        name=f'Q{qp[0]} {state_label_high} (I)',
                        showlegend=(k == 0),
                    ),
                    row=row, col=col,
                )

                x_fit = np.linspace(
                    self._times[qp][0], self._times[qp][-1], 200
                )
                x_fit_us = x_fit / us

                for seq, color in ((seq_low_x, blue), (seq_high_x, red)):
                    if self._fit[qp][seq].fit_success:
                        freq = self._fit[qp][seq].fit_params['c'].value
                        pfig.add_trace(
                            go.Scatter(
                                x=x_fit_us,
                                y=self._fit[qp][seq].predict(x_fit),
                                mode='lines',
                                line={'color': color, 'width': 2},
                                name=f'{freq / unit:.1f} {unit_str}',
                                showlegend=(k == 0),
                            ),
                            row=row, col=col,
                        )

                pfig.update_xaxes(
                    title_text='Time (µs)' if row == nrows else '',
                    automargin=True, showgrid=True,
                    title_standoff=10, showticklabels=True,
                    row=row, col=col,
                )
                pfig.update_yaxes(
                    title_text='|0⟩ Population' if col == 1 else '',
                    automargin=True, showgrid=True,
                    title_standoff=10, showticklabels=True,
                    row=row, col=col,
                )

            pfig.update_layout(
                height=pfig_height,
                width=pfig_width,
                margin=pfig_margin,
                legend={
                    'orientation': 'v',
                    'xanchor': 'left', 'x': _leg_x,
                    'yanchor': 'top',   'y': _leg_y,
                    'bgcolor': 'rgba(255,255,255,0.85)',
                    'bordercolor': '#c7c7c7',
                    'borderwidth': 1,
                },
                template='plotly_white',
                paper_bgcolor='white',
                plot_bgcolor='#fbfbfd',
            )
            pfig.update_xaxes(
                showline=True, mirror=True,
                linecolor='#c7c7c7', linewidth=1,
                gridcolor='#e5e7eb', zeroline=False, ticks='outside',
                side='bottom', title_standoff=10,
            )
            pfig.update_yaxes(
                showline=True, mirror=True,
                linecolor='#c7c7c7', linewidth=1,
                gridcolor='#e5e7eb', zeroline=False, ticks='outside',
                side='left', title_standoff=10,
            )
            pfig.show(config={
                'toImageButtonOptions': {
                    'format': 'png', 'filename': 'JAZZ', 'scale': 10,
                }
            })

            if settings.Settings.save_data:
                figsize = (5 * ncols, 4 * nrows)
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=figsize, layout='constrained'
                )
                k = -1
                for i in range(nrows):
                    for j in range(ncols):
                        k += 1
                        if len(self._qubits) == 1:
                            ax = axes
                        elif axes.ndim == 1:
                            ax = axes[j]
                        else:
                            ax = axes[i, j]
                        if k < len(self._qubits):
                            self._draw_ax(ax, self._qubits[k])
                        else:
                            ax.axis('off')
                base = self._data_manager._save_path + 'ZZ_characterization'
                fig.savefig(base + '.png', dpi=300)
                fig.savefig(base + '.pdf')
                fig.savefig(base + '.svg')
                plt.close(fig)

        def final(self) -> None:
            """Final calibration method."""
            for qp in self._qubits:
                if self._char_values[qp]:
                    self.set_param(
                        self._params[qp], self._char_values[qp]['val']
                    )
                    unit, unit_str = (MHz, 'MHz') if self._mq_gate else (
                        kHz, 'kHz'
                    )
                    print(
                        f"{qp} ZZ{self._conditional_phase}: ZZ = "
                        f"{self._char_values[qp]['val']/unit:.3f} "
                        f"({self._char_values[qp]['err']/unit:.3f}) {unit_str}"
                    )

            if settings.Settings.save_data:
                self._config.save()

            print(f"\nRuntime: {repr(self._runtime)[8:]}\n")

        def run(self):
            """Run all experimental methods and analyze results."""
            self.generate_circuits()
            qpu.run(self, self._circuits, save=False)
            self.analyze()
            self.save()
            self.plot()
            self.final()

    return JAZZ(
        config=config,
        qubit_pairs=qubit_pairs,
        t_max=t_max,
        detuning=detuning,
        conditional_phase=conditional_phase,
        n_elements=n_elements,
        mq_gate=mq_gate,
        mq_gate_params=mq_gate_params,
        params=params,
        **kwargs
    )
