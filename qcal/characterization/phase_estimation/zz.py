"""Submodule for estimating the ZZ phase (i.e. rate) between qubits.

"""
import logging
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from lmfit import Parameters

import qcal.settings as settings
from qcal.characterization.characterize import Characterize
from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle
from qcal.config import Config
from qcal.fitting.fit import FitDecayingCosine
from qcal.gate.gate import Gate
from qcal.gate.single_qubit import X90, Idle, Rz
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

    # Measurement preparation (closes the Ramsey)
    target_meas = {
        'GE': [  # GE pi/2 → maps back to computational basis
            Cycle({X90(p[1], subspace='GE') for p in qubit_pairs}),
        ],
        'GF': [  # EF pi then GE pi/2 → maps back to computational basis
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
            Cycle({X90(p[1], subspace='GE') for p in qubit_pairs}),
        ],
        'EF': [
            Cycle({X90(p[1], subspace='EF') for p in qubit_pairs}),
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
            circuit.extend(target_meas)
            circuit.measure()

            sequence.append(seq_label)
            phases.append(phase)
            circuits.append(circuit)

    circuits['sequence'] = sequence
    circuits['phase'] = phases
    return circuits


def JAZZ(
        qpu:               QPU,
        config:            Config,
        qubit_pairs:       List[Tuple],
        t_max:             float = 5*us,
        detuning:          float = 1*MHz,
        conditional_phase: str = '11',
        n_elements:        int = 50,
        mq_gate:           Gate | None = None,
        params:            Dict | None = None,
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
        qubit_pairs (List[Tuple]): pairs of qubit labels for the two-qubit gate
            calibration.
        t_max (float): maximum free-evolution time. Defaults to 5 µs.
        detuning (float): virtual detuning frequency. Defaults to 1 MHz.
        conditional_phase (str): which ZZ coupling to measure. Must be one of
            '11', '12', '21', '22'. Defaults to '11'.
        n_elements (int): number of time points. Defaults to 50.
        mq_gate (Callable | None): uninstantiated two-qubit gate class (e.g.
            ``CZ``) to use in place of the BIRD echo. Instantiated as
            ``mq_gate(qp)`` per qubit pair. Defaults to None.
        params (Dict | None): config parameter paths to write results to.

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
            qubit_pairs:       List[Tuple],
            t_max:             float = 5*us,
            detuning:          float = 1*MHz,
            conditional_phase: str = '11',
            n_elements:        int = 50,
            mq_gate:           Callable | None = None,
            params:            Dict | None = None,
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

            self._fit = {
                qp: {
                    self._seq_low: FitDecayingCosine(),
                    self._seq_high: FitDecayingCosine(),
                } for qp in qubit_pairs
            }
            self._freq_low = dict.fromkeys(qubit_pairs, False)
            self._freq_high = dict.fromkeys(qubit_pairs, False)

            self._circuits = None

        @property
        def qubit_pairs(self) -> List[Tuple]:
            """Qubit pair labels.

            Returns:
                List[Tuple]: qubit pairs.
            """
            return self._qubits

        @property
        def loss(self) -> Dict:
            """Loss for each qubit pair.

            This property can be used for parameter optimization.

            Returns:
                Dict: loss for each qubit pair.
            """
            return {
                qp: [self._char_values[qp]['val']]
                for qp in self._qubits
                if self._char_values[qp]
            }

        def generate_circuits(self) -> None:
            """Generate all JAZZ circuits."""
            logger.info(' Generating circuits...')

            self._circuits = make_JAZZ_circuits(
                self._qubits, self._times, self._detuning,
                target_subspace=self._ts,
                conditional_subspace=self._cs,
                mq_gate=self._mq_gate,
            )

        def _fit_pair(self, times, prob_low):
            """Build initial Parameters from prob_low for a decaying-cosine fit.
            """
            e = np.array(prob_low).min()
            a = np.array(prob_low).max() - e
            b = np.mean(np.diff(prob_low) / np.diff(times)) / (a if a else 1.)
            params = Parameters()
            params.add('a', value=a)
            params.add('b', value=b)
            params.add('c', value=self._detuning)
            params.add('d', value=0.)
            params.add('e', value=e)
            return params

        def analyze(self) -> None:
            """Analyze the data."""
            logger.info(' Analyzing the data...')

            qubits = sorted(flatten(self._qubits))
            seq_low, seq_high = self._seq_low, self._seq_high

            for qp in self._qubits:
                t_idx = qubits.index(qp[1])

                prob_low = [
                    c.results.marginalize(t_idx).populations['0']
                    for c in self._circuits[
                        self._circuits['sequence'] == seq_low].circuit
                ]
                prob_high = [
                    c.results.marginalize(t_idx).populations['0']
                    for c in self._circuits[
                        self._circuits['sequence'] == seq_high].circuit
                ]

                self._results[qp] = {seq_low: prob_low, seq_high: prob_high}

                params = self._fit_pair(self._times[qp], prob_low)
                self._fit[qp][seq_low].fit(
                    self._times[qp], prob_low, params=params
                )
                self._fit[qp][seq_high].fit(
                    self._times[qp], prob_high, params=params
                )

                if self._fit[qp][seq_low].fit_success:
                    self._freq_low[qp] = (
                        self._fit[qp][seq_low].fit_params['c'].value
                    )
                if self._fit[qp][seq_high].fit_success:
                    self._freq_high[qp] = (
                        self._fit[qp][seq_high].fit_params['c'].value
                    )

                if self._freq_low[qp] and self._freq_high[qp]:
                    val = abs(self._freq_low[qp] - self._freq_high[qp])
                    err = uncertainty_of_sum([
                        self._fit[qp][seq_low].error,
                        self._fit[qp][seq_high].error,
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

        def plot(self) -> None:
            """Plot the frequency sweep and fit results."""
            nrows, ncols = calculate_nrows_ncols(len(self._qubits))
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

            fig.set_tight_layout(True)
            self._save_fig(fig, 'ZZ_characterization')
            plt.show()

        def _draw_ax(self, ax, qp) -> None:
            """Draw a single axes panel."""
            seq_low, seq_high = self._seq_low, self._seq_high
            ax.set_xlabel(r'Time ($\mu$s)', fontsize=15)
            ax.set_ylabel(r'$|0\rangle$ Population', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True)

            state_label_low = seq_low.replace('C', '|') + '⟩'
            state_label_high = seq_high.replace('C', '|') + '⟩'
            ax.plot(
                self._times[qp] / us, self._results[qp][seq_low],
                'o', c='blue', label=rf'Q{qp[0]} {state_label_low}'
            )
            ax.plot(
                self._times[qp] / us, self._results[qp][seq_high],
                'o', c='red', label=rf'Q{qp[0]} {state_label_high}'
            )

            for seq, color in ((seq_low, 'b'), (seq_high, 'r')):
                if self._fit[qp][seq].fit_success:
                    freq = self._fit[qp][seq].fit_params['c'].value
                    x = np.linspace(
                        self._times[qp][0], self._times[qp][-1], 100
                    )
                    ax.plot(
                        x / us, self._fit[qp][seq].predict(x),
                        f'{color}-', label=f'{freq / kHz:.3f} kHz'
                    )

            title = f'{qp} ZZ{self._conditional_phase}'
            if self._char_values[qp]:
                val = self._char_values[qp]['val']
                err = self._char_values[qp]['err']
                title += f': {val / kHz:.3f} ({err / kHz:.3f}) kHz'
            ax.set_title(title)
            ax.legend(loc=0, fontsize=12)

        def _save_fig(self, fig, stem: str) -> None:
            """Save figure in png/pdf/svg if save_data is enabled."""
            if settings.Settings.save_data:
                base = self._data_manager._save_path + stem
                fig.savefig(base + '.png', dpi=300)
                fig.savefig(base + '.pdf')
                fig.savefig(base + '.svg')

        def final(self) -> None:
            """Final calibration method."""
            for qp in self._qubits:
                if self._char_values[qp]:
                    self.set_param(
                        self._params[qp], self._char_values[qp]['val']
                    )
                    print(
                        f"{qp} ZZ{self._conditional_phase}: ZZ = "
                        f"{self._char_values[qp]['val'] / kHz:.3f} "
                        f"({self._char_values[qp]['err'] / kHz:.3f}) kHz"
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
        params=params,
        **kwargs
    )
