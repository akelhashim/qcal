"""Unit tests for qcal.characterization.spectroscopy.sweep.Sweep2D.

As with TwoTone and Resonator (see test_spectroscopy_qubit.py), Sweep2D
.analyze() reads raw per-shot I/Q points via the 'Qn: iq_data'
CircuitSet column, which the Emulator never populates. generate_
circuits() is used only for its scaffolding, and synthetic I/Q data
tracing a Lorentzian dip in |IQ| along whichever sweep axis is marked
'freq' in `params` is hand-injected before calling analyze() directly.

Note: Sweep2D.plot() assumes the 'freq'-named sweep is always sweep2
(it indexes `self._sweep2[q]` with the heatmap's y-tick positions
regardless of which sweep axis `analyze()` actually treated as freq),
so it raises an IndexError whenever 'freq' is in `params[q][0]`
instead. That is a pre-existing bug in sweep.py, out of scope for this
test file; plotting is only exercised here with freq as sweep2, which
is the orientation that works.
"""
import numpy as np

from qcal.backend.emulator import Emulator
from qcal.characterization.spectroscopy.sweep import Sweep2D
from qcal.circuit import Circuit, Cycle
from qcal.gates.single_qubit import Meas, X
from qcal.units import MHz


def _dip_iq(freqs, f0, width, depth, n_shots, rng, noise_std=0.02):
    """Per-frequency synthetic I/Q points tracing a Lorentzian dip."""
    lorentzian = 1.0 / (1.0 + ((freqs - f0) / width) ** 2)
    mag = 1.0 - depth * lorentzian
    iq_data = []
    for m in mag:
        noise = (
            rng.normal(0, noise_std, n_shots)
            + 1j * rng.normal(0, noise_std, n_shots)
        )
        iq_data.append((m + noise).reshape(-1, 1))
    return iq_data


class TestSweep2DConstruction:

    def test_sweeps_are_broadcast_to_all_qubits(self, config):
        sweep1 = np.linspace(0, 1, 3)
        sweep2 = np.linspace(0, 1, 4)
        sw = Sweep2D(
            qpu=Emulator, config=config, qubits=[0, 1],
            sweep1=sweep1, sweep2=sweep2, params={0: ('p1', 'p2')},
        )
        assert sw.sweep1[0] is sw.sweep1[1]
        assert sw.sweep2[0] is sw.sweep2[1]

    def test_generate_circuits_produces_sweep1_by_sweep2_grid(self, config):
        sweep1 = {0: np.linspace(0, 1, 3)}
        sweep2 = {0: np.linspace(0, 1, 4)}
        sw = Sweep2D(
            qpu=Emulator, config=config, qubits=[0],
            sweep1=sweep1, sweep2=sweep2, params={0: ('p1', 'p2')},
        )
        sw.generate_circuits()
        assert sw._circuits.n_circuits == 3 * 4
        assert sw._circuits[0] == Circuit([Cycle({Meas(0)})])

    def test_generate_circuits_prepends_cycle_prepulse(self, config):
        sweep1 = {0: np.linspace(0, 1, 2)}
        sweep2 = {0: np.linspace(0, 1, 2)}
        sw = Sweep2D(
            qpu=Emulator, config=config, qubits=[0],
            sweep1=sweep1, sweep2=sweep2, params={0: ('p1', 'p2')},
            prepulse=Cycle({X(0)}),
        )
        sw.generate_circuits()
        assert sw._circuits[0] == Circuit([Cycle({X(0)}), Cycle({Meas(0)})])


class TestSweep2DSyntheticAnalysis:

    def test_analyze_with_freq_as_sweep2(self, config):
        # 'freq' in params[q][1] -> analyze() loops over sweep2 (freq)
        # as the outer/row dimension.
        f0 = config['readout/0/freq']
        sweep1 = {0: np.linspace(0.0, 1.0, 4)}
        sweep2 = {0: np.linspace(-10, 10, 41) * MHz + f0}
        sw = Sweep2D(
            qpu=Emulator, config=config, qubits=[0],
            sweep1=sweep1, sweep2=sweep2,
            params={0: ('some/amp/param', 'readout/0/freq')},
        )
        sw.generate_circuits()

        rng = np.random.default_rng(0)
        iq_col = []
        for s1 in sweep1[0]:
            depth = 0.3 + 0.3 * s1
            iq_col.extend(
                _dip_iq(
                    sweep2[0], f0, width=1.5 * MHz, depth=depth,
                    n_shots=30, rng=rng,
                )
            )
        sw._circuits['Q0: iq_data'] = iq_col
        sw.analyze()

        assert sw._mag[0].shape == (41, 4)
        assert sw._phase[0].shape == (41, 4)

    def test_analyze_with_freq_as_sweep1(self, config):
        # 'freq' in params[q][0] -> analyze() loops over sweep1 (freq)
        # as the outer/row dimension instead.
        f0 = config['readout/0/freq']
        sweep1 = {0: np.linspace(-10, 10, 41) * MHz + f0}
        sweep2 = {0: np.linspace(0.0, 1.0, 4)}
        sw = Sweep2D(
            qpu=Emulator, config=config, qubits=[0],
            sweep1=sweep1, sweep2=sweep2,
            params={0: ('readout/0/freq', 'some/other/param')},
        )
        sw.generate_circuits()

        rng = np.random.default_rng(1)
        iq_col = []
        for s2 in sweep2[0]:
            depth = 0.3 + 0.3 * s2
            iq_col.extend(
                _dip_iq(
                    sweep1[0], f0, width=1.5 * MHz, depth=depth,
                    n_shots=30, rng=rng,
                )
            )
        sw._circuits['Q0: iq_data'] = iq_col
        sw.analyze()

        assert sw._mag[0].shape == (41, 4)
        assert sw._phase[0].shape == (41, 4)


class TestSweep2DPlotting:

    def test_plot_runs_without_error_with_freq_as_sweep2(self, config):
        f0 = config['readout/0/freq']
        sweep1 = {0: np.linspace(0.0, 1.0, 3)}
        sweep2 = {0: np.linspace(-10, 10, 21) * MHz + f0}
        sw = Sweep2D(
            qpu=Emulator, config=config, qubits=[0],
            sweep1=sweep1, sweep2=sweep2,
            params={0: ('some/amp/param', 'readout/0/freq')},
        )
        sw.generate_circuits()

        rng = np.random.default_rng(2)
        iq_col = []
        for _ in sweep1[0]:
            iq_col.extend(
                _dip_iq(
                    sweep2[0], f0, width=1.5 * MHz, depth=0.5,
                    n_shots=10, rng=rng,
                )
            )
        sw._circuits['Q0: iq_data'] = iq_col
        sw.analyze()
        sw.plot()
