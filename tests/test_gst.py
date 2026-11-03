"""Unit tests for qcal.characterization.tomography.gst (GST, SingleQubitGST,
TwoQubitGST).

GST circuits are built from native pyGSTi `Circuit` objects, not qcal
`Circuit`s -- `GST.generate_circuits()` stores
`self._circuits = CircuitSet(list(self._edesign.all_circuits_needing_data))`
directly. The pyGSTi -> qcal conversion only happens inside
`qpu.run()` (via the `PyGSTiTranspiler` wired in as `self._transpiler`
by `GST.__init__`), which `GST.run()` calls internally. `GST.run()`
also calls `self.plot()` at the end, which needs `cvxpy` (declared via
`pyproject.toml`'s `pygsti[diamond-norm]` extra) for its diamond-norm
metrics -- harmless here since conftest.py's autouse fixture forces
non-interactive matplotlib/plotly renderers.

`StateVectorSimulator` with `n_shots=None` runs exact-probability (no
shot-noise) simulation, so a noiseless run should recover the target
gates to within floating-point precision -- there is no need to seed
`np.random` or tolerate statistical noise.

Circuit depths are kept to (1, 2) throughout (per the module's
exponential-spacing requirement, any 2-element sequence trivially
qualifies) to keep circuit counts -- and therefore simulation time --
small. Two-qubit GST uses modes=('Target',) only, since 'Target' just
copies the ideal gates into the estimate with no actual GST fit/gauge
optimization, which is both the cheapest possible check that the full
2-qubit pipeline (circuit generation, gate-name mapping e.g.
Gcphase->CZ, dataset construction, protocol execution) runs correctly,
and avoids the much more expensive classical post-processing that a
real 2-qubit fit (e.g. 'full TP') would require.
"""
import pytest
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYCPHASE

from qcal.backend.emulator import Emulator
from qcal.characterization.tomography.gst import GST, SingleQubitGST, TwoQubitGST
from qcal.simulation.simulators import StateVectorSimulator


def _1q_gst(config, **kwargs):
    """A GST instance for qubit 0 using the smq1Q_XYI modelpack."""
    return GST(
        qpu=Emulator, config=config, qubit_labels=[0],
        pspec=smq1Q_XYI.processor_spec([0]),
        target_model=smq1Q_XYI.target_model(qubit_labels=[0]),
        prep_fiducials=smq1Q_XYI.prep_fiducials([0]),
        meas_fiducials=smq1Q_XYI.meas_fiducials([0]),
        germs=smq1Q_XYI.germs([0]),
        circuit_depths=(1, 2),
        **kwargs,
    )


class TestGSTConstruction:

    def test_non_exponentially_spaced_depths_raises(self, config):
        with pytest.raises(ValueError):
            GST(
                qpu=Emulator, config=config, qubit_labels=[0],
                circuit_depths=(1, 2, 5),
            )

    def test_two_point_depths_are_always_accepted(self, config):
        # Any 2-element sequence is trivially "exponentially spaced",
        # since np.geomspace(a, b, 2) == [a, b] regardless of a, b.
        GST(qpu=Emulator, config=config, qubit_labels=[0], circuit_depths=(1, 2))

    def test_properties_after_construction(self, config):
        gst = _1q_gst(config, modes=('Target', 'full TP'))
        assert gst.circuit_depths == (1, 2)
        assert gst.qubits == [0]
        assert gst.qubit_labels == [0]
        assert gst.modes == ('Target', 'full TP')
        assert gst.germs == smq1Q_XYI.germs([0])
        assert gst.prep_fiducials == smq1Q_XYI.prep_fiducials([0])
        assert gst.meas_fiducials == smq1Q_XYI.meas_fiducials([0])


class TestGSTGenerateCircuits:

    def test_generate_circuits_matches_edesign(self, config):
        gst = _1q_gst(config, modes=('Target',))
        gst.generate_circuits()

        n_expected = len(gst.edesign.all_circuits_needing_data)
        assert n_expected > 0
        assert gst.circuits.n_circuits == n_expected
        assert 'pygsti_circuit' in gst.circuits._df.columns


class TestGSTSimulatedRun:

    def test_noiseless_run_recovers_target_gates(self, config):
        gst = _1q_gst(
            config, modes=('Target', 'full TP'),
            simulator=StateVectorSimulator(), n_shots=None,
        )
        gst.run()

        for infidelity in gst.avg_gate_infidelity['full TP'].values():
            assert infidelity == pytest.approx(0.0, abs=1e-4)
        assert gst.state_prep_fidelity['full TP'] == pytest.approx(
            1.0, abs=1e-4
        )


class TestGSTFiducialPairReduction:

    def test_fpr_reduces_circuits_and_still_fits(self, config):
        full_gst = _1q_gst(config, modes=('Target',))
        full_gst.generate_circuits()
        n_full = full_gst.circuits.n_circuits

        gst = _1q_gst(
            config, modes=('Target', 'full TP'), fpr=True,
            simulator=StateVectorSimulator(), n_shots=None,
        )
        gst.run()

        assert gst.circuits.n_circuits < n_full
        for infidelity in gst.avg_gate_infidelity['full TP'].values():
            assert infidelity == pytest.approx(0.0, abs=1e-4)


class TestSingleQubitGSTConstruction:

    def test_single_qubit_defaults_to_smq1Q_XYI(self, config):
        gst = SingleQubitGST(qpu=Emulator, config=config, qubits=[0])
        assert gst.germs == smq1Q_XYI.germs([0])
        assert gst.prep_fiducials == smq1Q_XYI.prep_fiducials([0])
        assert gst.meas_fiducials == smq1Q_XYI.meas_fiducials([0])

    def test_default_circuit_depths_loads_pregenerated_edesign(self, config):
        # With no custom pspec/fiducials/germs/circuit_depths, the
        # two-qubit simultaneous-single-qubit path defers fiducial/germ
        # computation to generate_circuits(), which should find a
        # pre-generated edesign for the default depths in
        # qcal/default_experiments/ rather than running the expensive
        # find_fiducials()/find_germs() searches.
        gst = SingleQubitGST(qpu=Emulator, config=config, qubits=[0, 1])
        assert gst.germs is None
        assert gst.prep_fiducials is None
        assert gst.meas_fiducials is None

        gst.generate_circuits()

        assert gst.edesign is not None
        assert list(gst.edesign.maxlengths) == list(gst.circuit_depths)
        assert gst.circuits.n_circuits == len(
            gst.edesign.all_circuits_needing_data
        )


class TestTwoQubitGSTConstruction:

    def test_defaults_to_smq2Q_XYCPHASE(self, config):
        gst = TwoQubitGST(qpu=Emulator, config=config, qubit_labels=[(0, 1)])
        assert gst.germs == smq2Q_XYCPHASE.germs((0, 1))
        assert gst.prep_fiducials == smq2Q_XYCPHASE.prep_fiducials((0, 1))
        assert gst.meas_fiducials == smq2Q_XYCPHASE.meas_fiducials((0, 1))


class TestTwoQubitGSTSimulatedRun:

    def test_noiseless_target_mode_run(self, config):
        gst = TwoQubitGST(
            qpu=Emulator, config=config, qubit_labels=[(0, 1)],
            circuit_depths=(1, 2), modes=('Target',),
            simulator=StateVectorSimulator(), n_shots=None,
        )
        gst.run()

        for infidelity in gst.avg_gate_infidelity['Target'].values():
            assert infidelity == pytest.approx(0.0, abs=1e-6)
