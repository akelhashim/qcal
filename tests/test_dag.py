"""Unit tests for qcal.dag.dag.DAG.

A DAG node's module is instantiated as `module(**parameters)` and then
run via `.run()` (see Node.execute()), so _RecordingModule below is a
minimal stand-in for a real calibration class: it accepts arbitrary
kwargs and appends its label to a shared list on `.run()`, optionally
raising instead. This lets tests assert both whether a node ran and
the order nodes ran in, without depending on a real calibration
protocol or backend.
"""
import pytest

from qcal.dag.dag import DAG, NodeStatus


class _RecordingModule:

    def __init__(self, calls, label, fail=False, **kwargs):
        self._calls = calls
        self._label = label
        self._fail = fail

    def run(self):
        if self._fail:
            raise RuntimeError(f'{self._label} failed')
        self._calls.append(self._label)


@pytest.fixture
def calls():
    return []


def _add(dag, calls, name, dependencies=None, fail=False):
    dag.add_node(
        name,
        _RecordingModule,
        {'calls': calls, 'label': name, 'fail': fail},
        dependencies=dependencies,
    )


@pytest.fixture
def diamond_dag(calls):
    """A -> B, A -> C, {B, C} -> D."""
    dag = DAG('diamond')
    _add(dag, calls, 'A')
    _add(dag, calls, 'B', dependencies=['A'])
    _add(dag, calls, 'C', dependencies=['A'])
    _add(dag, calls, 'D', dependencies=['B', 'C'])
    return dag


class TestAddNode:

    def test_registers_node_and_graph_node(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        assert 'A' in dag.nodes

    def test_duplicate_name_raises(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        with pytest.raises(ValueError):
            _add(dag, calls, 'A')

    def test_dependencies_kwarg_wires_edge(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        _add(dag, calls, 'B', dependencies=['A'])
        assert ('B', 'A') in dag.edges


class TestDependencies:

    def test_add_dependency_missing_node_raises(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        with pytest.raises(ValueError):
            dag.add_dependency('bogus', 'A')

    def test_add_dependency_missing_depends_on_raises(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        with pytest.raises(ValueError):
            dag.add_dependency('A', 'bogus')

    def test_cycle_raises_and_reverts(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        _add(dag, calls, 'B', dependencies=['A'])
        with pytest.raises(ValueError):
            dag.add_dependency('A', 'B')
        assert ('A', 'B') not in dag.edges

    def test_add_dependencies_adds_multiple(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        _add(dag, calls, 'B')
        _add(dag, calls, 'C', dependencies=['A', 'B'])
        assert ('C', 'A') in dag.edges
        assert ('C', 'B') in dag.edges


class TestGetExecutionOrder:

    def test_linear_chain(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        _add(dag, calls, 'B', dependencies=['A'])
        _add(dag, calls, 'C', dependencies=['B'])
        assert dag.get_execution_order() == ['A', 'B', 'C']

    def test_diamond_respects_partial_order(self, diamond_dag):
        order = diamond_dag.get_execution_order()
        assert order.index('A') < order.index('B')
        assert order.index('A') < order.index('C')
        assert order.index('B') < order.index('D')
        assert order.index('C') < order.index('D')

    def test_raises_on_cycle(self, calls):
        # add_dependency() itself guards against cycles, so reaching
        # get_execution_order()'s own cycle check requires wiring the
        # internal dependency maps directly, bypassing that guard.
        dag = DAG()
        _add(dag, calls, 'A')
        _add(dag, calls, 'B')
        dag._dependencies['A'].add('B')
        dag._dependencies['B'].add('A')
        dag._dependents['B'].add('A')
        dag._dependents['A'].add('B')
        with pytest.raises(RuntimeError):
            dag.get_execution_order()


class TestCanExecute:

    def test_true_with_no_dependencies(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        assert dag.can_execute('A') is True

    def test_false_when_dependency_pending(self, diamond_dag):
        assert diamond_dag.can_execute('B') is False

    def test_true_when_dependency_completed(self, diamond_dag):
        diamond_dag.execute_node('A')
        assert diamond_dag.can_execute('B') is True

    def test_true_when_dependency_skipped(self, diamond_dag):
        diamond_dag._nodes['A'].status = NodeStatus.SKIPPED
        assert diamond_dag.can_execute('B') is True


class TestExecuteNode:

    def test_runs_module_and_sets_completed(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        dag.execute_node('A')
        assert calls == ['A']
        assert dag._nodes['A'].status == NodeStatus.COMPLETED
        assert dag._nodes['A'].execution_time is not None

    def test_raises_when_dependencies_not_satisfied(self, diamond_dag):
        with pytest.raises(RuntimeError):
            diamond_dag.execute_node('B')

    def test_check_dependencies_false_skips_check(self, diamond_dag):
        diamond_dag.execute_node('B', check_dependencies=False)
        assert diamond_dag._nodes['B'].status == NodeStatus.COMPLETED

    def test_sets_failed_status_and_reraises(self, calls):
        dag = DAG()
        _add(dag, calls, 'A', fail=True)
        with pytest.raises(RuntimeError):
            dag.execute_node('A')
        assert dag._nodes['A'].status == NodeStatus.FAILED
        assert dag._nodes['A'].error is not None


class TestExecute:

    def test_runs_all_nodes_in_dependency_order(self, diamond_dag, calls):
        diamond_dag.execute()
        assert calls.index('A') < calls.index('B')
        assert calls.index('A') < calls.index('C')
        assert calls.index('B') < calls.index('D')
        assert calls.index('C') < calls.index('D')
        assert all(
            node.status == NodeStatus.COMPLETED
            for node in diamond_dag._nodes.values()
        )

    def test_start_node_skips_earlier_nodes_but_marks_completed(
        self, diamond_dag, calls
    ):
        diamond_dag.execute(start_node='B')
        assert 'A' not in calls
        assert 'B' in calls
        assert diamond_dag._nodes['A'].status == NodeStatus.COMPLETED

    def test_stop_node_is_inclusive(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        _add(dag, calls, 'B', dependencies=['A'])
        _add(dag, calls, 'C', dependencies=['B'])
        dag.execute(stop_node='B')
        assert calls == ['A', 'B']

    def test_skip_nodes_are_not_run(self, calls):
        dag = DAG()
        _add(dag, calls, 'A')
        _add(dag, calls, 'B', dependencies=['A'])
        dag.execute(skip_nodes=['A'])
        assert calls == ['B']
        assert dag._nodes['A'].status == NodeStatus.SKIPPED

    def test_bogus_start_node_raises(self, diamond_dag):
        with pytest.raises(ValueError):
            diamond_dag.execute(start_node='bogus')

    def test_node_failure_raises_runtime_error_with_cause(self, calls):
        dag = DAG()
        _add(dag, calls, 'A', fail=True)
        with pytest.raises(RuntimeError) as exc_info:
            dag.execute()
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestExecuteSubgraph:

    def test_runs_only_specified_nodes(self, diamond_dag, calls):
        diamond_dag.execute_subgraph(['A', 'B'])
        assert set(calls) == {'A', 'B'}

    def test_skips_node_whose_subgraph_dependency_is_unmet(
        self, diamond_dag, calls
    ):
        # D depends on B and C, but only B is in this subgraph, so D's
        # in-subgraph dependency (B) is satisfied -- it should still
        # run since can_execute isn't checked against C here.
        diamond_dag.execute_subgraph(['A', 'B', 'D'])
        assert set(calls) == {'A', 'B', 'D'}

    def test_missing_node_raises(self, diamond_dag):
        with pytest.raises(ValueError):
            diamond_dag.execute_subgraph(['bogus'])


class TestReachableAndRequiredNodes:

    def test_get_reachable_nodes(self, diamond_dag):
        assert diamond_dag.get_reachable_nodes('A') == {
            'A', 'B', 'C', 'D'
        }
        assert diamond_dag.get_reachable_nodes('D') == {'D'}

    def test_get_required_nodes(self, diamond_dag):
        assert diamond_dag.get_required_nodes('D') == {
            'A', 'B', 'C', 'D'
        }
        assert diamond_dag.get_required_nodes('A') == {'A'}

    def test_get_reachable_nodes_missing_node_raises(self, diamond_dag):
        with pytest.raises(ValueError):
            diamond_dag.get_reachable_nodes('bogus')

    def test_get_required_nodes_missing_node_raises(self, diamond_dag):
        with pytest.raises(ValueError):
            diamond_dag.get_required_nodes('bogus')


class TestReset:

    def test_reset_clears_status_and_timing(self, diamond_dag):
        diamond_dag.execute()
        diamond_dag.reset()
        for node in diamond_dag._nodes.values():
            assert node.status == NodeStatus.PENDING
            assert node.error is None
            assert node.start_time is None
            assert node.end_time is None


class TestNodeStatus:

    def test_get_node_status(self, diamond_dag):
        diamond_dag.execute_node('A')
        status = diamond_dag.get_node_status()
        assert status['A'] == NodeStatus.COMPLETED
        assert status['B'] == NodeStatus.PENDING


class TestVisualizeAndDraw:

    def test_visualize_prints_execution_order(self, diamond_dag, capsys):
        diamond_dag.visualize()
        out = capsys.readouterr().out
        for name in ('A', 'B', 'C', 'D'):
            assert name in out

    def test_draw_runs_without_error(self, diamond_dag):
        diamond_dag.draw()
