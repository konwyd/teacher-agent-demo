"""Workflow 测试：无 API key、无外部服务也能完整跑一轮（六节点）。"""

from app.agent.state import TeacherState, TeachingAction
from app.agent.workflow import build_workflow, run_turn

EXPECTED_ORDER = [
    "diagnose_node",
    "retrieve_node",
    "teaching_policy_node",
    "teach_node",
    "memory_update_node",
    "log_node",
]


def test_workflow_compiles() -> None:
    assert build_workflow() is not None


def test_turn_runs_all_nodes_in_order() -> None:
    state = run_turn("demo_student", "我不会解 2x+3=7")
    assert state.trace == EXPECTED_ORDER


def test_turn_detects_linear_equation() -> None:
    state = run_turn("demo_student", "我不会解 2x+3=7")
    assert state.detected_concept_id == "c_linear"
    assert state.detected_concept_name == "一元一次方程"


def test_turn_retrieves_prerequisites_and_exercises() -> None:
    state = run_turn("demo_student", "我不会解 2x+3=7")
    assert state.retrieval is not None
    assert "等式性质" in state.retrieval.prerequisites
    assert "移项" in state.retrieval.prerequisites
    assert len(state.retrieval.exercises) >= 1


def test_turn_without_api_key_returns_mock_response() -> None:
    state = run_turn("demo_student", "我不会解 2x+3=7")
    assert isinstance(state.action, TeachingAction)
    assert state.response != ""
    assert "一元一次方程" in state.response


def test_turn_updates_student_state_with_before_after() -> None:
    state = run_turn("demo_student_2", "去括号的题怎么做")
    assert state.detected_concept_id == "c_bracket"
    assert state.mastery_update is not None
    assert state.mastery_update.mastery_before == 0.0
    assert "c_bracket" in state.updated_student_state
    assert state.log_id


def test_state_validates() -> None:
    state = TeacherState(student_id="s1", message="hi")
    assert state.diagnosis is None
    assert state.trace == []
