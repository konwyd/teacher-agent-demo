"""API 测试：/turn 返回字段完整。"""

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)

REQUIRED_FIELDS = [
    "detected_concept",
    "intent",
    "student_gap",
    "teaching_action",
    "socratic_stage",
    "socratic_move",
    "socratic_reason",
    "retrieved_prerequisites",
    "retrieved_exercises",
    "response",
    "updated_student_state",
    "log_id",
    "student_mastery_before",
    "student_mastery_after",
    "mastery_delta",
    "known_misconceptions",
    "recommended_next_action",
]


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_turn_returns_all_fields() -> None:
    resp = client.post(
        "/turn",
        json={"student_id": "api_student", "message": "我不会解 2x+3=7"},
    )
    assert resp.status_code == 200
    data = resp.json()
    for field in REQUIRED_FIELDS:
        assert field in data, f"missing field: {field}"
    assert data["detected_concept"] == "一元一次方程"
    assert data["intent"] == "problem_solving"
    assert data["teaching_action"] == "give_hint_level_1"
    assert data["socratic_stage"] == "concept_probe"
    assert data["socratic_move"] == "generate_subquestion"
    assert "等式性质" in data["retrieved_prerequisites"]
    assert data["response"] != ""
    assert "c_linear" in data["updated_student_state"]
    assert data["log_id"]
    assert data["student_mastery_before"] == 0.5  # 新学生中性初始
    assert data["mastery_delta"] == -0.05         # 求助视为轻微未掌握信号
    assert data["recommended_next_action"] != ""
