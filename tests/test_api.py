"""API 测试：/turn 返回字段完整。"""

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)

REQUIRED_FIELDS = [
    "detected_concept",
    "intent",
    "student_gap",
    "teaching_action",
    "retrieved_prerequisites",
    "retrieved_exercises",
    "response",
    "updated_student_state",
    "log_id",
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
    assert "等式性质" in data["retrieved_prerequisites"]
    assert data["response"] != ""
    assert "c_linear" in data["updated_student_state"]
    assert data["log_id"]
