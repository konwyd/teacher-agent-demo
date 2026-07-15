import pytest
from fastapi.testclient import TestClient

from app.api.main import app


pytestmark = pytest.mark.safety_contract
client = TestClient(app)


def test_api_preserves_legacy_action_codes_and_adds_unified_plan() -> None:
    data = client.post(
        "/turn",
        json={"student_id": "api-action-contract", "message": "我不会解2x+3=7"},
    ).json()
    assert data["teaching_action"] == "give_hint_level_1"
    assert data["teaching_action_kind"] == "give_hint"
    assert data["teaching_state"]["action_plan"]["action"] == "give_hint"
    assert data["teaching_action_plan"]["action"] == "scaffold"
    assert data["teaching_action_plan"]["scaffold_level"] == 1
