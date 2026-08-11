from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class TeachingAction(str, Enum):
    ELICIT = "elicit"
    FOCUS = "focus"
    SCAFFOLD = "scaffold"
    EXPLAIN = "explain"
    CORRECT = "correct"
    CHECK = "check"
    MODEL = "model"
    PRACTICE = "practice"
    CONSOLIDATE = "consolidate"


class DisclosureLevel(str, Enum):
    NONE = "none"
    NEXT_STEP = "next_step"
    FINAL = "final"


class ExplanationDepth(str, Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class PracticeDifficulty(str, Enum):
    NONE = "none"
    EASIER = "easier"
    SIMILAR = "similar"
    HARDER = "harder"


class PracticeSimilarity(str, Enum):
    NONE = "none"
    SAME_STRATEGY = "same_strategy"
    TRANSFER = "transfer"


class SafetyConstraints(BaseModel):
    allowed_actions: list[TeachingAction]
    disclosure_ceiling: DisclosureLevel = DisclosureLevel.NONE
    worked_example_allowed: bool = False
    new_problem_allowed: bool = False
    student_response_required: bool = True
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidates(self) -> "SafetyConstraints":
        if not self.allowed_actions:
            raise ValueError("allowed_actions must not be empty")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("allowed_actions must be unique")
        return self


class TeachingActionPlan(BaseModel):
    action: TeachingAction
    target: str = ""
    scaffold_level: int = Field(default=0, ge=0, le=3)
    explanation_depth: ExplanationDepth = ExplanationDepth.STANDARD
    practice_difficulty: PracticeDifficulty = PracticeDifficulty.NONE
    practice_similarity: PracticeSimilarity = PracticeSimilarity.NONE
    rationale: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    requires_student_response: bool = True
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action_parameters(self) -> "TeachingActionPlan":
        if self.action is TeachingAction.SCAFFOLD and self.scaffold_level == 0:
            raise ValueError("SCAFFOLD requires scaffold_level 1..3")
        if self.action is not TeachingAction.SCAFFOLD and self.scaffold_level != 0:
            raise ValueError("only SCAFFOLD may set scaffold_level")
        practice_fields_set = (
            self.practice_difficulty is not PracticeDifficulty.NONE
            or self.practice_similarity is not PracticeSimilarity.NONE
        )
        if self.action is TeachingAction.PRACTICE and not (
            self.practice_difficulty is not PracticeDifficulty.NONE
            and self.practice_similarity is not PracticeSimilarity.NONE
        ):
            raise ValueError("PRACTICE requires difficulty and similarity")
        if self.action is not TeachingAction.PRACTICE and practice_fields_set:
            raise ValueError("only PRACTICE may set practice parameters")
        return self
