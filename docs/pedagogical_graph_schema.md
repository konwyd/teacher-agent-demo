# Pedagogical Graph Schema

## 1. Purpose

`data/sample/pedagogical_graph.json` is a local pedagogical decision graph for
the linear-equation MVP. It stores evidence about how to teach, not a corpus of
textbook answers.

The graph is read-only at runtime. Pedagogical evidence is a soft constraint:

- `math_checker` remains authoritative for mathematical correctness.
- The workflow remains authoritative for teaching-action selection.
- Retrieved examples and practice problems must not replace an unfinished
  `original_problem`.
- Memory and graph evidence may adjust explanation depth, hint level, review,
  and practice difficulty, but may not force the student through a script.

## 2. File Layout

Schema version `1.0.0` uses two primary sections:

```json
{
  "schema_version": "1.0.0",
  "graph_id": "pedagogical_graph_linear_equation_mvp",
  "nodes": {
    "concepts": [],
    "skills": [],
    "misconceptions": [],
    "diagnostic_questions": [],
    "hints": [],
    "worked_examples": [],
    "practice_problems": [],
    "teaching_strategies": [],
    "mastery_thresholds": []
  },
  "edges": []
}
```

Node IDs are stable strings and are unique within their node type. Edges use
those IDs in `from` and `to`. The temporary top-level `concepts` collection is
a legacy projection retained for old readers; `nodes` and `edges` are the new
graph representation.

## 3. Node Types

### Concept

A teachable concept or micro-concept.

Required fields:

- `concept_id`
- `name`
- `grade`
- `domain`
- `prerequisites`
- `skills`
- `misconceptions`
- `diagnostic_questions`
- `leveled_hints`
- `practice_problems`
- `teaching_strategies`
- `mastery_thresholds`

The MVP contains six concepts: linear equations, equality properties, moving
terms, dividing by the coefficient, removing parentheses, and fractional
coefficient equations.

### Skill

An observable mathematical capability that can accumulate mastery evidence.

Fields:

- `skill_id`
- `name`
- `observable_behaviors`
- `related_concepts`

Examples include preserving equality, moving terms with correct signs, clearing
denominators, and verifying a solution.

### Misconception

A recurring incorrect model or procedure.

Fields:

- `misconception_id`
- `concept_id`
- `description`
- `example_wrong_step`
- `diagnostic_signal`
- `repair_goal`

`example_wrong_step` is evidence for matching and evaluation. It is not a
worked example and must not be inserted into an ordinary teacher response.

### DiagnosticQuestion

A minimal question used to distinguish prerequisite gaps or misconceptions.

Fields:

- `question_id`
- `concept_id`
- `prompt`
- `checks`
- `expected_signal`
- `avoid_revealing_answer`

One turn should still contain at most one question, even if retrieval returns
multiple candidates.

### Hint

One step in a progressive hint ladder.

Fields:

- `hint_id`
- `concept_id`
- `level`
- `text`
- `reveals_final_answer`
- `next_hint_id`

Levels progress from structural attention, to a rule or operation, to a more
specific procedural cue. They must not disclose the final answer unless the
student has already solved the problem or explicitly requested it.

### WorkedExample

A guarded demonstration for a concept.

Fields:

- `example_id`
- `concept_id`
- `title`
- `stem`
- `teaching_focus`
- `solution_outline`
- `use_only_when_action`

The current guard is `show_worked_example`. A worked example may enter the
teacher context only when the workflow explicitly allows that action.

### PracticeProblem

A new exercise or variation used after the current problem is complete or when
the student explicitly requests practice.

Fields:

- `problem_id`
- `concept_id`
- `stem`
- `difficulty`
- `teaching_focus`
- `use_only_after_current_problem`

Difficulty is an ordinal value within the MVP dataset, not a universal scale.

### TeachingStrategy

A reusable teaching-policy recommendation.

Fields:

- `strategy_id`
- `name`
- `when_to_use`
- `preferred_actions`
- `avoid`
- `teacher_tone`

Strategies recommend actions; they do not override checker-first branches or
the workflow state machine.

### MasteryThreshold

A mapping from a mastery interval to a strategy and concept-specific strategy
text.

Fields:

- `threshold_id`
- `concept_id`
- `bands`

Each band contains its inclusive/exclusive bounds, `strategy_id`, and
`strategy_text`. The current bands are low `[0.0, 0.3)`, medium `[0.3, 0.7)`,
and high `[0.7, 1.0]`.

## 4. Edge Types

| Edge | Direction | Meaning |
|---|---|---|
| `REQUIRES` | Concept -> Concept | Source concept depends on the target concept. |
| `HAS_SKILL` | Concept -> Skill | Concept builds or requires an observable skill. |
| `HAS_MISCONCEPTION` | Concept -> Misconception | Misconception commonly occurs for the concept. |
| `DIAGNOSE_WITH` | Misconception -> DiagnosticQuestion | Question can test for that misconception. |
| `HINT_FOR` | Hint -> Concept | Hint belongs to the concept's hint ladder. |
| `EXAMPLE_FOR` | WorkedExample -> Concept | Worked example illustrates the concept. |
| `PRACTICE_FOR` | PracticeProblem -> Concept | Practice problem exercises the concept. |
| `REMEDIATES` | Hint -> Misconception | Hint can repair or expose the misconception. |
| `EASIER_THAN` | PracticeProblem -> PracticeProblem | Source problem is easier than the target. |
| `HARDER_THAN` | PracticeProblem -> PracticeProblem | Source problem is harder than the target. |
| `NEXT_CONCEPT` | Concept -> Concept | Suggested next step in a learning path. |
| `USE_STRATEGY_WHEN` | MasteryThreshold -> TeachingStrategy | Strategy applies to the edge's mastery band. |

Edges may carry small decision properties such as `importance`,
`mastery_band`, `level`, or action guards. They must not contain student
answers or mutable student state.

## 5. Retrieval Compatibility

`JsonPedagogicalRetriever` reads `nodes` first and resolves concept references
into the existing `PedagogicalConceptContext`. The public methods remain:

- `get_concept_context`
- `get_prerequisites`
- `get_misconceptions`
- `get_diagnostic_questions`
- `get_hint_levels`
- `get_recommended_exercises`
- `get_mastery_based_strategy`

This preserves the current workflow contract. The retriever also retains a
fallback for the old top-level `concepts` format, which makes schema migration
incremental and keeps tests and offline evaluation stable.

The compatibility mapping is:

| New graph nodes | Existing context field |
|---|---|
| Concept prerequisite references | `prerequisites` |
| Misconception nodes | `common_misconceptions` |
| DiagnosticQuestion nodes | `diagnostic_questions` |
| Hint nodes | `hint_levels` |
| PracticeProblem nodes | `recommended_exercises` |
| MasteryThreshold bands | `mastery_based_strategy` |

Worked examples, graph paths, and skill nodes are intentionally not injected
into the existing prompt automatically. Future retrieval methods may expose
them only behind the appropriate workflow action and grounding checks.

## 6. Validation Rules

- All nine node collections must exist.
- The six MVP concepts must remain available through legacy IDs and names.
- Every core concept must reference skills, misconceptions, hints, practice
  problems, teaching strategies, and mastery thresholds.
- Every reference and edge endpoint must resolve to a known node.
- Hint levels must be ordered and must not reveal final answers.
- Practice and worked-example nodes must carry action guards.
- Changes must pass `python -m pytest` and `python scripts/run_eval.py`.
