# Test Contract Layers

- `tests/contracts/`: permanent safety and public interfaces; assert structured behavior, never exact prose.
- `tests/behavior/`: new real multi-turn teaching behavior; assert allowed actions, progression, evidence, disclosure, and one final question.
- existing `tests/test_*.py`: frozen legacy locks; keep green and add no new cases.
- remove a legacy assertion only in the same commit that adds its replacement contract or behavior test.
