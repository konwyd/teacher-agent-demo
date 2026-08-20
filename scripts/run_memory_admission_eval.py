#!/usr/bin/env python3
"""Run the fixed Memory v2 Phase 2A admission benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.memory_admission import MemoryAdmissionEvaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = MemoryAdmissionEvaluator().run(args.split)
    payload = report.model_dump(mode="json", exclude={"records"})
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
