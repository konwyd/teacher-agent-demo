"""运行 Teacher Agent 离线或显式真实模型评测。

用法（项目根目录）：
    python scripts/run_eval.py
    USE_REAL_LLM_FOR_EVAL=true python scripts/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv


_EVAL_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "USE_REAL_LLM_FOR_EVAL",
    "TEACHER_FORCE_MOCK",
    "TEACHER_PROFILE_PATH",
)


def _restore_environment(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> int:
    previous = {key: os.environ.get(key) for key in _EVAL_ENV_KEYS}
    try:
        load_dotenv()
        use_real = os.getenv("USE_REAL_LLM_FOR_EVAL", "").lower() == "true"

        # DeepSeek client imports load .env themselves, so imports precede mock-mode
        # key removal. The scoped environment below is restored in finally.
        from app.evaluation.evaluator import run_all
        from app.evaluation.generic_math_evaluator import run_generic_math_eval

        if not use_real:
            os.environ.pop("DEEPSEEK_API_KEY", None)
            os.environ["TEACHER_FORCE_MOCK"] = "1"

        os.environ.setdefault(
            "TEACHER_PROFILE_PATH",
            str(Path(os.getenv("TEACHER_LOG_DIR", "logs")) / "eval_student_profiles.json"),
        )

        print(f"评测模式：{'真实 DeepSeek' if use_real else 'mock LLM（离线）'}")
        results = run_all() + run_generic_math_eval()

        n_pass = sum(1 for result in results if result.passed)
        n_fail = len(results) - n_pass
        print(f"\n总 case 数：{len(results)}  通过：{n_pass}  失败：{n_fail}\n")
        for result in results:
            mark = "PASS" if result.passed else "FAIL"
            print(f"[{mark}] ({result.case_type}) {result.case_id}")
            for failure in result.failures:
                print(f"       - {failure}")

        out_dir = Path(os.getenv("TEACHER_LOG_DIR", "logs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "eval_results.jsonl"
        timestamp = datetime.now(timezone.utc).isoformat()
        with out_file.open("a", encoding="utf-8") as output:
            for result in results:
                record = result.model_dump()
                record["timestamp"] = timestamp
                record["use_real_llm"] = use_real
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n结果已保存：{out_file}")
        return 0 if n_fail == 0 else 1
    finally:
        _restore_environment(previous)


if __name__ == "__main__":
    raise SystemExit(main())
