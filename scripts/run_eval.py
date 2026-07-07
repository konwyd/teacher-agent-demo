"""运行 Teacher Agent 评测。

用法（项目根目录）：
    python scripts/run_eval.py                      # 默认 mock LLM，不调真实 API
    USE_REAL_LLM_FOR_EVAL=true python scripts/run_eval.py   # 真实 DeepSeek 评测

输出总数/通过/失败与每个失败 case 的原因，并保存 logs/eval_results.jsonl。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

USE_REAL = os.getenv("USE_REAL_LLM_FOR_EVAL", "").lower() == "true"

# 注意：必须先完成所有 import 再移除 key——app.llm.deepseek_client 在模块导入时
# 会执行 load_dotenv()，如果先 pop 后 import，key 会被重新注入，导致“mock 评测”
# 实际调用了真实 DeepSeek。
from app.evaluation.evaluator import run_all  # noqa: E402

if not USE_REAL:
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ["TEACHER_FORCE_MOCK"] = "1"  # 双保险：is_configured() 直接短路

# 评测使用独立的学生档案文件，避免污染真实学生记忆
os.environ.setdefault(
    "TEACHER_PROFILE_PATH",
    str(Path(os.getenv("TEACHER_LOG_DIR", "logs")) / "eval_student_profiles.json"),
)


def main() -> int:
    print(f"评测模式：{'真实 DeepSeek' if USE_REAL else 'mock LLM（离线）'}")
    results = run_all()

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"\n总 case 数：{len(results)}  通过：{n_pass}  失败：{n_fail}\n")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] ({r.case_type}) {r.case_id}")
        for f in r.failures:
            print(f"       - {f}")

    out_dir = Path(os.getenv("TEACHER_LOG_DIR", "logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_results.jsonl"
    ts = datetime.now(timezone.utc).isoformat()
    with out_file.open("a", encoding="utf-8") as f:
        for r in results:
            record = r.model_dump()
            record["timestamp"] = ts
            record["use_real_llm"] = USE_REAL
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n结果已保存：{out_file}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
