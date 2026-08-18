"""写入一批人工审阅问答，用于验证优质问答参考库。

用法（需要 Chroma 已启动，通常是 127.0.0.1:8002）：

    python3 scripts/seed_qa_cache.py

历史回答只作为教师生成参考，不会绕过分类、教学契约和验证直接返回。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototypes.llm_led_teacher import qa_cache  # noqa: E402

# (问题, 答案, 学科, 学段)
_SEED_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "什么是方程？",
        "方程就是含有未知数的等式。关键点有两个：一是等式（左右两边相等），"
        "二是含有未知数（通常用字母 x、y 表示）。比如 2x+3=11 就是一个方程。",
        "数学",
        "初中",
    ),
    (
        "一元一次方程怎么解？",
        "一元一次方程只含一个未知数、未知数的次数是 1。解它的核心思路是"
        "“移项变号、合并同类项、系数化为 1”，把未知数逐步单独留在等式一边。",
        "数学",
        "初中",
    ),
    (
        "什么是浮力？",
        "浮力是液体或气体对浸在其中的物体向上托的力，方向竖直向上。"
        "物体受到的浮力大小等于它排开的液体所受的重力，这就是阿基米德原理。",
        "物理",
        "初中",
    ),
    (
        "什么是光合作用？",
        "光合作用是绿色植物利用光能，把二氧化碳和水转化成有机物（主要是淀粉），"
        "并释放氧气的过程，通常在叶绿体中进行。",
        "生物",
        "初中",
    ),
    (
        "什么是化合价？",
        "化合价表示元素原子之间相互化合时得失电子的数目。常见规律："
        "氢通常为 +1 价、氧通常为 -2 价、化合物中正负化合价代数和为 0。",
        "化学",
        "初中",
    ),
    (
        "记叙文开头怎么写？",
        "记叙文开头有几种常用方法：开门见山直接点题、用设问或悬念引入、"
        "先写场景或环境再带出人物事件。开头要短，尽快引出你要写的主要事件。",
        "语文",
        "初中",
    ),
]


def main() -> int:
    ok_count = 0
    for question, answer, subject, stage in _SEED_PAIRS:
        stored = qa_cache.store(question, answer, subject=subject, stage=stage)
        status = "OK  " if stored else "FAIL"
        print(f"[{status}] {subject}/{stage}  {question}")
        ok_count += int(stored)
    version = qa_cache.current_version()
    if ok_count:
        version = qa_cache.publish_new_version()
    print(
        f"\n写入 {ok_count}/{len(_SEED_PAIRS)} 条到集合 "
        f"{qa_cache.QA_CACHE_COLLECTION}，已发布版本 v{version}。"
    )
    return 0 if ok_count == len(_SEED_PAIRS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
