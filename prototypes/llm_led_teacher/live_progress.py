"""User-visible, auditable progress trace for one teaching turn."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Protocol


class StatusWriter(Protocol):
    def write(self, value: str) -> object: ...


_STAGES = {
    "observing": (
        1,
        "识别题目与教学节奏",
        "判断是否开启新题、学生是否要求完整答案，以及本轮应该推进一步还是完整回答。",
    ),
    "retrieving": (
        2,
        "检索教材与题库",
        "查询本地 Chroma 教材库和题库；新题在本地有效命中不足时才尝试联网回退。",
    ),
    "generating": (
        3,
        "让本地模型组织教师初稿",
        "把原题、对话历史、有效 guidance、确定性证据和有边界的 RAG 参考交给本地 DeepSeek。",
    ),
    "auditing": (
        4,
        "核验数学步骤与教学节奏",
        "提取回答中的方程，运行 verifier，并检查是否越步、泄露最终答案或包含多个结尾问题。",
    ),
    "repairing": (
        5,
        "修正未通过审计的初稿",
        "只把审计发现的冲突交给本地模型修正；未修正的初稿不会直接展示给学生。",
    ),
}

_COMPLETED_TITLES = {
    "observing": "题目与节奏判断",
    "retrieving": "教材与题库检索",
    "generating": "教师初稿生成",
    "auditing": "数学步骤与教学节奏核验",
    "repairing": "安全修正",
}


class LiveProgressTrace:
    """Append completed work and elapsed time to a Streamlit-like status box."""

    def __init__(
        self,
        status: StatusWriter,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._status = status
        self._clock = clock
        self._stage = "observing"
        self._started_at = clock()
        self._write_start(self._stage)

    def _write_start(self, stage: str) -> None:
        number, title, detail = _STAGES[stage]
        self._status.write(f"⏳ **{number}. 正在{title}**\n\n{detail}")

    def update(self, stage: str) -> None:
        now = self._clock()
        elapsed = now - self._started_at
        self._status.write(f"✅ 完成{_COMPLETED_TITLES[self._stage]} · **{elapsed:.3f} 秒**")
        if stage == "complete":
            return
        self._stage = stage
        self._started_at = now
        self._write_start(stage)
