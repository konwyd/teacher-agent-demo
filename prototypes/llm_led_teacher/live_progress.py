"""User-visible, auditable progress trace for one autonomous teaching turn."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Protocol


class StatusWriter(Protocol):
    def write(self, value: str) -> object: ...


_STAGES = {
    "observing": (
        1,
        "识别题目与教学节奏",
        "判断是否开启新题，以及本轮应该推进一步还是完整回答。",
    ),
    "retrieving": (
        2,
        "检索教材与题库",
        "读取有边界的本地参考；检索失败不会阻断教学。",
    ),
    "generating": (
        3,
        "组织教师初稿",
        "让 LLM 根据原题、对话、验证证据和可用参考自主教学。",
    ),
    "auditing": (
        4,
        "核验数学步骤与教学节奏",
        "提取可验证数学断言，并检查越步和最终答案披露。",
    ),
    "repairing": (
        5,
        "修正未通过审计的初稿",
        "只修正审计发现的冲突；未通过审计的初稿不会展示。",
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
        self._status.write(
            f"✅ 完成{_COMPLETED_TITLES[self._stage]} · **{elapsed:.3f} 秒**"
        )
        if stage == "complete":
            return
        self._stage = stage
        self._started_at = now
        self._write_start(stage)
