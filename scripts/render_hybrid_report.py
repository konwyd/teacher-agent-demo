"""Render static, dependency-free charts for Hybrid Evaluation runs."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSummary:
    directory: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]


_TERMINATION_LABELS = {
    "solved_and_verified": "完成并验算",
    "student_stuck": "卡住",
    "student_gave_up": "放弃",
    "max_turns": "达到轮数上限",
}
_TERMINATION_COLORS = {
    "solved_and_verified": "#26a69a",
    "student_stuck": "#ef5350",
    "student_gave_up": "#ffb300",
    "max_turns": "#5c6bc0",
}


def load_run_summary(directory: Path) -> RunSummary:
    manifest_path = directory / "manifest.json"
    summary_path = directory / "hybrid_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"incomplete hybrid run: {directory}")
    return RunSummary(directory=directory, manifest=manifest, summary=summary)


def _pct(value: object) -> str:
    return f"{float(value or 0) * 100:.1f}%"


def _number(value: object) -> str:
    return f"{float(value or 0):.2f}"


def _mode_rows(runs: list[RunSummary]) -> list[tuple[str, dict[str, Any], RunSummary]]:
    rows: list[tuple[str, dict[str, Any], RunSummary]] = []
    for run in runs:
        by_mode = run.summary.get("groups", {}).get("by_mode", {})
        for mode, metrics in by_mode.items():
            rows.append((str(mode), dict(metrics), run))
    return sorted(
        rows,
        key=lambda item: (item[0], not bool(item[2].manifest.get("use_real_teacher_llm"))),
    )


def _teacher_source(run: RunSummary) -> str:
    return (
        "真实 DeepSeek"
        if run.manifest.get("use_real_teacher_llm")
        else "离线 mock"
    )


def _metric_bar(label: str, value: object, color: str) -> str:
    percentage = max(0.0, min(100.0, float(value or 0) * 100))
    return (
        '<div class="metric">'
        f'<div class="metric-label"><span>{html.escape(label)}</span>'
        f"<strong>{percentage:.1f}%</strong></div>"
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{percentage:.2f}%;background:{color}"></div>'
        "</div></div>"
    )


def _termination_bar(metrics: dict[str, Any]) -> str:
    distribution = metrics.get("termination_distribution", {})
    segments: list[str] = []
    legend: list[str] = []
    for reason, label in _TERMINATION_LABELS.items():
        item = distribution.get(reason, {})
        rate = float(item.get("rate", 0) or 0)
        count = int(item.get("count", 0) or 0)
        color = _TERMINATION_COLORS[reason]
        if rate:
            segments.append(
                f'<span style="width:{rate * 100:.2f}%;background:{color}" '
                f'title="{html.escape(label)}: {count}"></span>'
            )
        legend.append(
            '<span class="legend-item">'
            f'<i style="background:{color}"></i>{html.escape(label)} {count}'
            "</span>"
        )
    return (
        '<div class="termination"><div class="stacked-bar">'
        f"{''.join(segments)}</div>"
        f'<div class="legend">{"".join(legend)}</div></div>'
    )


def _mastery_rows(run: RunSummary, mode: str) -> str:
    by_mastery = run.summary.get("groups", {}).get("by_mode_mastery", {})
    values: list[str] = []
    for mastery in ("low", "medium", "high"):
        metric = by_mastery.get(f"{mode}|{mastery}", {})
        values.append(f"<td>{_pct(metric.get('task_success_rate'))}</td>")
    return "".join(values)


def build_report(runs: list[RunSummary], *, title: str) -> str:
    if not runs:
        raise ValueError("at least one completed run is required")
    rows = _mode_rows(runs)
    mode_cards: list[str] = []
    table_rows: list[str] = []
    mastery_table_rows: list[str] = []
    for mode, metrics, run in rows:
        escaped_mode = html.escape(mode)
        source = _teacher_source(run)
        display_label = f"{mode} · {source}"
        escaped_display_label = html.escape(display_label)
        mode_cards.append(
            '<section class="card">'
            f"<h2>{escaped_display_label}</h2>"
            f'<p class="subtle">run: {html.escape(str(run.manifest.get("run_id", "")))}</p>'
            + _metric_bar("完成并验算", metrics.get("task_success_rate"), "#26a69a")
            + _metric_bar(
                "教学行为通过率",
                metrics.get("teacher_quality_pass_rate"),
                "#42a5f5",
            )
            + _metric_bar("解题进度", metrics.get("solution_progress"), "#7e57c2")
            + "<h3>终止分布</h3>"
            + _termination_bar(metrics)
            + "</section>"
        )
        table_rows.append(
            "<tr>"
            f"<td>{escaped_display_label}</td>"
            f"<td>{int(metrics.get('conversation_count', 0))}</td>"
            f"<td>{_pct(metrics.get('task_success_rate'))}</td>"
            f"<td>{_pct(metrics.get('teacher_quality_pass_rate'))}</td>"
            f"<td>{_pct(metrics.get('solution_progress'))}</td>"
            f"<td>{_number(metrics.get('teacher_turns'))}</td>"
            "</tr>"
        )
        mastery_table_rows.append(
            f"<tr><td>{escaped_display_label}</td>{_mastery_rows(run, mode)}</tr>"
        )

    sources = "".join(
        "<li>"
        f"{html.escape(str(run.directory))}"
        f"（{html.escape(str(run.manifest.get('trace_count', 0)))} traces）"
        "</li>"
        for run in runs
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{margin:0;background:#f4f7fb;color:#182230;font:15px/1.5 system-ui,-apple-system,"Microsoft YaHei",sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 22px 60px}}
h1{{margin:0 0 4px;font-size:30px}} h2{{margin:0;font-size:20px}} h3{{margin:24px 0 8px;font-size:14px}}
.subtitle,.subtle{{color:#68778c}} .subtle{{margin:4px 0 18px;font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px;margin:26px 0}}
.card,.panel{{background:#fff;border:1px solid #e1e8f0;border-radius:14px;padding:22px;box-shadow:0 3px 12px #1b31550d}}
.metric{{margin-top:16px}} .metric-label{{display:flex;justify-content:space-between;font-size:13px}}
.bar-track,.stacked-bar{{height:12px;background:#eaf0f6;border-radius:99px;overflow:hidden;margin-top:6px}}
.bar-fill{{height:100%;border-radius:99px}} .stacked-bar{{display:flex}} .stacked-bar span{{height:100%}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 13px;margin-top:10px;font-size:12px}} .legend-item{{display:flex;gap:5px;align-items:center}}
.legend i{{width:9px;height:9px;border-radius:99px;display:inline-block}}
table{{width:100%;border-collapse:collapse;margin-top:12px}} th,td{{padding:10px;border-bottom:1px solid #e7edf4;text-align:left}} th{{color:#596a80;font-size:13px}}
ul{{margin:8px 0 0;padding-left:20px;word-break:break-all}} .foot{{margin-top:24px;color:#68778c;font-size:12px}}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="subtitle">Hybrid Student Simulator Evaluation · 包含真实 DeepSeek 与离线 mock · 学生措辞使用确定性模板</p>
<div class="grid">{''.join(mode_cards)}</div>
<section class="panel"><h2>总体对比</h2>
<table><thead><tr><th>Teacher mode</th><th>对话数</th><th>完成并验算</th><th>教学行为通过率</th><th>解题进度</th><th>平均 Teacher turn</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></section>
<section class="panel" style="margin-top:18px"><h2>按初始掌握度的完成率</h2>
<table><thead><tr><th>Teacher mode</th><th>低</th><th>中</th><th>高</th></tr></thead>
<tbody>{''.join(mastery_table_rows)}</tbody></table></section>
<section class="panel" style="margin-top:18px"><h2>数据来源</h2><ul>{sources}</ul></section>
<p class="foot">评分完全由确定性规则产生，不使用 LLM-as-a-judge。</p>
</main></body></html>"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render static charts for completed Hybrid Evaluation runs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Completed run directory; repeat for comparisons.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HTML path.",
    )
    parser.add_argument("--title", default="Hybrid Student Simulator Evaluation")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(
        [load_run_summary(path) for path in args.run_dir],
        title=args.title,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"可视化报告已生成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
