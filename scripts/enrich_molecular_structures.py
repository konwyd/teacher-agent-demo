#!/usr/bin/env python3
"""CLI for conservative MolScribe + RDKit enrichment of OCR/RAG chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.data_pipeline.molecular_structure_enricher import (
    ImageResolver,
    JsonlPredictionCache,
    MolecularStructureEnricher,
    MolScribePredictionBackend,
    RDKitStructureAnalyzer,
    parse_path_maps,
    read_allowlist,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="给现有化学 OCR/RAG JSONL 追加 MolScribe 分子结构语义",
    )
    parser.add_argument("input", type=Path, help="现有 chunks.jsonl（只读，不会原地覆盖）")
    parser.add_argument("output", type=Path, help="增强后的 JSONL 输出路径")
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        help="图片存储根目录，可重复指定；按 metadata.images 文件名递归解析",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="将历史 source_md 路径前缀映射到当前挂载路径，可重复指定",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("keyword", "allowlist", "all"),
        default="keyword",
        help="keyword 最安全；all 会尝试化学块的所有图片，可能产生域外误识别",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        help="每行一个现有图片索引/文件名；可与 keyword 一起补充候选",
    )
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--min-atoms", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:N")
    parser.add_argument("--checkpoint", type=Path, help="本地 MolScribe .pth；省略则从 HF 下载")
    parser.add_argument(
        "--cache",
        type=Path,
        help="预测缓存 JSONL；默认放在输出文件旁边",
    )
    parser.add_argument(
        "--structures-output",
        type=Path,
        help="可选：输出独立的已接受分子结构 JSONL",
    )
    parser.add_argument("--report", type=Path, help="处理统计 JSON；默认放在输出文件旁边")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_path = args.cache or args.output.with_suffix(".molscribe-cache.jsonl")
    report_path = args.report or args.output.with_suffix(".molscribe-report.json")
    structures_path = args.structures_output or args.output.with_suffix(".molecular-structures.jsonl")

    resolver = ImageResolver(
        image_roots=args.image_root,
        path_maps=parse_path_maps(args.path_map),
    )
    enricher = MolecularStructureEnricher(
        predictor=MolScribePredictionBackend(args.checkpoint, device=args.device),
        analyzer=RDKitStructureAnalyzer(),
        resolver=resolver,
        min_confidence=args.min_confidence,
        min_atoms=args.min_atoms,
        candidate_mode=args.candidate_mode,
        allowed_image_ids=read_allowlist(args.allowlist),
        cache=JsonlPredictionCache(cache_path),
    )
    stats = enricher.enrich_jsonl(args.input, args.output, structures_path=structures_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "structures_output": str(structures_path),
                "candidate_mode": args.candidate_mode,
                "min_confidence": args.min_confidence,
                "stats": stats.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    print(f"增强数据：{args.output}")
    print(f"结构明细：{structures_path}")
    print(f"统计报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
