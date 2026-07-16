#!/usr/bin/env python3
"""通用 Markdown 知识库入库脚本：MD 文件 → 按章节切块 → ChromaDB 指定 collection。

用于将教资、课标、教参等 Markdown 源文件入库到对应集合。
复用与 k9_textbook 相同的向量库目录和 bge-m3 模型。

用法:
    # 教资内容 → teacher_cert 集合
    python build_kb_from_md.py --source-dir data/teacher_kb_source \\
        --pattern '教资_*.md' --collection teacher_cert --doc-type teacher_cert

    # 课标内容 → curriculum_std 集合
    python build_kb_from_md.py --source-dir data/teacher_kb_source \\
        --pattern '课标_*.md' --collection curriculum_std --doc-type curriculum_std
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_DB = Path("/data/wyd-workspace/education/data/china_textbook_k9_vectordb")
EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
DEFAULT_SOURCE = Path("/data/wyd-workspace/education/data/teacher_kb_source")

CHUNK_MAX = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 256


def parse_filename_meta(filepath: Path) -> dict[str, str]:
    """从文件名解析元数据。"""
    name = filepath.stem
    meta: dict[str, str] = {}
    if name.startswith("教资_"):
        meta["source_type"] = "教师资格证"
        if "综合素质" in name:
            meta["cert_subject"] = "综合素质"
        elif "中学" in name:
            meta["cert_subject"] = "教育知识与能力"
            meta["cert_stage"] = "中学"
        elif "小学" in name:
            meta["cert_subject"] = "教育教学知识与能力"
            meta["cert_stage"] = "小学"
    elif name.startswith("课标_"):
        meta["source_type"] = "课程标准"
        if "数学" in name:
            meta["subject"] = "数学"
        meta["stage"] = "义务教育"
    return meta


def split_markdown_sections(md_text: str) -> list[tuple[str, str]]:
    """按 ## 标题切分，返回 [(heading, text), ...]。"""
    lines = md_text.split("\n")
    sections: list[tuple[str, str]] = []
    cur_heading = ""
    cur_lines: list[str] = []
    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            if cur_lines:
                text = "\n".join(cur_lines).strip()
                if text:
                    sections.append((cur_heading, text))
            cur_heading = line[3:].strip()
            cur_lines = []
        elif line.startswith("# ") and not line.startswith("## "):
            if cur_lines:
                text = "\n".join(cur_lines).strip()
                if text:
                    sections.append((cur_heading, text))
            cur_heading = ""
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_lines:
        text = "\n".join(cur_lines).strip()
        if text:
            sections.append((cur_heading, text))
    return sections


def sub_chunk(text: str, max_len: int = CHUNK_MAX, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """长文本再切成 <=max_len 的小块，带 overlap。"""
    text = text.strip()
    if len(text) <= max_len:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_len
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_md_file(filepath: Path, doc_type: str) -> list[dict[str, Any]]:
    """处理单个 markdown 文件，返回 chunk 列表。"""
    md_text = filepath.read_text(encoding="utf-8", errors="ignore")
    file_meta = parse_filename_meta(filepath)
    sections = split_markdown_sections(md_text)
    chunks: list[dict[str, Any]] = []
    for chap_idx, (heading, sec_text) in enumerate(sections, 1):
        full = f"## {heading}\n{sec_text}" if heading else sec_text
        full = re.sub(r"\n{3,}", "\n\n", full).strip()
        for sub in sub_chunk(full):
            if not sub.strip():
                continue
            meta = {
                "doc_type": doc_type,
                "chapter": heading,
                "chapter_idx": chap_idx,
                "source_file": filepath.name,
                **file_meta,
            }
            chunks.append({
                "_id": hashlib.md5(
                    f"{filepath.name}|{chap_idx}|{sub[:100]}".encode()
                ).hexdigest()[:16],
                "text": sub,
                "metadata": meta,
            })
    return chunks


def ingest(
    chunks: list[dict[str, Any]],
    db_path: Path,
    collection_name: str,
    batch: int,
) -> None:
    if not chunks:
        print("无数据可入库，退出")
        return
    print(f"初始化 ChromaDB: {db_path}")
    import os
    _srv = os.environ.get("CHROMA_SERVER", "").strip()
    if _srv:
        _h, _, _p = _srv.partition(":")
        client = chromadb.HttpClient(host=_h, port=int(_p) if _p else 8000)
    else:
        client = chromadb.PersistentClient(path=str(db_path))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL, normalize_embeddings=True,
    )
    coll = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )
    existing = set()
    try:
        existing = set(coll.get()["ids"])
    except Exception:  # noqa: BLE001
        pass
    print(f"DB 已有 {len(existing)} 条")
    todo = [c for c in chunks if c["_id"] not in existing]
    print(f"待入库 {len(todo)}")
    if not todo:
        print("全部已入库，退出")
        return
    t0 = time.time()
    for i in range(0, len(todo), batch):
        b = todo[i : i + batch]
        ids = [c["_id"] for c in b]
        docs = [c["text"] for c in b]
        metas = []
        for c in b:
            m = dict(c["metadata"])
            for k, v in list(m.items()):
                if v is None:
                    m[k] = ""
            metas.append(m)
        coll.add(ids=ids, documents=docs, metadatas=metas)
        done = i + len(b)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        eta = (len(todo) - done) / rate if rate else 0
        print(f"[{done}/{len(todo)}] {rate:.1f} chunks/s ETA {eta:.0f}s", flush=True)
    print(f"\n=== 完成 === 集合 {collection_name} 共 {coll.count()} 条, 耗时 {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Markdown 知识库入库 → ChromaDB 指定集合")
    ap.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    ap.add_argument("--pattern", default="*.md", help="文件名 glob 模式")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--collection", required=True, help="ChromaDB 集合名")
    ap.add_argument("--doc-type", required=True, help="写入 metadata.doc_type")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    src = Path(args.source_dir)
    files = sorted(src.glob(args.pattern))
    print(f"源目录: {src}")
    print(f"匹配 {len(files)} 个文件: {[f.name for f in files]}")
    all_chunks: list[dict[str, Any]] = []
    for f in files:
        c = chunk_md_file(f, args.doc_type)
        print(f"  {f.name}: {len(c)} chunks")
        all_chunks.extend(c)
    print(f"\n合计 {len(all_chunks)} chunks")
    ingest(all_chunks, Path(args.db), args.collection, args.batch)


if __name__ == "__main__":
    main()
