#!/usr/bin/env python3
"""补解析缺失学科教材 PDF → ChromaDB k9_textbook 集合。

之前 MinerU 只解析了 130 本人教版教材，语文/历史/科学/道德与法治/艺术等
学科的 PDF 从未被解析。本脚本用 PyMuPDF 做轻量文本提取 + 分块入库，
补齐这些缺失学科。

用法:
    python build_textbook_from_pdf.py                 # 全部缺失学科
    python build_textbook_from_pdf.py --subject 语文  # 仅语文
    python build_textbook_from_pdf.py --subjects 语文,历史,道德与法治
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

SOURCE_ROOT = Path("/data/wyd-workspace/education/data/china_textbook_k9")
DEFAULT_DB = Path("/data/wyd-workspace/education/data/china_textbook_k9_vectordb")
EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
COLLECTION_NAME = "k9_textbook"

CHUNK_MAX = 400
CHUNK_OVERLAP = 80
BATCH_SIZE = 256

# 已入库的学科（跳过，避免重复）
EXISTING_SUBJECTS = {
    "数学", "英语", "音乐", "体育与健康", "日语", "俄语",
    "美术", "生物学", "物理", "化学", "地理", "地理图册",
}


def find_missing_pdfs(subjects: list[str] | None) -> list[Path]:
    """找到缺失学科的 PDF 文件。"""
    pdfs: list[Path] = []
    for p in sorted(SOURCE_ROOT.rglob("*.pdf")):
        if "_test_parse" in str(p):
            continue
        parts = p.relative_to(SOURCE_ROOT).parts
        subject = parts[1] if len(parts) > 1 else ""
        if subject in EXISTING_SUBJECTS:
            continue
        if subjects and subject not in subjects:
            continue
        pdfs.append(p)
    return pdfs


def parse_metadata(src_path: Path) -> dict[str, str]:
    """从源 PDF 路径解析元数据（与 chunk_markdown.py 同逻辑）。"""
    parts = src_path.relative_to(SOURCE_ROOT).parts
    stage = parts[0] if len(parts) > 0 else ""
    subject = parts[1] if len(parts) > 1 else ""
    version = parts[2] if len(parts) > 2 else ""
    grade = ""
    semester = ""
    stem = src_path.stem
    for pt in parts[3:]:
        m = re.search(r"([一二三四五六七八九]年级|\d+年级)", pt)
        if m:
            grade = m.group(1)
            break
    if not grade:
        m = re.search(r"([一二三四五六七八九]年级|\d+年级)", stem)
        if m:
            grade = m.group(1)
    m = re.search(r"(上册|下册|全一册|全册)", stem)
    if m:
        semester = m.group(1)
    return {
        "stage": stage,
        "subject": subject,
        "version": version,
        "grade": grade,
        "semester": semester,
        "stem": stem,
        "source_pdf": str(src_path),
        "images": "",
    }


def extract_pdf_text(pdf_path: Path) -> list[tuple[int, str]]:
    """用 PyMuPDF 提取每页文本，返回 [(page_num, text), ...]。"""
    import fitz

    pages: list[tuple[int, str]] = []
    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = text.strip()
        if text:
            pages.append((i + 1, text))
    doc.close()
    return pages


# 章节标题模式
_HEADING_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百零\d]+[单元章节课]", re.MULTILINE),
    re.compile(r"^(?:Lesson|Unit|Chapter)\s*\d+", re.MULTILINE),
    re.compile(r"^[一二三四五六七八九十]+[、.]", re.MULTILINE),
]


def detect_heading(line: str) -> str | None:
    """检测是否为章节标题行，返回标题名或 None。"""
    line = line.strip()
    if not line or len(line) > 50:
        return None
    for pat in _HEADING_PATTERNS:
        if pat.match(line):
            return line
    return None


def chunk_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """解析单个 PDF，返回 chunk 列表。"""
    meta = parse_metadata(pdf_path)
    pages = extract_pdf_text(pdf_path)
    if not pages:
        return []

    chunks: list[dict[str, Any]] = []
    cur_chapter = ""
    chap_idx = 0
    buf = ""

    for page_num, page_text in pages:
        for line in page_text.split("\n"):
            heading = detect_heading(line)
            if heading:
                # 保存前一段
                if buf.strip():
                    chap_idx += 1
                    for sub in _sub_chunk(buf.strip()):
                        chunks.append(_make_chunk(sub, meta, cur_chapter, chap_idx, pdf_path))
                cur_chapter = heading
                buf = line + "\n"
            else:
                buf += line + "\n"
        buf += "\n"

    # 最后一段
    if buf.strip():
        chap_idx += 1
        for sub in _sub_chunk(buf.strip()):
            chunks.append(_make_chunk(sub, meta, cur_chapter, chap_idx, pdf_path))

    # 如果没有检测到章节标题，按页面分块
    if not chunks:
        full_text = "\n".join(t for _, t in pages)
        for sub in _sub_chunk(full_text):
            if sub.strip():
                chap_idx += 1
                chunks.append(_make_chunk(sub, meta, f"第{chap_idx}段", chap_idx, pdf_path))

    return chunks


def _sub_chunk(text: str, max_len: int = CHUNK_MAX, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_len:
        return [text] if text else []
    result: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_len
        result.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return result


def _make_chunk(
    text: str, meta: dict[str, str], chapter: str, chap_idx: int, pdf_path: Path
) -> dict[str, Any]:
    full = f"## {chapter}\n{text}" if chapter else text
    m = {**meta, "chapter": chapter, "chapter_idx": chap_idx}
    return {
        "_id": hashlib.md5(
            f"{pdf_path.name}|{chap_idx}|{text[:100]}".encode()
        ).hexdigest()[:16],
        "text": full,
        "metadata": m,
    }


def ingest(chunks: list[dict[str, Any]], db_path: Path, batch: int) -> None:
    if not chunks:
        print("无数据可入库，退出")
        return
    print(f"初始化 ChromaDB: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL, normalize_embeddings=True,
    )
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=emb_fn,
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
        metas = [dict(c["metadata"]) for c in b]
        coll.add(ids=ids, documents=docs, metadatas=metas)
        done = i + len(b)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        eta = (len(todo) - done) / rate if rate else 0
        print(f"[{done}/{len(todo)}] {rate:.1f} chunks/s ETA {eta:.0f}s", flush=True)
    print(f"\n=== 完成 === 集合 {COLLECTION_NAME} 共 {coll.count()} 条, 耗时 {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="补解析缺失学科教材 PDF → k9_textbook")
    ap.add_argument("--subject", default="", help="单学科过滤")
    ap.add_argument("--subjects", default="", help="多学科过滤（逗号分隔）")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    subjects = None
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    elif args.subject:
        subjects = [args.subject.strip()]

    pdfs = find_missing_pdfs(subjects)
    print(f"找到 {len(pdfs)} 个待解析 PDF")
    if not pdfs:
        print("无缺失学科，退出")
        return

    all_chunks: list[dict[str, Any]] = []
    for pdf in pdfs:
        chunks = chunk_pdf(pdf)
        meta = parse_metadata(pdf)
        print(f"  {meta.get('stage', '')}·{meta.get('subject', '')} "
              f"{meta.get('grade', '')}{meta.get('semester', '')} "
              f"({len(chunks)} chunks)")
        all_chunks.extend(chunks)

    print(f"\n合计 {len(all_chunks)} chunks")
    ingest(all_chunks, Path(args.db), args.batch)


if __name__ == "__main__":
    main()
