#!/usr/bin/env python3
"""题库入库脚本：初中多学科 JSONL 题库 → ChromaDB collection `k9_exercises`。

B 方案：题库作为独立 collection，与 k9_textbook 共存于同一个向量库目录。
  向量库: /data/wyd-workspace/education/data/china_textbook_k9_vectordb/
  集合:   k9_exercises

数据源：
  1. 初中多学科题库 JSONL（10 学科，带答案/解析/知识点）
     /data/wqx/education/LlamaFactory/data/edu/data/raw/{train,test,valid}_初中{学科}.jsonl
  2. 河南中考真题 Markdown（7 学科，试卷+答案）
     /data/education/llm-test/testmd/paper/*.md + /data/education/llm-test/testmd/key/*.md

每条题目的 chunk 结构：
  text     = 【题干】...\\n【答案】...\\n【解析】...
  metadata = {doc_type, stage, subject, ques_type, difficulty, knowledges, source, ...}

用法:
    python build_exercise_db.py                          # 全部学科全部 split
    python build_exercise_db.py --subject 数学            # 仅数学
    python build_exercise_db.py --split train             # 仅 train 集
    python build_exercise_db.py --source jsonl            # 仅 JSONL 题库
    python build_exercise_db.py --source exam --subject 数学  # 仅中考真题数学
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

# ---- ChromaDB 兼容：切 SegmentAPI + 兼容旧版 pickle（必须在 import chromadb 前）----
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.retrieval.chroma_compat import ensure_chroma_compat, get_chroma_client

ensure_chroma_compat()

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_DB = Path("/data/wyd-workspace/education/data/china_textbook_k9_vectordb")
EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
COLLECTION_NAME = "k9_exercises"

JSONL_DIR = Path("/data/wqx/education/LlamaFactory/data/edu/data/raw")
EXAM_PAPER_DIR = Path("/data/education/llm-test/testmd/paper")
EXAM_KEY_DIR = Path("/data/education/llm-test/testmd/key")

SPLITS = ("train", "test", "valid")
# 学科名映射：文件名中的中文学科 → 元数据 subject 值
SUBJECTS = ("数学", "物理", "化学", "生物", "地理", "历史", "语文", "英语",
             "科学", "信息技术", "道德与法治", "道法")
# 道法 → 道德与法治 规范化映射
_SUBJECT_NORMALIZE = {"道法": "道德与法治"}

BATCH_SIZE = 512


# ---------------------------------------------------------------- JSONL 题库


def _parse_subject(raw: str) -> tuple[str, str]:
    """'初中数学' → ('初中', '数学')；'道法' 规范化为 '道德与法治'。"""
    for subj in SUBJECTS:
        if subj in raw:
            return "初中", _SUBJECT_NORMALIZE.get(subj, subj)
    return "初中", raw


def _to_str(v: Any) -> str:
    """把 JSONL 字段值统一转为字符串：dict 取嵌套 ques_content；list 用分号拼接。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return ";".join(str(x).strip() for x in v if x)
    if isinstance(v, dict):
        # 偶发嵌套：{ques_type, ques_content, ques_answer, ques_analyze}
        return _to_str(v.get("ques_content") or v.get("content") or "")
    return str(v).strip()


def _stable_id(*parts: str) -> str:
    """根据内容生成稳定 ID（去重 + 断点续跑用）。"""
    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"ex_{h}"


def load_jsonl_exercises(
    subjects: list[str] | None,
    splits: list[str],
) -> list[dict]:
    """从 JSONL 题库加载题目，返回 chunk 列表。"""
    chunks: list[dict] = []
    files = sorted(JSONL_DIR.glob("*.jsonl"))
    for f in files:
        # 解析文件名：train_初中数学.jsonl
        m = re.match(r"(train|test|valid)_初中(.+?)\.jsonl", f.name)
        if not m:
            continue
        split, subj_name = m.group(1), m.group(2)
        if split not in splits:
            continue
        if subjects and subj_name not in subjects:
            continue
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage, subject = _parse_subject(rec.get("subject", subj_name))
                content = _to_str(rec.get("ques_content"))
                if not content:
                    continue
                answer = _to_str(rec.get("ques_answer"))
                analyze = _to_str(rec.get("ques_analyze"))
                knowledges = rec.get("ques_knowledges") or []
                if isinstance(knowledges, list):
                    knowledges = ";".join(str(k) for k in knowledges)
                else:
                    knowledges = _to_str(knowledges)
                text_parts = [f"【题干】{content}"]
                if answer:
                    text_parts.append(f"【答案】{answer}")
                if analyze:
                    text_parts.append(f"【解析】{analyze}")
                text = "\n".join(text_parts)
                meta = {
                    "doc_type": "exercise",
                    "stage": stage,
                    "subject": subject,
                    "ques_type": rec.get("ques_type", ""),
                    "difficulty": rec.get("ques_difficulty", ""),
                    "knowledges": knowledges,
                    "source": "题库JSONL",
                    "source_file": f.name,
                    "split": split,
                }
                chunks.append({
                    "_id": _stable_id(content),
                    "text": text,
                    "metadata": meta,
                })
    return chunks


# ---------------------------------------------------------------- 中考真题 MD


def load_exam_md_exercises(subjects: list[str] | None) -> list[dict]:
    """从中考真题 Markdown 加载（试卷+答案配对），按题号切块。

    真题 MD 结构不统一，这里做保守处理：把试卷和答案配对后，
    按「题号」正则切分，每题一个 chunk；无法切分时整文件存一个 chunk。
    """
    chunks: list[dict] = []
    if not EXAM_PAPER_DIR.exists():
        return chunks
    for paper in sorted(EXAM_PAPER_DIR.glob("*.md")):
        # 2026数学试卷.md → subject=数学
        m = re.match(r"\d+(.+?)试卷", paper.name)
        if not m:
            continue
        subject = m.group(1)
        if subjects and subject not in subjects:
            continue
        paper_text = paper.read_text(encoding="utf-8", errors="ignore").strip()
        key_file = EXAM_KEY_DIR / paper.name.replace("试卷", "试卷答案")
        key_text = ""
        if key_file.exists():
            key_text = key_file.read_text(encoding="utf-8", errors="ignore").strip()
        # 尝试按题号切分（一、1.、(1) 等）
        questions = _split_exam_questions(paper_text)
        if not questions:
            # 无法切分：整卷存一个 chunk
            text = f"【试卷】{paper.stem}\n{paper_text[:2000]}"
            if key_text:
                text += f"\n【参考答案】\n{key_text[:2000]}"
            chunks.append({
                "_id": _stable_id(paper.stem),
                "text": text,
                "metadata": {
                    "doc_type": "exercise",
                    "stage": "初中",
                    "subject": subject,
                    "ques_type": "中考真题",
                    "difficulty": "",
                    "knowledges": "",
                    "source": "中考真题",
                    "source_file": paper.name,
                    "split": "exam",
                },
            })
            continue
        # 按题切块，尽量配对答案
        for q in questions:
            text = f"【题干】{q['text'][:1500]}"
            chunks.append({
                "_id": _stable_id(q["text"]),
                "text": text,
                "metadata": {
                    "doc_type": "exercise",
                    "stage": "初中",
                    "subject": subject,
                    "ques_type": "中考真题",
                    "difficulty": "",
                    "knowledges": "",
                    "source": "中考真题",
                    "source_file": paper.name,
                    "split": "exam",
                    "qno": q["qno"],
                },
            })
    return chunks


def _split_exam_questions(text: str) -> list[dict]:
    """按题号切分试卷文本，返回 [{"qno", "text"}, ...]。"""
    # 匹配常见题号：1. 2. / 1、2、 / (1)(2) / 一、二、
    pattern = re.compile(r"(?:^|\n)\s*(\d+)[.、．)]\s*(.+?)(?=\n\s*\d+[.、．)]|\Z)", re.S)
    matches = list(pattern.finditer(text))
    if len(matches) < 3:
        return []
    questions = []
    for mt in matches:
        qno = mt.group(1)
        body = mt.group(2).strip()
        if len(body) < 5:
            continue
        questions.append({"qno": qno, "text": body})
    return questions


# ---------------------------------------------------------------- 入库


def ingest(chunks: list[dict], db_path: Path, batch: int, use_json_id: bool) -> None:
    if not chunks:
        print("无数据可入库，退出")
        return
    print(f"初始化 ChromaDB: {db_path}")
    client = get_chroma_client(str(db_path))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL, normalize_embeddings=True,
    )
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing = set()
    try:
        existing = set(coll.get()["ids"])
    except Exception:  # noqa: BLE001
        pass
    print(f"DB 已有 {len(existing)} 条")

    if use_json_id:
        # 增量追加：用 chunk 自带 _id（稳定哈希），已存在则跳过
        todo = [c for c in chunks if c["_id"] not in existing]
    else:
        # 全新入库：清空后重灌
        if existing:
            print(f"清空旧数据 {len(existing)} 条...")
            coll.delete(ids=list(existing))
        todo = chunks
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
                elif isinstance(v, (list, dict)):
                    m[k] = json.dumps(v, ensure_ascii=False)
            metas.append(m)
        coll.add(ids=ids, documents=docs, metadatas=metas)
        done = i + len(b)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        eta = (len(todo) - done) / rate if rate else 0
        print(f"[{done}/{len(todo)}] {rate:.1f} chunks/s ETA {eta:.0f}s", flush=True)

    print(f"\n=== 完成 === 集合 {COLLECTION_NAME} 共 {coll.count()} 条, 耗时 {time.time()-t0:.0f}s")
    print(f"DB 路径: {db_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="题库入库 → ChromaDB k9_exercises 集合")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--source", choices=["jsonl", "exam", "all"], default="all",
                    help="数据源：jsonl=题库JSONL / exam=中考真题MD / all=全部")
    ap.add_argument("--subject", default="", help="学科过滤（数学/物理/...），留空=全部")
    ap.add_argument("--split", default="", help="split 过滤（train/test/valid），留空=全部")
    ap.add_argument("--use-json-id", action="store_true", default=True,
                    help="用稳定哈希 ID（增量追加，已存在跳过）")
    ap.add_argument("--no-json-id", dest="use_json_id", action="store_false",
                    help="不用稳定 ID（清空重灌）")
    ap.set_defaults(use_json_id=True)
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subject.split(",") if s.strip()] if args.subject else None
    splits = [s.strip() for s in args.split.split(",") if s.strip()] if args.split else list(SPLITS)

    chunks: list[dict] = []
    if args.source in ("jsonl", "all"):
        print(f"加载 JSONL 题库: subjects={subjects or '全部'} splits={splits}")
        c = load_jsonl_exercises(subjects, splits)
        print(f"  JSONL 题目 {len(c)} 条")
        chunks.extend(c)
    if args.source in ("exam", "all"):
        print(f"加载中考真题 MD: subjects={subjects or '全部'}")
        c = load_exam_md_exercises(subjects)
        print(f"  中考真题 {len(c)} 条")
        chunks.extend(c)

    print(f"\n合计 {len(chunks)} 条题目待入库")
    ingest(chunks, Path(args.db), args.batch, args.use_json_id)


if __name__ == "__main__":
    main()
