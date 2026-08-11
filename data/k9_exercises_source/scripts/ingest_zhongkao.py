#!/usr/bin/env python3
"""河南中考真题 → ChromaDB k9_exercises 集合（追加）

来源：/data/education/llm-test/testmd/
  paper/  → 试卷 MD（题目）
  key/    → 答案 MD（答案+解析）
"""

import re, hashlib, time, sys, os
from pathlib import Path

sys.path.insert(0, '/data/education/teacher-agent-demo')
from app.retrieval.chroma_compat import ensure_chroma_compat, get_chroma_client
ensure_chroma_compat()

import chromadb
from chromadb.utils import embedding_functions

PAPER_DIR = Path("/data/education/llm-test/testmd/paper")
KEY_DIR = Path("/data/education/llm-test/testmd/key")
DB_PATH = "/data/wyd-workspace/education/data/china_textbook_k9_vectordb"
EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
COLLECTION_NAME = "k9_exercises"
BATCH_SIZE = 128

SUBJECT_MAP = {
    "数学": "Math", "物理": "Physics", "化学": "Chemistry",
    "英语": "English", "语文": "Chinese", "历史": "History",
    "道法": "Politics",
}

def parse_questions(md_text: str, subject: str) -> list[dict]:
    """从 MD 中按 ## 标题切分题目。"""
    chunks = []
    # 按 ## 标题切分（每个大题一个chunk）
    sections = re.split(r'\n(?=## )', md_text)
    for i, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < 20:
            continue
        # 清理 CDN 图片链接（保留文字描述）
        clean = re.sub(r'!\[image\]\(https?://[^)]+\)', '[图片]', section)
        clean = re.sub(r'\n{3,}', '\n\n', clean)
        chunks.append({
            "text": clean,
            "idx": i,
        })
    return chunks

# 加载试卷
all_chunks = []
for paper_file in sorted(PAPER_DIR.glob("*.md")):
    fname = paper_file.stem
    # 提取学科：2026数学试卷 → 数学
    m = re.match(r'2026(.+?)试卷', fname)
    if not m:
        continue
    subject_cn = m.group(1)
    subject_en = SUBJECT_MAP.get(subject_cn, subject_cn)
    
    paper_text = paper_file.read_text(encoding='utf-8', errors='ignore')
    
    # 找对应答案
    key_file = KEY_DIR / f"{fname}答案.md"
    answer_text = ""
    if key_file.exists():
        answer_text = key_file.read_text(encoding='utf-8', errors='ignore')
    
    # 按大题切分
    questions = parse_questions(paper_text, subject_cn)
    
    for q in questions:
        text_parts = [q["text"]]
        if answer_text:
            text_parts.append(f"\n【参考答案】\n{answer_text[:500]}")
        
        meta = {
            "doc_type": "exercise",
            "subject": subject_en,
            "ques_type": "中考真题",
            "difficulty": "中等",
            "knowledges": f"河南中考{subject_cn}",
            "source": "中考真题",
            "source_file": fname,
            "has_answer": bool(answer_text),
            "stage": "初中",
        }
        
        chunk_id = hashlib.md5(f"zhongkao|{fname}|{q['idx']}".encode()).hexdigest()[:16]
        all_chunks.append({
            "_id": chunk_id,
            "text": "\n".join(text_parts),
            "metadata": meta,
        })
    
    print(f"  {subject_cn}: {len(questions)} 个大题")

print(f"\n总 chunk 数: {len(all_chunks)}")

# 入库
client = get_chroma_client(DB_PATH)
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
except Exception:
    pass
print(f"DB 已有 {len(existing)} 条")

todo = [c for c in all_chunks if c["_id"] not in existing]
print(f"待入库 {len(todo)} 条")
if not todo:
    print("全部已入库")
    sys.exit(0)

t0 = time.time()
for i in range(0, len(todo), BATCH_SIZE):
    batch = todo[i:i+BATCH_SIZE]
    ids = [c["_id"] for c in batch]
    docs = [c["text"] for c in batch]
    metas = [dict(c["metadata"]) for c in batch]
    coll.add(ids=ids, documents=docs, metadatas=metas)
    done = i + len(batch)
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed else 0
    eta = (len(todo) - done) / rate if rate else 0
    print(f"[{done}/{len(todo)}] {rate:.1f} chunks/s ETA {eta:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"\n=== 完成 === 集合 {COLLECTION_NAME} 共 {coll.count()} 条, 耗时 {elapsed:.0f}s")
