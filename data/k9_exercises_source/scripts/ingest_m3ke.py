#!/usr/bin/env python3
"""M3KE 数据集 → ChromaDB k9_exercises 集合

数据源: https://github.com/tjunlp-lab/M3KE
格式: 四选一单选题，20,477 题，71 学科
"""

import json, hashlib, time, os, sys
from pathlib import Path
from collections import defaultdict

# 项目路径
sys.path.insert(0, '/data/education/teacher-agent-demo')
from app.retrieval.chroma_compat import ensure_chroma_compat, get_chroma_client
ensure_chroma_compat()

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path("/tmp/m3ke_data")
DB_PATH = "/data/wyd-workspace/education/data/china_textbook_k9_vectordb"
EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
COLLECTION_NAME = "k9_exercises"
BATCH_SIZE = 256

# ---- 1. 加载 dev 集的答案 ----
dev_answers: dict[str, dict[str, str]] = defaultdict(dict)
for f in sorted((DATA_DIR / "dev").glob("*.jsonl")):
    with open(f) as fh:
        for line in fh:
            d = json.loads(line)
            key = f"{f.stem}|{d['id']}"
            dev_answers[f.stem][str(d['id'])] = d.get("answer", "")

print(f"dev 集: {len(dev_answers)} 个学科, 共 {sum(len(v) for v in dev_answers.values())} 条答案")

# ---- 2. 加载 test 集并补充答案 ----
all_chunks = []
stats = defaultdict(lambda: {"total": 0, "with_answer": 0, "without_answer": 0})

for f in sorted((DATA_DIR / "test").glob("*.jsonl")):
    fname = f.stem
    dev_for_file = dev_answers.get(fname, {})
    with open(f) as fh:
        for line in fh:
            d = json.loads(line)
            qid = str(d['id'])
            answer = d.get("answer", "") or dev_for_file.get(qid, "")
            
            # 拼接题干+选项为检索文本
            text_parts = [f"【题目】{d['question']}"]
            for opt in ('A', 'B', 'C', 'D'):
                if d.get(opt):
                    text_parts.append(f"{opt}. {d[opt]}")
            if answer:
                text_parts.append(f"【答案】{answer}")
            
            # 从文件名解析元数据
            # 格式: "Math-Natural Sciences-Junior high school"
            parts = fname.split("-")
            subject = parts[0] if parts else fname
            
            stats[fname]["total"] += 1
            if answer:
                stats[fname]["with_answer"] += 1
            else:
                stats[fname]["without_answer"] += 1
            
            meta = {
                "doc_type": "exercise",
                "subject": subject,
                "ques_type": "单选题",
                "difficulty": "",
                "knowledges": "",
                "source": "M3KE",
                "source_file": fname,
                "question_id": d['id'],
                "has_answer": bool(answer),
            }
            
            chunk_id = hashlib.md5(f"M3KE|{fname}|{d['id']}".encode()).hexdigest()[:16]
            all_chunks.append({
                "_id": chunk_id,
                "text": "\n".join(text_parts),
                "metadata": meta,
            })

print(f"\ntest 集: 共 {len(all_chunks)} 题")
print(f"\n各学科统计:")
for fname in sorted(stats):
    s = stats[fname]
    print(f"  {fname:60s}  {s['total']:>4d} 题  (有答案:{s['with_answer']}, 无答案:{s['without_answer']})")

# ---- 3. 入库 ChromaDB ----
print(f"\n入库到 ChromaDB: {DB_PATH}")
client = get_chroma_client(DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL, normalize_embeddings=True,
)
coll = client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"},
)

# 跳过已存在的
existing = set()
try:
    existing = set(coll.get()["ids"])
except Exception:
    pass
print(f"DB 已有 {len(existing)} 条")

todo = [c for c in all_chunks if c["_id"] not in existing]
print(f"待入库 {len(todo)} 条")
if not todo:
    print("全部已入库，退出")
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
print(f"\n=== 完成 === 集合 {COLLECTION_NAME} 共 {coll.count()} 条, 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
