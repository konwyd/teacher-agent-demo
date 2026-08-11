#!/usr/bin/env python3
"""APE210K 中文数学应用题 → ChromaDB k9_exercises 集合（追加）

来源: https://github.com/Chenny0808/ape210k
200,488 道中文数学应用题（非选择题！含方程和答案）
"""

import json, hashlib, time, sys, os
from pathlib import Path

sys.path.insert(0, '/data/education/teacher-agent-demo')
from app.retrieval.chroma_compat import ensure_chroma_compat, get_chroma_client
ensure_chroma_compat()

import chromadb
from chromadb.utils import embedding_functions

DATA_FILE = '/tmp/ape210k_train.jsonl'
DB_PATH = "/data/wyd-workspace/education/data/china_textbook_k9_vectordb"
EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
COLLECTION_NAME = "k9_exercises"
BATCH_SIZE = 256

# 推断学段（根据题目内容和方程复杂度）
def infer_stage(text, equation, ans):
    text = (text or '').lower()
    eq = (equation or '')
    ans_val = (ans or '')
    # 高中特征
    if any(k in text for k in ['log', 'ln', 'sin', 'cos', 'tan', '导数', '积分',
                                 '椭圆', '双曲线', '抛物线', '向量', '复数']):
        return '高中'
    # 初中特征
    if any(k in text for k in ['一元二次', '勾股', '相似三角形', '三角函数', '二次函数',
                                 '方差', '标准差', '概率']):
        return '初中'
    if 'x^2' in eq or '**2' in eq or 'sqrt' in eq:
        return '初中'
    # 小学特征
    if any(k in text for k in ['三年级', '四年级', '五年级', '六年级']):
        return '小学'
    return '小学'  # APE210K 主要是小学初中

all_chunks = []
with open(DATA_FILE) as f:
    for line in f:
        d = json.loads(line.strip())
        text = d.get('original_text', d.get('segmented_text', ''))
        equation = d.get('equation', '')
        ans = d.get('ans', '')
        stage = infer_stage(text, equation, ans)
        
        text_parts = [f"【题目】{text}"]
        if equation:
            text_parts.append(f"【方程】{equation}")
        if ans:
            text_parts.append(f"【答案】{ans}")
        
        meta = {
            "doc_type": "exercise",
            "subject": "Math",
            "ques_type": "应用题",
            "difficulty": "",
            "knowledges": "",
            "source": "APE210K",
            "source_file": "train.ape.json",
            "question_id": d.get('id', ''),
            "has_answer": bool(ans),
            "stage": stage,
        }
        
        chunk_id = hashlib.md5(f"APE210K|{d.get('id','')}".encode()).hexdigest()[:16]
        all_chunks.append({
            "_id": chunk_id,
            "text": "\n".join(text_parts),
            "metadata": meta,
        })

print(f"加载题目: {len(all_chunks)} 条")

# 学段统计
from collections import Counter
stage_cnt = Counter(c['metadata']['stage'] for c in all_chunks)
print(f"学段分布: {dict(stage_cnt)}")

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
    if done % (BATCH_SIZE * 20) == 0 or done == len(todo):
        print(f"[{done}/{len(todo)}] {rate:.1f} chunks/s ETA {eta:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"\n=== 完成 === 集合 {COLLECTION_NAME} 共 {coll.count()} 条, 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
