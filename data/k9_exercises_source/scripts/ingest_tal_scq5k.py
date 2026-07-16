#!/usr/bin/env python3
"""TAL-SCQ5K 中文数学竞赛题 → ChromaDB k9_exercises 集合（追加）

来源: https://huggingface.co/datasets/math-eval/TAL-SCQ5K
好未来（学而思母公司）开源，MIT 许可
5,000 题，小学+初中+高中，选择题（含详细解题步骤）
"""

import json, hashlib, time, sys, os
from pathlib import Path

sys.path.insert(0, '/data/education/teacher-agent-demo')
from app.retrieval.chroma_compat import ensure_chroma_compat, get_chroma_client
ensure_chroma_compat()

import chromadb
from chromadb.utils import embedding_functions

DATA_FILES = ['/tmp/tal_scq5k_train.jsonl', '/tmp/tal_scq5k_test.jsonl']
DB_PATH = "/data/wyd-workspace/education/data/china_textbook_k9_vectordb"
EMBED_MODEL = "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3"
COLLECTION_NAME = "k9_exercises"
BATCH_SIZE = 256

# 学段推断（根据竞赛来源和知识点）
def infer_stage(problem, knowledge_routes):
    """从题目内容和知识点路径推断学段。"""
    text = problem + ' '.join(knowledge_routes)
    if any(k in text for k in ['小学', '三年级', '四年级', '五年级', '六年级']):
        return '小学'
    if any(k in text for k in ['初中', '七年级', '八年级', '九年级', '初一', '初二', '初三']):
        return '初中'
    if any(k in text for k in ['高中', '高一', '高二', '高三']):
        return '高中'
    return ''

def parse_options(option_list):
    """解析 TAL-SCQ5K 的嵌套选项格式。"""
    lines = []
    for opt_group in option_list:
        for opt in opt_group:
            lines.append(f"{opt['aoVal']}. {opt['content']}")
    return '\n'.join(lines)

# 加载数据
all_chunks = []
for fpath in DATA_FILES:
    if not os.path.exists(fpath):
        print(f"文件不存在: {fpath}")
        continue
    with open(fpath) as f:
        for line in f:
            d = json.loads(line.strip())
            
            options = parse_options(d.get('answer_option_list', []))
            answer = d.get('answer_value', '')
            analysis_list = d.get('answer_analysis', [])
            analysis = '\n'.join(analysis_list) if analysis_list else ''
            knowledge = ';'.join(d.get('knowledge_point_routes', []))
            stage = infer_stage(d.get('problem', ''), d.get('knowledge_point_routes', []))
            
            text_parts = [f"【题目】{d.get('problem', '')}"]
            if options:
                text_parts.append(options)
            if answer:
                text_parts.append(f"【答案】{answer}")
            if analysis:
                text_parts.append(f"【解析】{analysis[:300]}")
            
            meta = {
                "doc_type": "exercise",
                "subject": "Math",
                "ques_type": "选择题",
                "difficulty": d.get('difficulty', ''),
                "knowledges": knowledge,
                "source": "TAL-SCQ5K",
                "source_file": d.get('dataset_name', 'TAL-SCQ5K'),
                "question_id": d.get('qid', ''),
                "has_answer": bool(answer),
                "stage": stage,
            }
            
            chunk_id = hashlib.md5(f"TAL-SCQ5K|{d.get('queId','')}".encode()).hexdigest()[:16]
            all_chunks.append({
                "_id": chunk_id,
                "text": "\n".join(text_parts),
                "metadata": meta,
            })

print(f"加载题目: {len(all_chunks)} 条")

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
