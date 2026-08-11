#!/usr/bin/env python3
"""河南中考真题 → ChromaDB k9_exercises 集合（v4：整卷入库，7 条）

v3（旧）：按大题切分，150 条碎片，每条 30~2600 字
v4（新）：整张试卷 + 完整答案作为一条 chunk，7 条，每条 6000~21000 字
"""

import re, hashlib, time, sys, os
from pathlib import Path

# ---- 路径配置 ----
PAPER_DIR = Path(__file__).parent / "testmd" / "paper"
KEY_DIR = Path(__file__).parent / "testmd" / "key"

# 使用 Chroma Server（已启动在 8002 端口）
CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8002
COLLECTION_NAME = "k9_exercises"
EMBED_MODEL = "/home/dockerContainerSave/.cache/modelscope/BAAI/bge-m3"

SUBJECT_MAP = {
    "数学": "Math", "物理": "Physics", "化学": "Chemistry",
    "英语": "English", "语文": "Chinese", "历史": "History",
    "道法": "Politics",
}

def clean_text(text: str) -> str:
    """清理 CDN 图片链接"""
    text = re.sub(r'!\[image\]\(https?://[^)]+\)', '[图片]', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ---- 加载试卷 ----
all_chunks = []
for paper_file in sorted(PAPER_DIR.glob("*.md")):
    fname = paper_file.stem
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

    # 整卷：试卷 + 完整答案
    full_text = f"# 2026年河南中考{subject_cn}试卷\n\n{paper_text}"
    if answer_text:
        full_text += f"\n\n# 参考答案及评分标准\n\n{answer_text}"
    full_text = clean_text(full_text)

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

    chunk_id = hashlib.md5(f"zhongkao_v4|{fname}".encode()).hexdigest()[:16]
    all_chunks.append({
        "_id": chunk_id,
        "text": full_text,
        "metadata": meta,
    })
    print(f"  {subject_cn}: {len(full_text)} 字（含答案: {bool(answer_text)}）")

print(f"\n总 chunk 数: {len(all_chunks)}")

# ---- 入库到 Chroma Server ----
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL, normalize_embeddings=True,
)
coll = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

# 先删除旧的中考真题（source=="中考真题" 的碎片）
print("\n删除旧中考真题...")
old = coll.get(where={"source": "中考真题"})
old_ids = old.get("ids", [])
if old_ids:
    coll.delete(ids=old_ids)
    print(f"  已删除 {len(old_ids)} 条旧碎片")

# 入库新数据
print(f"\n入库 {len(all_chunks)} 条完整试卷...")
t0 = time.time()
ids = [c["_id"] for c in all_chunks]
docs = [c["text"] for c in all_chunks]
metas = [c["metadata"] for c in all_chunks]
coll.add(ids=ids, documents=docs, metadatas=metas)
elapsed = time.time() - t0
print(f"完成！集合 {COLLECTION_NAME} 共 {coll.count()} 条, 耗时 {elapsed:.0f}s")
