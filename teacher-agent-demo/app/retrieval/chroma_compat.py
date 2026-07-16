"""ChromaDB 兼容层：强制 SegmentAPI + 兼容旧版数据格式。

背景：
  chromadb 1.5.9 默认 RustBindingsAPI 后端在读取旧版 hnswlib 格式数据时
  segfault。现有向量库（k9_textbook / k9_exercises 等）均为旧版 SegmentAPI +
  chroma-hnswlib 写入，存在两处不兼容：
    1. Rust 后端不兼容旧 hnswlib 索引格式 → 切回 SegmentAPI；
    2. 旧版 `index_metadata.pickle` 存的是 dict（且 dimensionality=None），
       新版 PersistentData.load_from_file 期望 PersistentData 对象 → patch 转换。

本模块（必须在 `import chromadb` 之前调用 ensure_chroma_compat()）：
  1. 设置 CHROMA_API_IMPL=chromadb.api.segment.SegmentAPI；
  2. patch PersistentData.load_from_file：旧版 dict → PersistentData 对象，
     dimensionality 为 None 时从 chroma.sqlite3 的 collections.dimension 读取。

依赖：chroma-hnswlib（conda-forge 预编译，提供 file_handle_count 等扩展方法）。
"""
from __future__ import annotations

import os
import pickle
import sqlite3
from typing import Any

_PATCHED = False


def get_chroma_client(path: str = "") -> Any:
    """获取 chromadb client：优先 HttpClient（server 模式，持久化可靠），fallback PersistentClient。

    通过环境变量 CHROMA_SERVER=host:port 指定 server 地址。
    server 模式下 chroma server 进程持续运行、定期 flush，避免嵌入式 Rust 后端
    进程退出丢数据的问题。
    """
    import chromadb

    server = os.environ.get("CHROMA_SERVER", "").strip()
    if server:
        host, _, port = server.partition(":")
        return chromadb.HttpClient(host=host or "127.0.0.1", port=int(port) if port else 8000)
    return chromadb.PersistentClient(path=path)


def ensure_chroma_compat() -> None:
    """切到 SegmentAPI 并兼容旧版 pickle 格式（幂等，可多次调用）。"""
    global _PATCHED
    # 注：旧 hnsw index 在 chromadb 1.5.9 下已不可用（Rust/SegmentAPI 均 crash），
    # 向量库已从源数据用 Rust 后端全新重建。此处不再强制 SegmentAPI，使用默认 Rust 后端。
    # PersistentData patch 保留，仅对旧版 dict pickle 生效（全新库无影响）。
    if _PATCHED:
        return
    _PATCHED = True
    # 2. patch PersistentData.load_from_file：旧版 dict → PersistentData 对象
    try:
        from chromadb.segment.impl.vector.local_persistent_hnsw import PersistentData

        _orig_load = PersistentData.load_from_file

        @staticmethod
        def _patched_load_from_file(filename: str):  # type: ignore[override]
            with open(filename, "rb") as f:
                ret = pickle.load(f)
            if not isinstance(ret, dict):
                return ret
            # 旧版 chromadb 存 dict，转换成 PersistentData 对象
            dim = ret.get("dimensionality")
            if dim is None:
                # 旧版 pickle dimensionality=None，从 chroma.sqlite3 collections 表读真实维度
                dim = _read_collection_dimension(filename)
            pd = PersistentData(
                dimensionality=dim,
                total_elements_added=ret.get("total_elements_added", 0),
                id_to_label=ret.get("id_to_label", {}),
                label_to_id=ret.get("label_to_id", {}),
                id_to_seq_id=ret.get("id_to_seq_id", {}),
            )
            pd.max_seq_id = ret.get("max_seq_id")
            return pd

        PersistentData.load_from_file = _patched_load_from_file  # type: ignore[assignment]
    except Exception:  # pragma: no cover - patch 失败不阻断，chromadb 自会报错
        pass


def _read_collection_dimension(metadata_path: str) -> int | None:
    """从 index_metadata.pickle 路径反查 collection 的 dimension。

    metadata_path 形如 .../vectordb/{segment_uuid}/index_metadata.pickle
    → 用 segment_uuid 查 segments.collection → collections.dimension。
    """
    try:
        seg_dir = os.path.dirname(metadata_path)
        seg_uuid = os.path.basename(seg_dir)
        db_root = os.path.dirname(seg_dir)
        db_file = os.path.join(db_root, "chroma.sqlite3")
        con = sqlite3.connect(db_file)
        row = con.execute(
            "SELECT c.dimension FROM segments s "
            "JOIN collections c ON c.id = s.collection WHERE s.id = ?",
            (seg_uuid,),
        ).fetchone()
        con.close()
        return row[0] if row else None
    except Exception:  # noqa: BLE001
        return None
