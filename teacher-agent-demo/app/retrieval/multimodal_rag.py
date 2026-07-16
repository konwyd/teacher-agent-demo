"""多模态 RAG：文本向量检索 + 关联图片返回。

方案：文本驱动图片检索（Text-to-Image via Metadata）
- 每个教材 chunk 的 metadata.images 记录了该 chunk 关联的图片文件名
- bge-m3 文本向量检索找到最相关 chunk → 从 metadata 提取图片路径
- 返回结果同时包含文本上下文和图片 URL/路径

图片路径构造规则：
  source_md = /data/.../parsed/{stem}/auto/{stem}.md
  images/{filename}  →  /data/.../parsed/{stem}/auto/images/{filename}

检索流程：
  query → bge-m3 编码 → ChromaDB k9_textbook cosine top-k
  → 对每个命中 chunk：
      text = chunk 原文
      images = [构造完整路径的图片URL列表]
  → 返回图文混合结果
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any

from app.retrieval.textbook_rag import get_textbook_rag, TextbookRAG

logger = logging.getLogger(__name__)

# 图片服务的 base URL（通过环境变量配置，供前端直接访问）
_IMAGE_BASE_URL = os.getenv("TEACHER_IMAGE_BASE_URL", "http://10.250.30.117:8000/images")

# 解析后教材的根目录
_PARSED_ROOT = Path("/data/education/data/china_textbook_k9_parsed")


def _image_url_from_source_md(source_md: str, filename: str) -> str:
    """从 source_md 路径和图片文件名构造可访问的 URL。

    source_md 例: /data/.../parsed/义务教育教科书·数学七年级上册/auto/xxx.md
    → stem = 义务教育教科书·数学七年级上册
    → 图片路径 = parsed/{stem}/auto/images/{filename}
    → URL = {base_url}/parsed/{stem}/images/{filename}
    """
    md_path = Path(source_md)
    # auto 目录的父目录就是 stem 目录
    auto_dir = md_path.parent  # .../parsed/{stem}/auto/
    stem_dir = auto_dir.parent  # .../parsed/{stem}/
    stem = stem_dir.name
    # URL 编码 stem 中的特殊字符（如 · 和空格）
    stem_encoded = urllib.parse.quote(stem, safe='')
    filename_encoded = urllib.parse.quote(filename, safe='')
    return f"{_IMAGE_BASE_URL}/parsed/{stem_encoded}/images/{filename_encoded}"


def _image_local_path(source_md: str, filename: str) -> Path:
    """构造图片的本地文件路径（用于服务端读取）。"""
    md_path = Path(source_md)
    auto_dir = md_path.parent
    stem_dir = auto_dir.parent
    return stem_dir / "auto" / "images" / filename


class MultimodalHit:
    """多模态检索命中结果。"""

    def __init__(self, text: str, meta: dict[str, Any], distance: float,
                 images: list[dict[str, str]]):
        self.text = text
        self.meta = meta
        self.distance = distance
        self.images = images  # [{"filename": "xxx.jpg", "url": "...", "local_path": "..."}, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "meta": self.meta,
            "distance": self.distance,
            "images": self.images,
        }


class MultimodalRAG:
    """多模态 RAG 检索器：文本向量检索 + 图片路径补充。

    复用 TextbookRAG 的 ChromaDB client 和 bge-m3 嵌入模型，
    在文本检索结果上附加图片路径。
    """

    def __init__(self) -> None:
        self._rag = get_textbook_rag()
        if self._rag is None:
            raise RuntimeError("TextbookRAG 不可用，MultimodalRAG 无法初始化")
        logger.info("MultimodalRAG 初始化完成")

    @property
    def is_available(self) -> bool:
        return self._rag is not None

    def retrieve(
        self,
        query: str,
        topk: int | None = None,
        where: dict[str, Any] | None = None,
        include_images: bool = True,
    ) -> list[dict[str, Any]]:
        """多模态检索。

        Args:
            query: 检索查询
            topk: 返回数量，默认使用配置的 TEACHER_RAG_TOPK
            where: ChromaDB 过滤条件（如 {"subject": "数学"}）
            include_images: 是否包含图片信息

        Returns:
            [{"text", "meta", "distance", "images": [{"filename", "url", "local_path"}, ...]}, ...]
        """
        hits = self._rag.retrieve(query, topk=topk, where=where)

        if not include_images:
            return [
                {"text": h["text"], "meta": h["meta"], "distance": h["distance"], "images": []}
                for h in hits
            ]

        result = []
        for h in hits:
            images = self._resolve_images(h.get("meta", {}))
            result.append({
                "text": h["text"],
                "meta": h["meta"],
                "distance": h["distance"],
                "images": images,
            })
        return result

    def _resolve_images(self, meta: dict[str, Any]) -> list[dict[str, str]]:
        """从 metadata 解析图片引用，构造完整 URL 和本地路径。"""
        images_field = meta.get("images", "")
        if not images_field:
            return []

        source_md = meta.get("source_md", "")
        if not source_md:
            return []

        # images 在 ChromaDB 中是逗号分隔的字符串
        if isinstance(images_field, list):
            filenames = images_field
        else:
            filenames = [f.strip() for f in str(images_field).split(",") if f.strip()]

        result = []
        for fname in filenames:
            local = _image_local_path(source_md, fname)
            url = _image_url_from_source_md(source_md, fname)
            exists = local.exists()
            result.append({
                "filename": fname,
                "url": url,
                "local_path": str(local),
                "exists": exists,
            })
        return result

    def format_context(
        self, hits: list[dict[str, Any]], max_chars: int = 2400, include_image_refs: bool = True
    ) -> str:
        """格式化为可注入 prompt 的教材上下文（含图片引用说明）。"""
        parts: list[str] = []
        used = 0
        for i, h in enumerate(hits, 1):
            m = h.get("meta", {}) or {}
            header = (
                f"{m.get('stage', '')}{m.get('subject', '')} "
                f"{m.get('grade', '')}{m.get('semester', '')} "
                f"《{m.get('chapter', '')}》"
            ).strip()
            block = f"[教材参考{i}] {header}\n{h.get('text', '')}"
            if include_image_refs:
                imgs = h.get("images", [])
                if imgs:
                    img_names = [img["filename"] for img in imgs if img.get("exists")]
                    if img_names:
                        block += f"\n[关联图片: {', '.join(img_names[:5])}{'...' if len(img_names) > 5 else ''}]"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def format_refs(self, hits: list[dict[str, Any]], limit: int = 3) -> list[str]:
        """简短引用（含图片数量）。"""
        refs: list[str] = []
        for h in hits[:limit]:
            m = h.get("meta", {}) or {}
            ref = (
                f"{m.get('subject', '')}·{m.get('grade', '')}{m.get('semester', '')} "
                f"《{m.get('chapter', '')}》(相似度{1 - h.get('distance', 0):.2f})"
            ).strip()
            imgs = h.get("images", [])
            n_imgs = sum(1 for img in imgs if img.get("exists"))
            if n_imgs:
                ref += f" 📷×{n_imgs}"
            refs.append(ref)
        return refs

    def format_image_urls(self, hits: list[dict[str, Any]], max_images: int = 5) -> list[str]:
        """提取所有命中结果的图片 URL（去重，供前端直接渲染）。"""
        seen: set[str] = set()
        urls: list[str] = []
        for h in hits:
            for img in h.get("images", []):
                if img.get("exists") and img["url"] not in seen:
                    seen.add(img["url"])
                    urls.append(img["url"])
                    if len(urls) >= max_images:
                        return urls
        return urls


# ---------------------------------------------------------------- 单例管理

_multimodal_rag: MultimodalRAG | None = None
_multimodal_tried: bool = False


def get_multimodal_rag() -> MultimodalRAG | None:
    """进程级单例。"""
    global _multimodal_rag, _multimodal_tried
    if _multimodal_tried:
        return _multimodal_rag
    _multimodal_tried = True
    try:
        _multimodal_rag = MultimodalRAG()
    except Exception as exc:
        logger.warning("MultimodalRAG 初始化失败: %s", exc)
        return None
    return _multimodal_rag


def reset_multimodal_rag() -> None:
    """重置缓存（测试用）。"""
    global _multimodal_rag, _multimodal_tried
    _multimodal_rag = None
    _multimodal_tried = False
