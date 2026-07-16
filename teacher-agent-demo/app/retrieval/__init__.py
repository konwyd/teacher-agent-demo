"""检索模块：Graph RAG（概念图 mock）+ Textbook RAG（教材向量检索）+ KBManager（多集合）。"""

from app.retrieval.graph_retriever import get_retriever
from app.retrieval.kb_manager import (
    COLLECTIONS,
    KBManager,
    get_kb_manager,
    get_kb_manager_forced,
    reset_kb_manager,
)
from app.retrieval.textbook_rag import (
    TextbookRAG,
    get_textbook_rag,
    is_rag_available,
    retrieve_textbook,
)

__all__ = [
    "get_retriever",
    "TextbookRAG",
    "get_textbook_rag",
    "is_rag_available",
    "retrieve_textbook",
    "KBManager",
    "COLLECTIONS",
    "get_kb_manager",
    "get_kb_manager_forced",
    "reset_kb_manager",
]
