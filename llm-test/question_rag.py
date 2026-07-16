"""
单题联网 RAG 推理脚本

用法:
  # 交互模式（输入题目，联网搜索后作答）
  python question_rag.py

  # 命令行传入题目
  python question_rag.py "已知x=2是方程x²-mx=6的一个根，则m的值为" -s 数学

  # 对比模式：同时输出无 RAG 和有 RAG 的答案
  python question_rag.py "简述光合作用的过程" -s 生物 --compare

  # 使用本地模型
  python question_rag.py "勾股定理是什么" -s 数学 --local

  # 指定搜索条数
  python question_rag.py "辛亥革命的意义" -s 历史 -k 8
"""

from __future__ import annotations

import argparse
import sys
import logging

logging.basicConfig(level=logging.WARNING)  # 默认不打印 debug 日志

# ── 复用已有模块 ──────────────────────────────────────────
from online_rag import get_online_rag
from llm_client import make_teacher_client, make_local_teacher_client


def build_rag_query(question: str, subject: str) -> str:
    """根据题目构造 Tavily 搜索 query（限制 400 字符）"""
    prefix = f"{subject} 知识点 考点 "
    max_q = 380 - len(prefix)
    # 优先用完整题目，超长则截断
    body = question if len(question) <= max_q else question[:max_q]
    return prefix + body


def search_and_format(question: str, subject: str, topk: int = 5):
    """执行联网搜索并格式化结果"""
    online = get_online_rag()
    if online is None:
        print("❌ 联网 RAG 不可用，请检查 config.py 中 Tavily API key 是否配置")
        return "", []

    query = build_rag_query(question, subject)
    print(f"🔍 搜索 query: {query}\n")

    hits = online.search(query, max_results=topk)

    if not hits:
        print("⚠️ 未搜索到结果")
        return "", []

    context = online.format_context(hits)
    refs = []
    print("📚 搜索结果:")
    for i, h in enumerate(hits[:min(len(hits), 5)], 1):
        m = h.get("meta", {}) or {}
        title = m.get("source", "未知来源")
        url = m.get("url", "")
        score = h.get("score", 0)
        print(f"  [{i}] {title}")
        if url:
            print(f"      {url}")
        print(f"      相似度: {score:.2f} | 内容长度: {len(h.get('text', ''))} 字")
        print()

        ref = f"🌐 {title}"
        if url:
            ref += f" ({url})"
        refs.append(ref)

    return context, refs


def answer_question(
    question: str,
    subject: str,
    rag_context: str = "",
    use_local: bool = False,
) -> str:
    """让模型回答单道题"""
    # 初始化模型
    if use_local:
        client = make_local_teacher_client()
    else:
        client = make_teacher_client()

    # 构造 prompt
    model_tag = "本地 DeepSeek-R1-7B" if use_local else "deepseek-chat"

    if rag_context:
        system_content = (
            f"你是一位{subject}学科的辅导老师。请参考以下从互联网检索到的相关资料，"
            f"回答学生的问题。\n\n"
            f"### 参考资料\n{rag_context}\n\n"
            f"---\n"
            f"回答要求：\n"
            f"1. 先给出最终答案\n"
            f"2. 然后结合参考资料解释解题思路或知识要点\n"
            f"3. 不要编造资料中没有的信息"
        )
    else:
        system_content = (
            f"你是一位{subject}学科的辅导老师。请根据你的知识回答学生的问题。\n\n"
            f"回答要求：\n"
            f"1. 先给出最终答案\n"
            f"2. 然后解释解题思路或知识要点"
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]

    rag_label = " [联网RAG]" if rag_context else ""
    print(f"🤖 模型推理中 ({model_tag}{rag_label})...\n")
    answer = client.chat_with_retry(messages)
    return answer


# ── 交互模式 ──────────────────────────────────────────────

def interactive_mode(args):
    """交互式输入题目"""
    subject = args.subject or input("📘 科目: ").strip()
    if not subject:
        print("科目不能为空")
        return

    print("\n📝 请输入题目（输入 END 结束）:")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    question = "\n".join(lines).strip()
    if not question:
        print("题目不能为空")
        return

    run_single(question, subject, args)


# ── 单题推理 ──────────────────────────────────────────────

def run_single(question: str, subject: str, args):
    """执行单题推理"""
    print(f"{'='*60}")
    print(f"📘 科目: {subject}")
    print(f"📝 题目: {question}")
    print(f"{'='*60}\n")

    # 联网搜索
    rag_context, refs = search_and_format(
        question, subject, topk=args.topk
    )

    if args.compare:
        # ── 对比模式：先无 RAG，再有 RAG ──
        print(f"\n{'─'*60}")
        print("  第一轮：无 RAG（纯模型知识）")
        print(f"{'─'*60}\n")
        answer_no_rag = answer_question(
            question, subject, rag_context="", use_local=args.local
        )
        print(f"{'─'*60}")
        print("【无 RAG 答案】")
        print(f"{'─'*60}")
        print(answer_no_rag)

        print(f"\n{'─'*60}")
        print("  第二轮：挂载联网 RAG")
        print(f"{'─'*60}\n")
        answer_rag = answer_question(
            question, subject, rag_context=rag_context, use_local=args.local
        )
        print(f"{'─'*60}")
        print("【联网 RAG 答案】")
        print(f"{'─'*60}")
        print(answer_rag)

        # 对比摘要
        print(f"\n{'='*60}")
        print("📊 对比摘要")
        print(f"{'='*60}")
        print(f"  无RAG 答案长度: {len(answer_no_rag)} 字")
        print(f"  有RAG 答案长度: {len(answer_rag)} 字")
        print(f"  参考资料条数: {len(refs)} 条")

    else:
        # ── 单模式 ──
        answer = answer_question(
            question, subject, rag_context=rag_context, use_local=args.local
        )
        print(f"{'─'*60}")
        print("【答案】")
        print(f"{'─'*60}")
        print(answer)

    # 输出引用来源
    if refs:
        print(f"\n{'='*60}")
        print("🔗 参考资料")
        print(f"{'='*60}")
        for i, ref in enumerate(refs, 1):
            print(f"  [{i}] {ref}")


# ── CLI 入口 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="单题联网 RAG 推理 —— 搜索互联网资料辅助模型答题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python question_rag.py "勾股定理的应用" -s 数学
  python question_rag.py "光合作用的过程" -s 生物 --compare
  python question_rag.py "辛亥革命的历史意义" -s 历史 -k 8 --compare
  python question_rag.py -s 数学 --local          # 进入交互模式，使用本地模型
        """,
    )
    parser.add_argument(
        "question", nargs="?", default=None,
        help="题目内容（不传则进入交互模式）",
    )
    parser.add_argument(
        "-s", "--subject", default=None,
        help="学科名称（如 数学、历史、物理）",
    )
    parser.add_argument(
        "-k", "--topk", type=int, default=5,
        help="联网搜索结果数量（默认 5）",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="对比模式：同时输出无 RAG 和联网 RAG 的答案",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="使用本地 DeepSeek-R1-7B 模型（默认用 API）",
    )

    args = parser.parse_args()

    if not args.question:
        interactive_mode(args)
    else:
        if not args.subject:
            print("❌ 请用 -s 指定学科，如: -s 数学")
            sys.exit(1)
        run_single(args.question, args.subject, args)


if __name__ == "__main__":
    main()
