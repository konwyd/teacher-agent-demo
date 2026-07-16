"""
较难题测评脚本：对 JSONL 中的难题批量推理 + 自动评分

用法:
  python hard_question_eval.py                           # 无 RAG，全部 56 题
  python hard_question_eval.py --rag --online-rag auto     # 阈值触发联网 RAG
  python hard_question_eval.py --rag --online-rag online   # 强制联网 RAG
  python hard_question_eval.py --compare                  # 对比模式（无RAG vs 有RAG）
  python hard_question_eval.py --local                    # 本地模型
  python hard_question_eval.py --limit 5                  # 只跑前 5 题（调试用）
"""

from __future__ import annotations

import json
import os
import re
import sys
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)

from llm_client import make_teacher_client, make_judge_client, make_local_teacher_client
from rag_retriever import retrieve_for_exam, is_rag_available
from online_rag import get_online_rag, is_online_rag_available

# ── 配置 ──────────────────────────────────────────────────
JSONL_PATH = "testmd/json/初中数学_较难.jsonl"
OUTPUT_DIR = "results/hard_questions"

# ── 工具 ──────────────────────────────────────────────────

def load_questions(path: str) -> list[dict]:
    """加载 JSONL 题目"""
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def format_answer_key(question: dict) -> str:
    """将标准答案格式化为评分参考"""
    answers = question.get("ques_answer", [])
    analyze = question.get("ques_analyze", "")

    lines = ["## 参考答案"]
    lines.append(f"**题型**: {question.get('ques_type', '未知')}")
    lines.append(f"**正确答案**: {'；'.join(answers)}")
    if analyze:
        lines.append(f"\n**解析**: {analyze}")
    lines.append(f"\n**知识点**: {', '.join(question.get('ques_knowledges', []))}")
    return "\n".join(lines)


def strip_options(text: str) -> str:
    """把题目中的选项信息适度压缩，保留题目主干"""
    # 去掉 "选项: A. ... B. ..." 这种格式
    text = re.sub(r'\s*选项:\s*.*$', '', text, flags=re.DOTALL)
    # 去掉 "选项: A. ... B. ..." 的变体
    text = re.sub(r'\s*选项：\s*.*$', '', text, flags=re.DOTALL)
    return text.strip()


def build_teacher_prompt(question: dict, rag_context: str = "") -> list[dict]:
    """为一道题构造教师答题 prompt"""
    q_type = question.get("ques_type", "解答题")
    content = strip_options(question.get("ques_content", ""))

    if rag_context:
        system = (
            f"你是一位初中数学教师。以下是从题库中检索到的参考资料（含相关题目、答案和解析），"
            f"请仔细对比当前题目与参考题目：\n"
            f"- 如果参考题目与当前题目高度相似或几乎一模一样，请参考题库中给出的答案和解题过程，"
            f"在此基础上理解并作答。但如果你经过仔细验算后确信题库答案有误，请相信自己的验算结果。\n"
            f"- 如果参考题目只是知识点相关而非同一道题，则作为知识点参考。\n\n"
            f"### 参考资料\n{rag_context}\n\n"
            f"---\n"
            f"答题要求：\n"
            f"1. 这是一道{q_type}，请先给出最终答案\n"
            f"2. 如果是解答题，请写出完整的解题步骤\n"
            f"3. 如果涉及多个子问题，请分别标注答案\n"
            f"4. 选择题只写选项字母即可，填空题直接写数值"
        )
    else:
        system = (
            f"你是一位初中数学教师。请回答下面的数学题。\n\n"
            f"答题要求：\n"
            f"1. 这是一道{q_type}，请先给出最终答案\n"
            f"2. 如果是解答题，请写出完整的解题步骤\n"
            f"3. 如果涉及多个子问题，请分别标注答案\n"
            f"4. 选择题只写选项字母即可，填空题直接写数值"
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def build_judge_prompt(teacher_answer: str, question: dict) -> list[dict]:
    """构造逐题评分 prompt"""
    answer_key = format_answer_key(question)
    q_type = question.get("ques_type", "解答题")

    system = {
        "role": "system",
        "content": (
            "你是一位严格的初中数学裁判。请根据参考答案和解析，"
            "对教师的作答进行评分。\n\n"
            "评分依据：以参考答案为评分标准。逐项对比教师答案与参考答案，"
            "判断教师的最终结论是否与参考答案一致。"
            "如果教师答案与参考答案不一致即判错扣分。\n\n"
            "评分规则：\n"
            "- 解答题：逐子问题比对最终答案，最终答案与参考答案不一致即判错扣分；"
            "步骤正确但最终答案错误，最多给一半分数\n"
            "- 填空题/选择题：答案必须与参考答案完全一致，不一致直接 0 分\n"
            "- 多个子问题的解答题：每个子问题独立评分，最终汇总\n"
            "- 答案正确但步骤不完整，扣少量分\n"
            "- 答案明显错误或无关，给 0 分\n"
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"请评阅以下{q_type}：\n\n"
            f"{answer_key}\n\n"
            f"---\n\n"
            f"## 教师答案\n{teacher_answer}\n\n"
            f"请输出评分结果，格式为：\n"
            f"**得分**: X/10\n"
            f"**评分理由**: （简短说明为什么给这个分数）"
        ),
    }
    return [system, user]


def extract_question_score(grading_result: str) -> tuple[int | None, str]:
    """从评分结果中提取得分和理由"""
    score_match = re.search(r'得分[：:]\s*(\d+(?:\.\d+)?)\s*/\s*(\d+)', grading_result)
    if not score_match:
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+)', grading_result)

    score = float(score_match.group(1)) if score_match else None
    total = int(score_match.group(2)) if score_match else 10

    reason_match = re.search(r'评分理由.*?[：:]\s*(.+)', grading_result, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else grading_result

    return (int(score) if score else None, total, reason)


# ── 单题评测 ──────────────────────────────────────────────

def evaluate_one_question(
    question: dict,
    idx: int,
    teacher_client,
    judge_client,
    use_rag: bool = False,
    online_mode: str | None = None,
    verbose: bool = False,
) -> dict:
    """评测一道题，返回结果字典"""
    q_type = question.get("ques_type", "?")
    content = strip_options(question.get("ques_content", ""))

    # 1. RAG 检索（auto 模式会先查本地 ChromaDB，阈值不够才联网）
    rag_context = ""
    rag_refs: list[str] = []
    if use_rag:
        if verbose:
            print(f"\n═══ Q{idx+1} {q_type} ═══")
        rag_context, rag_refs, _ = retrieve_for_exam(
            content, "初中数学", topk=5, online_mode=online_mode, verbose=verbose,
        )

    # 2. 教师模型答题（始终注入 RAG 上下文）
    messages = build_teacher_prompt(question, rag_context)
    teacher_answer = teacher_client.chat_with_retry(messages)

    # 3. 裁判评分
    judge_messages = build_judge_prompt(teacher_answer, question)
    grading_result = judge_client.chat_with_retry(judge_messages)

    score, total, reason = extract_question_score(grading_result)

    return {
        "index": idx,
        "ques_type": q_type,
        "ques_content": content,
        "ques_answer": question.get("ques_answer", []),
        "teacher_answer": teacher_answer,
        "grading_result": grading_result,
        "score": score,
        "total": total,
        "reason": reason,
        "rag_enabled": use_rag,
        "rag_refs": rag_refs,
        "rag_context_snippet": rag_context[:600] if rag_context else "",
        "rag_context_full": rag_context,
        "rag_is_online": online_mode in ("online", "auto") and any("Tavily" in r or "zhangmen" in r or "http" in r for r in rag_refs),
    }


# ── 批量运行 ──────────────────────────────────────────────

def run_batch(
    questions: list[dict],
    teacher_client,
    judge_client,
    use_rag: bool = False,
    online_mode: str | None = None,
    desc: str = "",
    verbose: bool = False,
) -> list[dict]:
    """批量评测"""
    results = []
    rag_label = " [RAG]" if use_rag else ""
    total = len(questions)

    with tqdm(total=total, desc=f"📋 {desc}{rag_label}", unit="题",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for i, q in enumerate(questions):
            pbar.set_description(f"🔍 Q{i+1} {q.get('ques_type','')[:4]}{rag_label}")

            result = evaluate_one_question(
                q, i, teacher_client, judge_client,
                use_rag=use_rag, online_mode=online_mode, verbose=verbose,
            )
            results.append(result)

            score_str = f"{result['score']}/{result['total']}" if result['score'] is not None else "?"
            tqdm.write(f"  Q{i+1} {result['ques_type']} 得分: {score_str} | {result['reason'][:60]}")
            pbar.update(1)

    return results


# ── 汇总 ──────────────────────────────────────────────────

def _normalize_answer(text: str) -> str:
    """将答案归一化，去掉 LaTeX、空白、多余标点"""
    text = re.sub(r'\\\\?frac\{[^}]*\}\{[^}]*\}', '', text)
    text = re.sub(r'\\\\?left\|?', '', text)
    text = re.sub(r'\\\\?right\|?', '', text)
    text = re.sub(r'\\\\?[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\\\?[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '').replace('$', '')
    text = re.sub(r'\s+', '', text).strip()
    return text


def compute_hit_rate(results: list[dict]) -> dict:
    """计算 RAG 命中率

    两层判断：
      1. 字面命中（Literal Match）—— 答案值直接出现在检索上下文里（已去LaTeX/空白）
      2. 语义命中（Semantic Match）—— bge-m3 计算 题目+答案 vs 检索片段的余弦相似度 ≥ 0.65

    命中定义：任一判断通过即为命中。

    返回统计字典：
      total / hits / hit_rate / per_question / literal_hits / semantic_hits
    """
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(
            "/home/lizhaoyuan/.cache/modelscope/BAAI/bge-m3", device="cpu"
        )
    except Exception:
        print("⚠️  无法加载 bge-m3 模型，仅做字面命中检查")
        model = None

    per_question: list[dict] = []
    total = 0
    hits = 0
    literal_hits = 0
    semantic_hits = 0

    for r in results:
        if not r.get("rag_enabled"):
            continue

        rag_ctx = r.get("rag_context_full", "")
        if not rag_ctx:
            continue

        answers = r.get("ques_answer", [])
        if not answers:
            continue

        total += 1

        # ── 第一层：字面命中 ──
        ctx_clean = _normalize_answer(rag_ctx)
        any_literal = False
        matched_answers = []
        for ans in answers:
            ans_norm = _normalize_answer(str(ans))
            if len(ans_norm) >= 1 and ans_norm in ctx_clean:
                any_literal = True
                matched_answers.append(str(ans))

        # ── 第二层：语义命中 ──
        max_sim = 0.0
        best_chunk = ""
        is_semantic = False

        if model is not None:
            # 拆分 context 为片段
            chunks = [c.strip() for c in rag_ctx.split("\n\n") if c.strip()]
            sub_chunks = []
            for ch in chunks:
                parts = re.split(r'(?=\[)', ch)
                for p in parts:
                    p = p.strip()
                    if len(p) >= 20:
                        sub_chunks.append(p)

            if sub_chunks:
                # 用 题目片段+答案 做 query（比纯答案更有语义信息）
                ques_snippet = r.get("ques_content", "")[:200]
                answer_str = " ".join(str(a) for a in answers)[:100]
                query_text = f"{ques_snippet} 答案: {answer_str}"

                all_texts = [query_text] + sub_chunks
                try:
                    embeddings = model.encode(all_texts, normalize_embeddings=True)
                    answer_emb = embeddings[0]
                    chunk_embs = embeddings[1:]
                    scores = np.dot(chunk_embs, answer_emb)
                    max_sim = float(scores.max()) if len(scores) > 0 else 0.0
                    best_chunk = sub_chunks[int(scores.argmax())][:120] if len(scores) > 0 else ""
                except Exception:
                    max_sim = 0.0

            is_semantic = max_sim >= 0.65

        is_hit = any_literal or is_semantic
        if is_hit:
            hits += 1
        if any_literal:
            literal_hits += 1
        if is_semantic:
            semantic_hits += 1

        per_question.append({
            "index": r["index"],
            "answer": str(answers[0])[:80],
            "literal_hit": any_literal,
            "matched": matched_answers[:3],
            "max_sim": round(max_sim, 4),
            "semantic_hit": is_semantic,
            "best_chunk": best_chunk,
            "is_hit": is_hit,
        })

    hit_rate = hits / total if total > 0 else 0.0
    return {
        "total": total,
        "hits": hits,
        "hit_rate": round(hit_rate, 4),
        "literal_hits": literal_hits,
        "semantic_hits": semantic_hits,
        "per_question": per_question,
    }


def print_summary(results: list[dict], title: str):
    """打印汇总"""
    scored = [r for r in results if r["score"] is not None]
    total_score = sum(r["score"] for r in scored)
    max_score = sum(r["total"] for r in scored)
    avg = total_score / len(scored) * 10 if scored else 0  # 归一化到 10 分制

    print(f"\n{'='*60}")
    print(f"🏆 {title}")
    print(f"{'='*60}")
    print(f"  总题数: {len(results)}")
    print(f"  有效评分: {len(scored)}/{len(results)}")
    print(f"  总分: {total_score}/{max_score}")
    print(f"  均分: {avg:.1f}/10" if scored else "  均分: N/A")

    # 按题型统计
    by_type: dict[str, list] = {}
    for r in results:
        t = r.get("ques_type", "?")
        by_type.setdefault(t, []).append(r)
    for t, rs in sorted(by_type.items()):
        s = [r["score"] for r in rs if r["score"] is not None]
        if s:
            print(f"  {t}: {sum(s)}/{sum(r['total'] for r in rs if r['score'] is not None)} "
                  f"(均{sum(s)/len(s):.1f})")

    # RAG 命中率
    if any(r.get("rag_enabled") for r in results):
        h = compute_hit_rate(results)
        if h["total"] > 0:
            print(f"\n  🎯 RAG 命中率: {h['hits']}/{h['total']} = {h['hit_rate']:.1%}")
            print(f"     字面命中: {h.get('literal_hits', 0)} | 语义命中: {h.get('semantic_hits', 0)}")
            if h["per_question"]:
                misses = [q for q in h["per_question"] if not q["is_hit"]]
                if misses:
                    m_list = ", ".join(f"Q{q['index']+1}" for q in misses)
                    print(f"  未命中 ({len(misses)} 题): {m_list}")


def print_compare_table(no_rag_results: list[dict], rag_results: list[dict]):
    """打印 RAG 对比"""
    print(f"\n{'='*70}")
    print("🏆 RAG 对比")
    print(f"{'='*70}")
    print(f"  {'题号':<6} {'题型':<8} {'无RAG':>8} {'有RAG':>8} {'差值':>8}")
    print(f"  {'-'*38}")

    diffs = []
    for nr, rr in zip(no_rag_results, rag_results):
        s_no = nr.get("score") or 0
        s_rag = rr.get("score") or 0
        diff = s_rag - s_no
        diffs.append(diff)
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"  #{nr['index']+1:<5} {nr['ques_type']:<8} {s_no:>8} {s_rag:>8} {diff_str:>8}")

    total_diff = sum(diffs)
    print(f"  {'-'*38}")
    print(f"  合计差值: {'+' + str(total_diff) if total_diff > 0 else total_diff}")


# ── 保存 ──────────────────────────────────────────────────

def save_results(results: list[dict], mode: str):
    """保存结果"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = os.path.join(OUTPUT_DIR, f"hard_math_{mode}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Markdown
    md_path = os.path.join(OUTPUT_DIR, f"hard_math_{mode}_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 初中数学较难题测评报告 ({mode})\n\n")
        f.write(f"**时间**: {ts}\n")
        f.write(f"**题目数**: {len(results)}\n\n")
        scored = [r for r in results if r["score"] is not None]
        if scored:
            total = sum(r["score"] for r in scored)
            max_s = sum(r["total"] for r in scored)
            f.write(f"**总分**: {total}/{max_s}  |  **均分**: {total/len(scored):.1f}/10\n\n")

        # 命中率
        if any(r.get("rag_enabled") for r in results):
            h = compute_hit_rate(results)
            if h["total"] > 0:
                f.write(f"**🎯 RAG命中率**: {h['hits']}/{h['total']} = {h['hit_rate']:.1%}")
                f.write(f" (字面{h.get('literal_hits',0)} + 语义{h.get('semantic_hits',0)})\n\n")

        f.write(f"---\n\n")

        for r in results:
            f.write(f"## Q{r['index']+1} — {r['ques_type']}\n\n")
            f.write(f"**题目**: {r['ques_content']}\n\n")
            f.write(f"**参考答案**: {'；'.join(r['ques_answer'])}\n\n")
            if r.get("rag_refs"):
                label = "联网参考" if r.get("rag_is_online") else "本地参考"
                f.write(f"**{label}**: {' | '.join(r['rag_refs'][:3])}\n\n")
            if r.get("rag_context_full"):
                f.write(f"<details>\n<summary>传给模型的实际内容</summary>\n\n```\n{r['rag_context_full']}\n```\n\n</details>\n\n")
            f.write(f"**教师答案**:\n\n{r['teacher_answer']}\n\n")
            f.write(f"**评分**: {r.get('score','?')}/{r.get('total','10')}\n\n")
            f.write(f"**评分理由**: {r.get('reason','')}\n\n")
            f.write(f"<details>\n<summary>裁判完整评分原文</summary>\n\n{r['grading_result']}\n\n</details>\n\n")
            f.write(f"---\n\n")

    print(f"\n💾 结果已保存:")
    print(f"  {json_path}")
    print(f"  {md_path}")


# ── 入口 ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="初中数学较难题批量测评")
    parser.add_argument("--rag", action="store_true", help="启用 RAG")
    parser.add_argument("--online-rag", default=None,
                        help="联网 RAG 模式: auto / online（需配置 Tavily key）")
    parser.add_argument("--compare", action="store_true", help="对比模式")
    parser.add_argument("--local", action="store_true", help="使用本地模型")
    parser.add_argument("--limit", type=int, default=0, help="限制题目数（调试用）")
    parser.add_argument("--verbose", action="store_true", help="打印完整 RAG 检索过程")
    args = parser.parse_args()

    # 解析 online_mode
    online_mode = args.online_rag
    use_rag = args.rag or online_mode is not None

    # 加载题目
    questions = load_questions(JSONL_PATH)
    if args.limit > 0:
        questions = questions[:args.limit]

    print(f"📚 加载 {len(questions)} 道初中数学较难题\n")

    # 检查 RAG 可用性
    if use_rag:
        local_ok = is_rag_available()
        online_ok = is_online_rag_available()
        if not local_ok and not online_ok:
            print("⚠️  本地 RAG 和联网 RAG 均不可用，将使用无 RAG 模式\n")
            use_rag = False
            online_mode = None
        elif not local_ok:
            print("⚠️  本地 RAG 不可用，将直接使用联网搜索\n")
            online_mode = "online" if online_mode else "auto"
        elif not online_ok:
            print("⚠️  联网 RAG 不可用，auto 模式将仅使用本地检索\n")

    # 初始化模型
    print("🔗 初始化模型...")
    if args.local:
        teacher = make_local_teacher_client()
        print("  ✅ 教师模型（本地 DeepSeek-R1-7B）已加载")
    else:
        teacher = make_teacher_client()
        print("  ✅ 教师模型已连接")
    judge = make_judge_client()
    print("  ✅ 裁判模型已连接\n")

    if args.compare:
        # ── 对比模式 ──
        print("=" * 60)
        print("  第一轮：无 RAG 模式")
        print("=" * 60)
        no_rag = run_batch(questions, teacher, judge, use_rag=False, desc="无RAG", verbose=args.verbose)
        print_summary(no_rag, "无 RAG")

        print(f"\n{'='*60}")
        print("  第二轮：挂载 RAG 模式")
        print("=" * 60)
        with_rag = run_batch(questions, teacher, judge, use_rag=True,
                             online_mode=online_mode or "auto", desc="有RAG", verbose=args.verbose)
        print_summary(with_rag, "有 RAG")
        print_compare_table(no_rag, with_rag)

        save_results(no_rag, "no_rag")
        save_results(with_rag, "rag")
    else:
        # ── 单模式 ──
        results = run_batch(questions, teacher, judge,
                            use_rag=use_rag, online_mode=online_mode,
                            desc="有RAG" if use_rag else "无RAG", verbose=args.verbose)
        print_summary(results, "有RAG" if use_rag else "无RAG")
        mode_str = "rag" if use_rag else "no_rag"
        save_results(results, mode_str)

    print(f"\n✅ 全部完成，结果保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
