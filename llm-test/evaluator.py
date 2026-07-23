"""
测评主逻辑：考生模型答题 → 阅卷模型按标准评分
支持 RAG 对比模式：不挂载 RAG vs 挂载 RAG 教材检索
"""

import os
import re
import json
import sys
from datetime import datetime

from tqdm import tqdm

from config import PAPER_DIR, KEY_DIR, SUBJECT_MAP, GRADER_SYSTEM_PROMPT
from llm_client import make_student_client, make_grader_client, make_local_student_client
from rag_retriever import retrieve_for_exam_questionwise, is_rag_available

# 全卷测评专属输出目录（与 config.py 的 OUTPUT_DIR 区分）
PAPER_OUTPUT_DIR = "results/paper"


def _strip_boilerplate(text: str) -> str:
    """去掉试卷中的注意事项等锅炉板文字，减少 RAG 检索噪音"""
    # 去掉注意事项段：从"注意事项"开始到第一个非注意事项编号行结束
    # 注意事项通常是 1. 2. 3. 这样的编号列表，下一段是 # 或 ## 或非编号文字
    text = re.sub(
        r'注意事项[：:]\s*\n(?:\d+[\.、]\s*[^\n]*\n?)+',
        '', text,
    )
    # 去掉试卷头部标题行（"# 河南省 2026 年..."），减少跨试卷匹配
    text = re.sub(r'^#\s*.+?年.+?试卷.*?\n', '', text, count=1)
    # 去掉末尾无关文字（"祝考试顺利"等）
    text = re.sub(r'\n(?:祝考试顺利[!！]?|END)\s*$', '', text)
    return text


def find_paper_key_pairs() -> list[dict]:
    """扫描 paper 和 key 目录，匹配试卷-答案对"""
    pairs = []
    paper_dir = os.path.join(os.path.dirname(__file__), PAPER_DIR)
    key_dir = os.path.join(os.path.dirname(__file__), KEY_DIR)

    paper_files = sorted(os.listdir(paper_dir))
    key_files = sorted(os.listdir(key_dir))

    for pf in paper_files:
        if not pf.endswith(".md"):
            continue
        m = re.match(r"2026(.+?)试卷\.md$", pf)
        if not m:
            continue
        subject = m.group(1)
        kf = f"2026{subject}试卷答案.md"
        if kf in key_files:
            pairs.append({
                "subject": SUBJECT_MAP.get(subject, subject),
                "paper_path": os.path.join(paper_dir, pf),
                "key_path": os.path.join(key_dir, kf),
            })

    return pairs


def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_student_prompt(paper_content: str, rag_context: str = "") -> list[dict]:
    """构建考生模型的对话消息，可选注入 RAG 教材上下文"""
    if rag_context:
        system_content = (
            "你是一位参加2026年河南省中考的考生。请认真阅读以下试卷，"
            "按照试卷要求逐题作答。\n\n"
            "以下是从知识库中检索到的参考资料，可能包含与试卷直接相关或间接相关的内容。"
            "请自行判断每条资料的相关性：\n"
            "- 如果某条资料与试卷题目高度相似或完全一致（如同一张试卷的真题+答案），"
            "请直接参考其答案和解题过程作答。\n"
            "- 如果某条资料只是知识点相关或背景信息，可选择性参考。\n"
            "- 如果某条资料与题目无关，请忽略它。\n\n"
            f"{rag_context}\n\n"
            "---\n\n"
            "作答要求：\n"
            "1. 选择题请明确写出选项（如 A、B、C、D）\n"
            "2. 填空题请写出答案\n"
            "3. 解答题请写出完整的解题步骤\n"
            "4. 作文题请写完整的作文\n"
            "5. 请按题号顺序作答，格式为「题号. 答案」"
        )
    else:
        system_content = (
            "你是一位参加2026年河南省中考的考生。请认真阅读以下试卷，"
            "按照试卷要求逐题作答。\n\n"
            "作答要求：\n"
            "1. 选择题请明确写出选项（如 A、B、C、D）\n"
            "2. 填空题请写出答案\n"
            "3. 解答题请写出完整的解题步骤\n"
            "4. 作文题请写完整的作文\n"
            "5. 请按题号顺序作答，格式为「题号. 答案」"
        )

    system = {"role": "system", "content": system_content}
    user = {"role": "user", "content": f"请认真作答以下试卷：\n\n{paper_content}"}
    return [system, user]


def build_grader_prompt(student_answer: str, key_content: str) -> list[dict]:
    """构建阅卷模型的对话消息"""
    system = {
        "role": "system",
        "content": GRADER_SYSTEM_PROMPT,
    }
    user = {
        "role": "user",
        "content": f"""请严格批改以下试卷。

【参考答案及评分标准】
{key_content}

【考生答案】
{student_answer}

请按以下格式给出评分结果：

## 评分明细
（逐题列出：题号、满分、得分、扣分原因）

## 总分统计
| 科目 | 总分 | 得分 |
|------|------|------|

## 综合评价
（对考生整体表现的简短评语，50字以内）""",
    }
    return [system, user]


def extract_score(grading_result: str) -> str:
    """从评分结果中提取得分"""
    # 优先解析“总分统计”后的得分，避免把第一道小题的“得分2”当成总分。
    total_section = grading_result[grading_result.find("总分统计"):]
    total_match = re.search(r'得分\s*\|?\s*(\d+)', total_section)
    if not total_match:
        total_match = re.search(r'得分\s*\|?\s*(\d+)', grading_result)
    if not total_match:
        # 尝试另一种格式：得分: XX
        total_match = re.search(r'得分[：:]\s*(\d+)', grading_result)
    if not total_match:
        # 尝试表格格式第二行第二列
        lines = grading_result.split("\n")
        for line in lines:
            if "|" in line and re.search(r'\d+', line):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2 and parts[-1].isdigit():
                    return parts[-1]
    return total_match.group(1) if total_match else "?"


def evaluate_one(
    pair: dict,
    student_client,
    grader_client,
    save_raw: bool = True,
    use_rag: bool = False,
    online_mode: str | None = None,
    rag_profile: str = "standard",
    subject_pbar: tqdm = None,
    verbose: bool = False,
) -> dict:
    """评测一份试卷"""
    subject = pair["subject"]

    # 1. 读取试卷和答案
    paper = read_file(pair["paper_path"])
    key = read_file(pair["key_path"])

    # 2. RAG 检索（如果启用）
    rag_context = ""
    rag_refs: list[str] = []
    rag_source = "本地教材"  # 记录来源
    if use_rag:
        if subject_pbar:
            subject_pbar.set_description(f"🔍 {subject} RAG检索中")
        # 用剥离锅炉板后的内容做检索 query，避免"注意事项"等噪音吃掉检索名额
        paper_for_rag = _strip_boilerplate(paper)
        rag_context, rag_refs, used_online = retrieve_for_exam_questionwise(
            paper_for_rag, subject, online_mode=online_mode, verbose=verbose,
            profile=rag_profile,
            target_source_file=os.path.splitext(os.path.basename(pair["paper_path"]))[0],
            answer_content=key if rag_profile == "oracle" else "",
        )
        if rag_context:
            rag_source = "🌐 联网搜索" if used_online else "📚 本地教材"
            tqdm.write(f"  {rag_source} {subject} 检索到 {len(rag_refs)} 条参考")
        else:
            tqdm.write(f"  ⚠️ {subject} RAG 不可用，降级为无教材模式")

    # 3. 考生模型答题
    rag_label = "📚" if use_rag else ""
    if subject_pbar:
        subject_pbar.set_description(f"🧑 {subject} 考生答题中 {rag_label}")
    student_messages = build_student_prompt(paper, rag_context)
    student_answer = student_client.chat_with_retry(student_messages)

    # 4. 阅卷模型评分
    if subject_pbar:
        subject_pbar.set_description(f"👨‍🏫 {subject} 阅卷评分中 {rag_label}")
    grader_messages = build_grader_prompt(student_answer, key)
    grading_result = grader_client.chat_with_retry(grader_messages)

    result = {
        "subject": subject,
        "paper_file": os.path.basename(pair["paper_path"]),
        "key_file": os.path.basename(pair["key_path"]),
        "student_answer": student_answer,
        "grading_result": grading_result,
        "timestamp": datetime.now().isoformat(),
        "rag_enabled": use_rag,
        "rag_profile": rag_profile if use_rag else "none",
        "rag_refs": rag_refs,
        "rag_source": rag_source,
        "rag_context_full": rag_context,
        "rag_is_online": "🌐" in rag_source,
        "web_exact_paper_hit": bool(
            used_online and re.search(rf"2026.*{re.escape(subject)}|{re.escape(subject)}.*2026",
                                      "\n".join(rag_refs), re.IGNORECASE)
        ),
        "web_answer_hit": bool(
            used_online and re.search(r"参考答案|试题答案|答案解析|详解|标准答案",
                                      "\n".join(rag_refs) + "\n" + rag_context)
        ),
    }

    # 5. 保存结果
    if save_raw:
        if subject_pbar:
            subject_pbar.set_description(f"💾 {subject} 保存结果中 {rag_label}")
        save_result(result, subject, rag_mode=use_rag)

    # 6. 摘要
    score = extract_score(grading_result)
    rag_tag = " [RAG]" if use_rag else ""
    tqdm.write(f"  ✅ {subject}{rag_tag} 测评完成 得分: {score}")

    if subject_pbar:
        subject_pbar.update(1)

    return result


def save_result(result: dict, subject: str, rag_mode: bool = False):
    """保存评测结果到 results/paper/"""
    out_dir = os.path.join(os.path.dirname(__file__), PAPER_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_subject = subject.replace("/", "_")
    profile = result.get("rag_profile", "standard")
    mode_suffix = f"_{profile}" if rag_mode and profile != "standard" else ("_rag" if rag_mode else "")

    # 保存完整结果 JSON
    json_path = os.path.join(out_dir, f"{safe_subject}{mode_suffix}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存人类可读的 Markdown
    md_path = os.path.join(out_dir, f"{safe_subject}{mode_suffix}_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        rag_badge = "📚 挂载 RAG" if rag_mode else "📝 无 RAG"
        f.write(f"# {subject} 测评报告 {rag_badge}\n\n")
        f.write(f"**测评时间**: {result['timestamp']}\n\n")
        if result.get("rag_refs"):
            label = "🌐 联网参考" if result.get("rag_is_online") else "📚 本地参考"
            f.write(f"**{label}**:\n")
            for ref in result["rag_refs"]:
                f.write(f"- {ref}\n")
            f.write(f"\n")
        if result.get("rag_context_full"):
            f.write(f"<details>\n<summary>📋 传给模型的实际 RAG 上下文</summary>\n\n```\n{result['rag_context_full']}\n```\n\n</details>\n\n")
        f.write(f"---\n\n")
        f.write(f"## 一、考生答案\n\n{result['student_answer']}\n\n")
        f.write(f"---\n\n")
        f.write(f"## 二、阅卷评分\n\n{result['grading_result']}\n\n")
        f.write(f"<details>\n<summary>📋 阅卷模型完整评分原文</summary>\n\n{result['grading_result']}\n\n</details>\n")

    tqdm.write(f"  💾 结果已保存: {json_path}")
    tqdm.write(f"  💾 结果已保存: {md_path}")


# ============================================================
# 运行入口
# ============================================================


def _validate_pairs(pairs: list[dict]) -> bool:
    """预检查所有试卷和答案文件是否存在"""
    missing = []
    for p in pairs:
        if not os.path.exists(p["paper_path"]):
            missing.append(f"试卷: {p['paper_path']}")
        if not os.path.exists(p["key_path"]):
            missing.append(f"答案: {p['key_path']}")
    if missing:
        print("❌ 以下文件缺失，请检查:")
        for m in missing:
            print(f"  - {m}")
        return False
    return True


def _init_clients(local_student: bool = False):
    """初始化考生和阅卷模型客户端"""
    print("🔗 初始化模型连接...")
    if local_student:
        student = make_local_student_client()
        print("  ✅ 考生模型（本地 DeepSeek-R1-7B）已加载")
    else:
        student = make_student_client()
        print("  ✅ 考生模型已连接")
    grader = make_grader_client()
    print("  ✅ 阅卷模型已连接\n")
    return student, grader


def _run_batch(pairs: list[dict], student, grader, use_rag: bool,
               desc: str, online_mode: str | None = None, verbose: bool = False,
               rag_profile: str = "standard") -> list[dict]:
    """批量运行测评"""
    num = len(pairs)
    rag_label = " [RAG]" if use_rag else ""
    results = []
    with tqdm(total=num, desc=f"📋 {desc}{rag_label}", unit="科",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for pair in pairs:
            result = evaluate_one(pair, student, grader, use_rag=use_rag,
                                  online_mode=online_mode, subject_pbar=pbar,
                                  verbose=verbose, rag_profile=rag_profile)
            results.append(result)
    return results


def _print_summary(results: list[dict], title: str):
    """打印测评汇总"""
    print(f"\n{'='*60}")
    print(f"🏆 {title}")
    print(f"{'='*60}")
    for r in results:
        score = extract_score(r['grading_result'])
        rag_tag = " [RAG]" if r.get('rag_enabled') else ""
        print(f"  {r['subject']}{rag_tag}: 得分 {score}")


def _print_compare_table(no_rag_results: list[dict], rag_results: list[dict]):
    """打印 RAG 对比表"""
    print(f"\n{'='*70}")
    print("🏆 RAG 对比测评")
    print(f"{'='*70}")
    header = f"  {'科目':<12} {'无RAG':>8} {'有RAG':>8} {'差值':>8}"
    print(header)
    print(f"  {'-'*40}")

    for nr, rr in zip(no_rag_results, rag_results):
        score_no = extract_score(nr['grading_result'])
        score_rag = extract_score(rr['grading_result'])
        try:
            diff = int(score_rag) - int(score_no)
            diff_str = f"+{diff}" if diff > 0 else str(diff)
        except ValueError:
            diff_str = "?"
        print(f"  {nr['subject']:<12} {score_no:>8} {score_rag:>8} {diff_str:>8}")

    print(f"\n所有结果已保存到 {PAPER_OUTPUT_DIR}/ 目录")


def run_compare_all(local_student: bool = False, online_mode: str | None = None,
                    verbose: bool = False):
    """对比模式：不挂 RAG 和挂 RAG 各跑一遍，输出对比"""
    pairs = find_paper_key_pairs()
    if not pairs:
        print("❌ 未找到任何试卷-答案配对")
        return

    num = len(pairs)
    subjects = [p['subject'] for p in pairs]
    print(f"共找到 {num} 套试卷: {subjects}\n")

    if not _validate_pairs(pairs):
        return

    # 检查 RAG 可用性
    rag_ok = is_rag_available()
    if not rag_ok:
        print("⚠️  本地 RAG 不可用（缺少依赖或向量库不存在），将只跑无 RAG 模式\n")

    if online_mode and online_mode != "off":
        from online_rag import is_online_rag_available
        if not is_online_rag_available():
            print("⚠️  联网 RAG 不可用（未配置 Tavily API key），降级为 auto 模式\n")
        else:
            print(f"🌐 联网 RAG 模式: {online_mode}\n")

    student, grader = _init_clients(local_student=local_student)

    mode_tag = " [本地模型]" if local_student else ""

    # 第一轮：无 RAG
    print("=" * 60)
    print(f"  第一轮：无 RAG 模式{mode_tag}")
    print("=" * 60)
    no_rag_results = _run_batch(pairs, student, grader, use_rag=False, desc=f"无RAG{mode_tag}",
                                verbose=verbose)

    _print_summary(no_rag_results, "无 RAG 测评汇总")

    if not rag_ok:
        return

    # 第二轮：有 RAG
    online_tag = f" [联网:{online_mode}]" if online_mode and online_mode != "off" else ""
    print(f"\n{'='*60}")
    print(f"  第二轮：挂载 RAG 模式{mode_tag}{online_tag}")
    print("=" * 60)
    rag_results = _run_batch(pairs, student, grader, use_rag=True, desc=f"有RAG{mode_tag}",
                             online_mode=online_mode, verbose=verbose)

    _print_summary(rag_results, "有 RAG 测评汇总")

    # 对比表
    _print_compare_table(no_rag_results, rag_results)


def run_compare_one(subject: str, local_student: bool = False,
                    online_mode: str | None = None, verbose: bool = False):
    """单科对比：某一科不挂 RAG 和挂 RAG 各跑一遍，输出对比"""
    pairs = find_paper_key_pairs()
    target = None
    for p in pairs:
        if subject in p["subject"]:
            target = p
            break
    if not target:
        print(f"❌ 未找到科目: {subject}")
        print(f"可选科目: {[p['subject'] for p in pairs]}")
        return

    mode_tag = " [本地模型]" if local_student else ""
    online_tag = f" [联网:{online_mode}]" if online_mode and online_mode != "off" else ""
    print(f"📋 单科对比: {target['subject']}{mode_tag}{online_tag}\n")

    # 检查 RAG 可用性
    rag_ok = is_rag_available()
    if not rag_ok:
        print("⚠️  本地 RAG 不可用\n")

    if online_mode and online_mode != "off":
        from online_rag import is_online_rag_available
        if not is_online_rag_available():
            print("⚠️  联网 RAG 不可用（未配置 Tavily API key）\n")

    student, grader = _init_clients(local_student=local_student)

    # 第一轮：无 RAG
    print("=" * 60)
    print(f"  第一轮：无 RAG 模式{mode_tag}")
    print("=" * 60)
    no_rag_result = _run_batch([target], student, grader, use_rag=False, desc=f"无RAG{mode_tag}",
                               verbose=verbose)
    _print_summary(no_rag_result, "无 RAG 测评汇总")

    if not rag_ok:
        return

    # 第二轮：有 RAG
    print(f"\n{'='*60}")
    print(f"  第二轮：挂载 RAG 模式{mode_tag}{online_tag}")
    print("=" * 60)
    rag_result = _run_batch([target], student, grader, use_rag=True, desc=f"有RAG{mode_tag}",
                            online_mode=online_mode, verbose=verbose)
    _print_summary(rag_result, "有 RAG 测评汇总")

    # 对比表
    _print_compare_table(no_rag_result, rag_result)


def run_all(use_rag: bool = False, local_student: bool = False,
            online_mode: str | None = None, verbose: bool = False):
    """运行全部7科测评"""
    pairs = find_paper_key_pairs()
    if not pairs:
        print("❌ 未找到任何试卷-答案配对，请检查 testmd/paper 和 testmd/key 目录")
        return

    num = len(pairs)
    subjects = [p['subject'] for p in pairs]
    rag_label = " [挂载RAG]" if use_rag else ""
    online_tag = f" [联网:{online_mode}]" if online_mode and online_mode != "off" else ""
    print(f"共找到 {num} 套试卷: {subjects}{rag_label}{online_tag}")

    if not _validate_pairs(pairs):
        return

    if use_rag:
        if not is_rag_available():
            print("⚠️  本地 RAG 不可用，降级为无教材模式\n")
            use_rag = False
        else:
            print("📚 RAG 教材检索已启用\n")

    student, grader = _init_clients(local_student=local_student)

    desc = "有RAG" if use_rag else "无RAG"
    all_results = _run_batch(pairs, student, grader, use_rag=use_rag, desc=desc,
                             online_mode=online_mode, verbose=verbose)

    _print_summary(all_results, f"{desc} 测评汇总")
    print(f"\n所有结果已保存到 {PAPER_OUTPUT_DIR}/ 目录")


def run_one(subject: str, use_rag: bool = False, local_student: bool = False,
            online_mode: str | None = None, verbose: bool = False,
            rag_profile: str = "standard"):
    """运行单科测评，如 run_one('数学')"""
    pairs = find_paper_key_pairs()
    target = None
    for p in pairs:
        if subject in p["subject"]:
            target = p
            break
    if not target:
        print(f"❌ 未找到科目: {subject}")
        print(f"可选科目: {[p['subject'] for p in pairs]}")
        return

    rag_label = " [挂载RAG]" if use_rag else ""
    online_tag = f" [联网:{online_mode}]" if online_mode and online_mode != "off" else ""
    if use_rag and not is_rag_available():
        print("⚠️  本地 RAG 不可用，降级为无教材模式\n")
        use_rag = False

    if local_student:
        student = make_local_student_client()
    else:
        student = make_student_client()
    grader = make_grader_client()

    with tqdm(total=1, desc=f"📋 {target['subject']}{rag_label}{online_tag} 测评", unit="科",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        evaluate_one(target, student, grader, use_rag=use_rag,
                     online_mode=online_mode, subject_pbar=pbar, verbose=verbose,
                     rag_profile=rag_profile)


def print_usage():
    print("用法:")
    print("  python evaluator.py                         无 RAG，全部科目")
    print("  python evaluator.py 数学                     无 RAG，单科")
    print("  python evaluator.py --rag                   挂载 RAG，全部科目")
    print("  python evaluator.py 数学 --rag               挂载 RAG，单科")
    print("  python evaluator.py --compare               对比模式（无RAG vs 有RAG）")
    print("  python evaluator.py --local                 本地模型答题，全部科目")
    print("  python evaluator.py 数学 --local --rag       本地模型 + RAG，单科")
    print("  python evaluator.py --compare --local       对比模式，本地模型答题")
    print("")
    print("  联网 RAG（需在 config.py 配置 Tavily API key）:")
    print("  python evaluator.py 历史 --rag --online-rag auto    阈值触发联网")
    print("  python evaluator.py 历史 --rag --online-rag online  强制全程联网")
    print("  python evaluator.py --compare --online-rag auto     对比模式 + 联网回退")
    print("  python evaluator.py 物理 --local --rag --rag-profile local_clean")
    print("  python evaluator.py 物理 --local --rag --rag-profile oracle")
    print("  python evaluator.py 物理 --local --rag --rag-profile web  本地不足时不受限联网")
    print("")
    print("  详细 RAG 检索过程:")
    print("  python evaluator.py 数学 --rag --verbose             打印完整 RAG 检索过程")
    print("  python evaluator.py --compare --verbose              对比模式 + 详细检索日志")
    print("")
    print("  默认考生模型:  deepseek-chat (API)")
    print("  --local 后:    使用 LOCAL_LLM_BASE_URL 指定的本地 OpenAI 兼容模型")
    print("  阅卷模型始终:  deepseek-v4-flash (API，保证评分一致性)")


if __name__ == "__main__":
    args = sys.argv[1:]

    # 提取标志位
    use_rag = "--rag" in args
    use_local = "--local" in args
    use_compare = "--compare" in args
    use_verbose = "--verbose" in args

    rag_profile = "standard"
    for i, a in enumerate(args):
        if a == "--rag-profile" and i + 1 < len(args):
            rag_profile = args[i + 1]
            break
    if rag_profile not in {"standard", "local_clean", "oracle", "web"}:
        raise SystemExit(f"不支持的 --rag-profile: {rag_profile}")
    if rag_profile != "standard":
        use_rag = True

    # 提取 --online-rag 的值
    online_mode = None
    for i, a in enumerate(args):
        if a == "--online-rag":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                online_mode = args[i + 1]
            else:
                online_mode = "auto"  # 默认值
            break

    # 提取科目名（去掉标志位和其后面的值）
    skip_next = False
    subjects = []
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("--"):
            # 如果这个标志有后续值参数
            if a in {"--online-rag", "--rag-profile"} and len(args) > args.index(a) + 1 and not args[args.index(a) + 1].startswith("--"):
                skip_next = True
            continue
        subjects.append(a)

    # 仅联网 RAG，不带 --rag 时自动启用 RAG
    if online_mode and not use_rag:
        use_rag = True
    if rag_profile == "web":
        online_mode = "auto"

    if not args:
        run_all(use_rag=False)
    elif "--help" in args or "-h" in args:
        print_usage()
    elif use_compare:
        if subjects:
            run_compare_one(subjects[0], local_student=use_local, online_mode=online_mode,
                            verbose=use_verbose)
        else:
            run_compare_all(local_student=use_local, online_mode=online_mode,
                            verbose=use_verbose)
    elif use_rag:
        if subjects:
            run_one(subjects[0], use_rag=True, local_student=use_local, online_mode=online_mode,
                    verbose=use_verbose, rag_profile=rag_profile)
        else:
            run_all(use_rag=True, local_student=use_local, online_mode=online_mode,
                    verbose=use_verbose)
    elif use_local:
        if subjects:
            run_one(subjects[0], use_rag=False, local_student=True, verbose=use_verbose)
        else:
            run_all(use_rag=False, local_student=True, verbose=use_verbose)
    else:
        run_one(args[0], use_rag=False, verbose=use_verbose)
