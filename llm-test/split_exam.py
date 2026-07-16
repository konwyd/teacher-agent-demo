"""
将每套试卷拆分为"题目"和"答案"两个独立的 markdown 文件
"""

import os
import re

INPUT_DIR = "/data/education/llm-test/markdown"
OUTPUT_DIR = "/data/education/llm-test"

# 参考答案的匹配模式 - 处理字符间可能存在的空格
ANSWER_PATTERNS = [
    re.compile(r'^# .*参\s*考\s*答\s*案'),
    re.compile(r'^# .*试\s*题\s*答\s*案'),
    re.compile(r'^# .*考\s*试\s*答\s*案'),
]

SEPARATOR = "\n\n---\n\n" + "=" * 60 + "\n" + "=" * 60 + "\n\n"


def find_answer_line(lines):
    """找到答案部分的起始行号"""
    for i, line in enumerate(lines):
        for pat in ANSWER_PATTERNS:
            if pat.match(line.strip()):
                return i
    return None


def process_file(filepath):
    basename = os.path.basename(filepath)
    print(f"处理: {basename}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines()

    answer_start = find_answer_line(lines)

    if answer_start is None:
        print(f"  [警告] 未找到答案分界线，跳过")
        return

    # 题目部分: 从开头到答案分界线（不包含答案标题）
    question_lines = lines[:answer_start]

    # 答案部分: 从答案分界线开始到结束
    answer_lines = lines[answer_start:]

    # 生成文件名（去掉 MinerU_markdown_ 前缀）
    name = basename
    if name.startswith("MinerU_markdown_"):
        name = name[len("MinerU_markdown_"):]
    name = name.replace(".md", "")

    # 提取科目名
    subject_map = {
        "化学": "化学",
        "物理": "物理",
        "英语": "英语",
        "道法": "道德与法治",
        "历史": "历史",
        "数学": "数学",
        "语文": "语文",
    }

    subject = None
    for key, val in subject_map.items():
        if key in basename:
            subject = val
            break
    if subject is None:
        subject = name

    question_file = os.path.join(OUTPUT_DIR, f"{name}_题目.md")
    answer_file = os.path.join(OUTPUT_DIR, f"{name}_答案.md")

    # 找到题目和答案之间的分界线
    # 找到第一个 "# " 标题行（即答案标题）
    question_text = "\n".join(question_lines).strip()
    
    # 答案文件中，在第一个标题前添加明确的分界线
    answer_text = "\n".join(answer_lines).strip()

    # 组装题目文件
    question_output = question_text
    question_output += f"\n\n{SEPARATOR}"
    question_output += f"> **📝 以上为《{subject}》试题部分，以下为答案部分的分界线，答案请参见独立答案文件。**"
    question_output += f"\n{SEPARATOR}"

    with open(question_file, 'w', encoding='utf-8') as f:
        f.write(question_output)

    # 组装答案文件
    # 在答案文件开头添加说明和分界线
    answer_header = f"{SEPARATOR}"
    answer_header += f"> **📋 以下为《{subject}》参考答案部分。**"
    answer_header += f"\n{SEPARATOR}\n\n"
    answer_output = answer_header + answer_text

    with open(answer_file, 'w', encoding='utf-8') as f:
        f.write(answer_output)

    print(f"  -> {os.path.basename(question_file)} ({len(question_lines)} 行)")
    print(f"  -> {os.path.basename(answer_file)} ({len(answer_lines)} 行)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted([
        f for f in os.listdir(INPUT_DIR) if f.endswith('.md')
    ])

    for f in files:
        filepath = os.path.join(INPUT_DIR, f)
        process_file(filepath)

    print("\n全部处理完成！")


if __name__ == "__main__":
    main()
