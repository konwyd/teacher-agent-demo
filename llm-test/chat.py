#!/home/lizhaoyuan/miniconda3/envs/rag/bin/python3
"""
DeepSeek-R1-Distill-Qwen-7B 本地对话脚本
使用 llama-cpp-python 加载 GGUF 模型，交互式多轮对话
"""

import re
import sys
from llama_cpp import Llama

MODEL_PATH = "/data/education/llm-test/model/DeepSeek-R1-Distill-Qwen-7B-Q4_K_L.gguf"

SYSTEM_PROMPT = "你是一个有帮助的AI助手，请用中文回答问题。"


def load_model():
    """加载模型"""
    print(f"🔄 正在加载模型: {MODEL_PATH}")
    print("   Q4_K_L 量化模型，加载预计 10-15 秒...")

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=16384,        # 上下文窗口（输入+输出），Q4量化后16K可稳定运行
        n_threads=32,
        n_batch=512,
        verbose=False,
    )
    print("✅ 模型加载完成！输入 'quit' / 'exit' 退出对话。\n")
    return llm


def format_output(text: str, show_thinking: bool = False):
    """
    格式化输出：将 <think>..</think> 部分与最终回答分开显示
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = ""
    answer = text

    if think_match:
        thinking = think_match.group(1).strip()
        answer = text[think_match.end():].strip()

    if thinking and show_thinking:
        print(f"\033[90m💭 思考过程:\n{thinking}\033[0m")
        print(f"\n\033[0m{answer}")
    elif thinking:
        print(f"\033[90m💭 (已隐藏思考过程，用 /think 切换显示)\033[0m")
        print(f"\n{answer}")
    else:
        print(answer)


def chat_loop(llm: Llama):
    """交互式对话循环"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    show_thinking = False

    print("命令提示： /think  切换是否显示思考过程")
    print("          /clear  清空对话历史")
    print("          quit    退出\n")

    while True:
        try:
            user_input = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        elif user_input == "/think":
            show_thinking = not show_thinking
            print(f"   思考过程显示: {'开启' if show_thinking else '隐藏'}")
            continue
        elif user_input == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("   ✅ 对话历史已清空")
            continue

        messages.append({"role": "user", "content": user_input})

        print(f"\n🤖 助手: ", end="", flush=True)

        full_response = ""
        try:
            stream = llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=8192,     # 单次回复最大 tokens（输入长时自动压缩输出）
                stream=True,
            )

            for chunk in stream:
                choices = chunk.get("choices", [])
                if choices and len(choices) > 0:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        full_response += content

            print("\n")
            messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"\n❌ 生成出错: {e}")
            messages.pop()


def main():
    llm = load_model()
    chat_loop(llm)


if __name__ == "__main__":
    main()
