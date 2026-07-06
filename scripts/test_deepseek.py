"""DeepSeek 连通性测试脚本。

用法（项目根目录）：python scripts/test_deepseek.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证从项目根目录以外也能 import app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.llm.deepseek_client import DeepSeekClient, DeepSeekError, is_configured  # noqa: E402


def main() -> int:
    if not is_configured():
        print("未检测到 DEEPSEEK_API_KEY。")
        print("请检查项目根目录的 .env 文件（可从 .env.example 复制），填写 DEEPSEEK_API_KEY。")
        return 1

    client = DeepSeekClient()
    print(f"model: {client.settings.deepseek_model}")
    print(f"base_url: {client.settings.deepseek_base_url}")
    try:
        reply = client.chat(
            system="你是一个连通性测试助手。",
            user="请用一句话说明 DeepSeek API 已连接成功",
            temperature=0.0,
        )
    except DeepSeekError as exc:
        print(f"连接失败：{exc}")
        return 1
    print(f"模型返回：{reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
