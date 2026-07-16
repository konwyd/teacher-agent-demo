"""
LLM 客户端封装（OpenAI 兼容接口 + 本地 llama-cpp 模型）
"""

import json
import time
import re
from openai import OpenAI


class LLMClient:
    """通用的 LLM API 客户端"""

    def __init__(self, base_url: str, api_key: str, model_name: str,
                 temperature: float = 0.0, max_tokens: int = 8192):
        if not base_url or not api_key or not model_name:
            raise ValueError("请先在 config.py 中填写 API 配置信息")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: list, **kwargs) -> str:
        """发送对话请求，返回模型回复文本"""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content
        reasoning = getattr(choice.message, "reasoning_content", None)
        finish = choice.finish_reason

        # 推理模型可能把所有 token 用于思考，content 为空
        if not content and reasoning:
            print(f"  ⚠️ 模型推理耗尽 {choice.message.reasoning_content and len(reasoning)} tokens，"
                  f"finish_reason={finish}，回退使用推理内容末尾")
            # 取推理内容的最后部分作为答案（通常末尾包含最终回答）
            content = reasoning[-4000:] if len(reasoning) > 4000 else reasoning
        elif not content:
            print(f"  ⚠️ 模型返回空内容，finish_reason={finish}")
            content = ""

        if finish == "length":
            print(f"  ⚠️ 输出被截断（max_tokens={max_tokens}），建议增大 max_tokens")
        elif finish not in ("stop", None):
            print(f"  ⚠️ 异常结束: finish_reason={finish}")

        return content

    def chat_with_retry(self, messages: list, retries: int = 3, **kwargs) -> str:
        """带重试的对话请求"""
        for i in range(retries):
            try:
                return self.chat(messages, **kwargs)
            except Exception as e:
                if i == retries - 1:
                    raise
                wait = 2 ** i
                print(f"  [重试] 请求失败: {e}，{wait}秒后重试...")
                time.sleep(wait)
        return ""


def make_teacher_client():
    """根据配置创建教师模型客户端"""
    from config import TEACHER_MODEL
    return LLMClient(
        base_url=TEACHER_MODEL["base_url"],
        api_key=TEACHER_MODEL["api_key"],
        model_name=TEACHER_MODEL["model_name"],
        temperature=TEACHER_MODEL["temperature"],
        max_tokens=TEACHER_MODEL["max_tokens"],
    )


def make_judge_client():
    """根据配置创建裁判模型客户端"""
    from config import JUDGE_MODEL
    return LLMClient(
        base_url=JUDGE_MODEL["base_url"],
        api_key=JUDGE_MODEL["api_key"],
        model_name=JUDGE_MODEL["model_name"],
        temperature=JUDGE_MODEL["temperature"],
        max_tokens=JUDGE_MODEL["max_tokens"],
    )


# ============================================================
# 本地 llama-cpp 模型适配器
# ============================================================

class LocalLLMClient:
    """
    封装 llama-cpp-python 模型，暴露出与 LLMClient 一致的接口。
    用法：和 LLMClient 一样调用 .chat_with_retry(messages)
    """

    def __init__(self, model_path: str, n_ctx: int = 16384, n_threads: int = 32):
        from llama_cpp import Llama
        print(f"🔄 加载本地模型: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False,
        )
        self._model_path = model_path
        self._n_ctx = n_ctx
        self.model_name = "local-deepseek-r1-7b"
        self.temperature = 0.3
        self.max_tokens = 4096
        print(f"✅ 本地模型加载完成 (n_ctx={n_ctx})")

    def _strip_thinking(self, text: str) -> str:
        """移除 <think>...</think> 思考过程，只保留最终回答"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def chat(self, messages: list, **kwargs) -> str:
        """发送对话请求，返回模型回复文本"""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        resp = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        content = resp["choices"][0]["message"]["content"] or ""

        # 去掉思考过程
        content = self._strip_thinking(content)

        if not content:
            print(f"  ⚠️ 本地模型返回空内容")
            content = ""

        return content

    def chat_with_retry(self, messages: list, retries: int = 2, **kwargs) -> str:
        """带重试的对话请求"""
        for i in range(retries):
            try:
                return self.chat(messages, **kwargs)
            except Exception as e:
                if i == retries - 1:
                    raise
                wait = 2 ** i
                print(f"  [重试] 本地模型生成失败: {e}，{wait}秒后重试...")
                time.sleep(wait)
        return ""


def make_local_teacher_client() -> LocalLLMClient:
    """创建本地模型教师客户端"""
    from config import LOCAL_MODEL
    return LocalLLMClient(
        model_path=LOCAL_MODEL["model_path"],
        n_ctx=LOCAL_MODEL.get("n_ctx", 16384),
        n_threads=LOCAL_MODEL.get("n_threads", 32),
    )
