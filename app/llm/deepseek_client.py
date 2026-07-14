"""DeepSeek client wrapper (OpenAI-compatible API).

All LLM calls in this project must go through this module.
API key is read from environment / .env — never hard-coded, never printed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# 在模块导入时把 .env 写入环境变量（已存在的变量不会被覆盖）
load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIMEOUT_SECONDS = 30.0


class DeepSeekError(RuntimeError):
    """DeepSeek API 调用失败（配置、网络或服务端错误）。"""


class DeepSeekSettings(BaseSettings):
    """Loaded from environment variables or .env file."""

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> DeepSeekSettings:
    return DeepSeekSettings()


def is_configured() -> bool:
    """是否配置了 DEEPSEEK_API_KEY（load_dotenv 已在模块导入时执行）。

    TEACHER_FORCE_MOCK=1 时强制返回 False（评测 mock 模式的双保险）。
    """
    if os.getenv("TEACHER_FORCE_MOCK", "").lower() in ("1", "true"):
        return False
    return bool(os.getenv("DEEPSEEK_API_KEY"))


class DeepSeekClient:
    """Thin wrapper over the OpenAI-compatible DeepSeek API.

    - ``chat``       : free-form text (explanations, hints)
    - ``chat_json``  : JSON mode + Pydantic validation (routing, grading, state updates)
    """

    def __init__(
        self,
        settings: DeepSeekSettings | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 1,
    ) -> None:
        self.settings = settings or DeepSeekSettings()
        if not self.settings.deepseek_api_key:
            raise DeepSeekError(
                "DEEPSEEK_API_KEY 未设置。请复制 .env.example 为 .env 并填写 key。"
            )
        self._client = OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        try:
            request_args = {
                "model": self.settings.deepseek_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            }
            if max_tokens is not None:
                request_args["max_tokens"] = max_tokens
            resp = self._client.chat.completions.create(
                **request_args,
            )
        except Exception as exc:  # noqa: BLE001 - 统一转成带上下文的领域异常
            raise DeepSeekError(
                f"DeepSeek API 调用失败（model={self.settings.deepseek_model}, "
                f"base_url={self.settings.deepseek_base_url}）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return resp.choices[0].message.content or ""

    def chat_json(
        self,
        system: str,
        user: str,
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> T:
        """Call DeepSeek in JSON mode and validate against a Pydantic model."""
        system_with_schema = (
            f"{system}\n\n"
            f"你必须只输出一个 JSON 对象，符合以下 JSON Schema：\n"
            f"{schema.model_json_schema()}"
        )
        try:
            request_args = {
                "model": self.settings.deepseek_model,
                "messages": [
                    {"role": "system", "content": system_with_schema},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            if max_tokens is not None:
                request_args["max_tokens"] = max_tokens
            resp = self._client.chat.completions.create(
                **request_args,
            )
        except Exception as exc:  # noqa: BLE001
            raise DeepSeekError(
                f"DeepSeek API 调用失败（JSON mode, model={self.settings.deepseek_model}）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raw = resp.choices[0].message.content or "{}"
        return schema.model_validate_json(raw)
