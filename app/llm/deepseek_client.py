"""DeepSeek client wrapper (OpenAI-compatible API).

All LLM calls in this project must go through this module.
API key is read from environment / .env — never hard-coded, never printed.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Iterator, Type, TypeVar
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 在模块导入时把 .env 写入环境变量（已存在的变量不会被覆盖）
load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIMEOUT_SECONDS = 300.0
_LLAMA_SLOT_ID: ContextVar[int | None] = ContextVar("llama_slot_id", default=None)
_JSON_CODE_FENCE = re.compile(
    r"\A\s*```(?:json)?\s*(?P<body>\{.*\})\s*```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def _unwrap_json_code_fence(raw: str) -> str:
    """Accept one whole-response JSON fence without guessing through prose."""

    match = _JSON_CODE_FENCE.fullmatch(raw)
    if match is None:
        return raw
    return match.group("body")


@contextmanager
def llama_slot(slot_id: int | None) -> Iterator[None]:
    """Temporarily route OpenAI-compatible requests to one llama-server slot."""

    if slot_id is not None and slot_id < 0:
        raise ValueError("slot_id must be zero or greater")
    token = _LLAMA_SLOT_ID.set(slot_id)
    try:
        yield
    finally:
        _LLAMA_SLOT_ID.reset(token)


def _request_slot_extra_body() -> dict[str, object] | None:
    slot_id = _LLAMA_SLOT_ID.get()
    if slot_id is None:
        return None
    return {"id_slot": slot_id, "cache_prompt": True}


class DeepSeekError(RuntimeError):
    """DeepSeek API 调用失败（配置、网络或服务端错误）。"""


class DeepSeekSettings(BaseSettings):
    """Loaded from environment variables or .env file."""

    teacher_llm_backend: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    deepseek_context_window_tokens: int = 131072

    teacher_local_llm_api_key: str = "local"
    teacher_local_llm_base_url: str = "http://127.0.0.1:8080/v1"
    teacher_local_llm_model: str = "deepseek-v4-flash"
    teacher_local_llm_context_window_tokens: int = 524288

    teacher_api_llm_api_key: str = ""
    teacher_api_llm_base_url: str = "https://api.deepseek.com"
    teacher_api_llm_model: str = "deepseek-chat"
    teacher_api_llm_context_window_tokens: int = 131072

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def select_backend_profile(self) -> "DeepSeekSettings":
        """Select one endpoint profile without changing any call-site behavior."""

        backend = self.teacher_llm_backend.strip().lower()
        if backend not in {"", "local", "api"}:
            raise ValueError(
                "TEACHER_LLM_BACKEND 只能是 local 或 api（留空沿用 DEEPSEEK_*）"
            )
        self.teacher_llm_backend = backend
        if backend == "local":
            self.deepseek_api_key = self.teacher_local_llm_api_key
            self.deepseek_base_url = self.teacher_local_llm_base_url
            self.deepseek_model = self.teacher_local_llm_model
            self.deepseek_context_window_tokens = (
                self.teacher_local_llm_context_window_tokens
            )
        elif backend == "api":
            self.deepseek_api_key = self.teacher_api_llm_api_key
            self.deepseek_base_url = self.teacher_api_llm_base_url
            self.deepseek_model = self.teacher_api_llm_model
            self.deepseek_context_window_tokens = (
                self.teacher_api_llm_context_window_tokens
            )
        return self


@lru_cache
def get_settings() -> DeepSeekSettings:
    return DeepSeekSettings()


def is_configured() -> bool:
    """Whether the selected backend has a key (local uses a placeholder key).

    TEACHER_FORCE_MOCK=1 时强制返回 False（评测 mock 模式的双保险）。
    """
    if os.getenv("TEACHER_FORCE_MOCK", "").lower() in ("1", "true"):
        return False
    backend = os.getenv("TEACHER_LLM_BACKEND", "").strip().lower()
    if backend == "local":
        return bool(os.getenv("TEACHER_LOCAL_LLM_API_KEY", "local"))
    if backend == "api":
        return bool(os.getenv("TEACHER_API_LLM_API_KEY", ""))
    if backend:
        DeepSeekSettings()  # Raise the same configuration error as the client.
    return bool(os.getenv("DEEPSEEK_API_KEY"))


class DeepSeekClient:
    """Thin wrapper over the OpenAI-compatible DeepSeek API.

    - ``chat``       : free-form text (explanations, hints)
    - ``chat_json``  : JSON mode + Pydantic validation (routing, grading, state updates)
    """

    def __init__(
        self,
        settings: DeepSeekSettings | None = None,
        timeout: float | None = None,
        max_retries: int = 1,
    ) -> None:
        self.settings = settings or DeepSeekSettings()
        self.last_chat_metadata: dict[str, object] = {}
        self.usage_history: list[dict[str, object]] = []
        if not self.settings.deepseek_api_key:
            raise DeepSeekError(
                "DEEPSEEK_API_KEY 未设置。请复制 .env.example 为 .env 并填写 key。"
            )
        self._client = OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            timeout=(
                self.settings.deepseek_timeout_seconds if timeout is None else timeout
            ),
            max_retries=max_retries,
        )

    def _request_extra_body(self) -> dict[str, object] | None:
        """Combine local llama.cpp controls with the current KV slot lease."""

        extra_body: dict[str, object] = {}
        host = urlparse(self.settings.deepseek_base_url).hostname
        if host in {"127.0.0.1", "localhost", "::1"}:
            extra_body.update(
                {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "reasoning_budget_tokens": 0,
                }
            )
        slot_extra_body = _request_slot_extra_body()
        if slot_extra_body is not None:
            extra_body.update(slot_extra_body)
        return extra_body or None

    def _record_response_metadata(
        self,
        resp: object,
        *,
        reasoning: str = "",
    ) -> dict[str, object]:
        """Keep safe usage counters while deliberately dropping raw reasoning."""

        choices = getattr(resp, "choices", []) or []
        choice = choices[0] if choices else None
        usage = getattr(resp, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        reasoning_tokens = int(
            getattr(completion_details, "reasoning_tokens", 0) or 0
        )
        metadata: dict[str, object] = {
            "finish_reason": getattr(choice, "finish_reason", "") or "",
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "reasoning_characters": len(reasoning),
            "reasoning_available": bool(reasoning),
        }

        prompt_tokens_raw = getattr(usage, "prompt_tokens", None)
        if prompt_tokens_raw is not None:
            prompt_tokens = int(prompt_tokens_raw or 0)
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens_raw = getattr(usage, "prompt_cache_hit_tokens", None)
            if cached_tokens_raw is None:
                cached_tokens_raw = getattr(prompt_details, "cached_tokens", None)
            cache_miss_raw = getattr(usage, "prompt_cache_miss_tokens", None)
            cache_write_raw = getattr(usage, "prompt_cache_write_tokens", None)
            if cache_write_raw is None:
                cache_write_raw = getattr(usage, "cache_creation_input_tokens", None)
            cached_tokens = int(cached_tokens_raw or 0)
            cache_miss_tokens = int(
                cache_miss_raw
                if cache_miss_raw is not None
                else max(prompt_tokens - cached_tokens, 0)
            )
            total_tokens = int(
                getattr(usage, "total_tokens", None)
                or prompt_tokens + completion_tokens
            )
            cache_denominator = cached_tokens + cache_miss_tokens
            metadata.update(
                {
                    "prompt_tokens": prompt_tokens,
                    "prompt_cache_hit_tokens": cached_tokens,
                    "prompt_cache_miss_tokens": cache_miss_tokens,
                    "cache_write_tokens": (
                        int(cache_write_raw or 0)
                        if cache_write_raw is not None
                        else None
                    ),
                    "cache_hit_rate": (
                        cached_tokens / cache_denominator
                        if cache_denominator > 0
                        else 0.0
                    ),
                    "cache_usage_available": cached_tokens_raw is not None
                    or cache_miss_raw is not None,
                    "response_tokens": max(completion_tokens - reasoning_tokens, 0),
                    "total_tokens": total_tokens,
                    "context_used_tokens": total_tokens,
                    "context_window_tokens": self.settings.deepseek_context_window_tokens,
                }
            )

        self.last_chat_metadata = metadata
        self.usage_history.append(dict(metadata))
        return metadata

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
            extra_body = self._request_extra_body()
            if extra_body is not None:
                request_args["extra_body"] = extra_body
            resp = self._client.chat.completions.create(
                **request_args,
            )
        except Exception as exc:  # noqa: BLE001 - 统一转成带上下文的领域异常
            raise DeepSeekError(
                f"DeepSeek API 调用失败（model={self.settings.deepseek_model}, "
                f"base_url={self.settings.deepseek_base_url}）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        choice = resp.choices[0]
        message = choice.message
        reasoning = getattr(message, "reasoning_content", None) or ""
        self._record_response_metadata(resp, reasoning=reasoning)
        return message.content or ""

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
            extra_body = self._request_extra_body()
            if extra_body is not None:
                request_args["extra_body"] = extra_body
            resp = self._client.chat.completions.create(
                **request_args,
            )
        except Exception as exc:  # noqa: BLE001
            raise DeepSeekError(
                f"DeepSeek API 调用失败（JSON mode, model={self.settings.deepseek_model}）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        message = resp.choices[0].message
        raw = _unwrap_json_code_fence(message.content or "{}")
        reasoning = getattr(message, "reasoning_content", None) or ""
        self._record_response_metadata(resp, reasoning=reasoning)
        return schema.model_validate_json(raw)


def summarize_usage_metadata(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate all LLM calls in a turn while keeping peak context occupancy."""

    if not records:
        return {}

    latest = records[-1]
    sum_fields = (
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "reasoning_characters",
        "response_tokens",
        "total_tokens",
    )
    summary: dict[str, object] = {
        field: sum(int(record.get(field, 0) or 0) for record in records)
        for field in sum_fields
    }
    cache_writes = [
        int(record["cache_write_tokens"] or 0)
        for record in records
        if record.get("cache_write_tokens") is not None
    ]
    cache_hits = int(summary["prompt_cache_hit_tokens"])
    cache_misses = int(summary["prompt_cache_miss_tokens"])
    cache_denominator = cache_hits + cache_misses
    peak = max(
        records,
        key=lambda record: int(record.get("context_used_tokens", 0) or 0),
    )
    context_used = int(peak.get("context_used_tokens", 0) or 0)
    context_window = int(peak.get("context_window_tokens", 0) or 0)
    summary.update(
        {
            "llm_call_count": len(records),
            "finish_reason": latest.get("finish_reason", "") or "",
            "reasoning_available": any(
                bool(record.get("reasoning_available", False)) for record in records
            ),
            "cache_write_tokens": sum(cache_writes) if cache_writes else None,
            "cache_usage_available": any(
                bool(record.get("cache_usage_available", False)) for record in records
            ),
            "cache_hit_rate": (
                cache_hits / cache_denominator if cache_denominator > 0 else 0.0
            ),
            "context_used_tokens": context_used,
            "context_window_tokens": context_window,
            "context_usage_percent": (
                context_used / context_window * 100 if context_window > 0 else 0.0
            ),
        }
    )
    return summary
