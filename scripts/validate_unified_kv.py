"""Live four-slot isolation and prompt-cache validation for local llama-server."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from app.llm.deepseek_client import DeepSeekClient, DeepSeekSettings, llama_slot


MARKERS = (
    "甲槽验证通过7319",
    "乙槽验证通过2846",
    "丙槽验证通过9052",
    "丁槽验证通过4173",
)
SIGNATURES = ("7319", "2846", "9052", "4173")


def _request(
    slot_id: int,
    marker: str,
    base_url: str,
    padding_repetitions: int,
) -> dict[str, object]:
    settings = DeepSeekSettings(
        deepseek_api_key="local-unified-validation",
        deepseek_base_url=base_url,
        deepseek_model="deepseek-v4-flash",
        deepseek_timeout_seconds=300,
        deepseek_context_window_tokens=524_288,
    )
    client = DeepSeekClient(settings=settings, max_retries=0)
    padding = "九年级数学学习记录。" * padding_repetitions
    user_message = f"以下是隔离填充，不需要复述：\n{padding}\n只输出标记：{marker}"
    with llama_slot(slot_id):
        response = client.chat(
            "你是缓存隔离测试器。只原样输出用户给出的测试标记，不要添加其他文字。",
            user_message,
            temperature=0.0,
            max_tokens=64,
        )
    return {
        "slot": slot_id,
        "marker": marker,
        "response": response,
        "metadata": client.last_chat_metadata,
    }


def validate(
    base_url: str,
    rounds: int,
    padding_repetitions: int = 0,
) -> list[dict[str, object]]:
    all_results: list[dict[str, object]] = []
    for round_id in range(1, rounds + 1):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda item: _request(
                        item[0], item[1], base_url, padding_repetitions
                    ),
                    enumerate(MARKERS),
                )
            )
        for result in results:
            response = str(result["response"])
            marker = str(result["marker"])
            slot = int(result["slot"])
            signature = SIGNATURES[slot]
            foreign = [
                value for index, value in enumerate(SIGNATURES)
                if index != slot and value in response
            ]
            if signature not in response:
                raise RuntimeError(
                    f"slot {slot} did not return its own signature {signature}: {response!r}"
                )
            if foreign:
                raise RuntimeError(
                    f"slot {result['slot']} leaked foreign markers: {foreign}"
                )
            result["round"] = round_id
            all_results.append(result)
    return all_results


def main() -> int:
    # Local validation must not accidentally route loopback traffic through a
    # corporate HTTP proxy inherited from the shell.
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--padding-repetitions",
        type=int,
        default=0,
        help="repeat a neutral Chinese sentence to cross compressed-cache block boundaries",
    )
    args = parser.parse_args()

    results = validate(
        args.base_url,
        max(1, args.rounds),
        max(0, args.padding_repetitions),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
