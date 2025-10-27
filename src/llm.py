"""LLM interaction helpers for generation and self-judging."""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import requests

from .prompts import MC_SYSTEM, MC_USER_FMT

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120


def _chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str,
    endpoint: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Invalid LLM response: {data}") from exc
    return content.strip()


def generate_reply(
    system_prompt: str,
    user_prompt: str,
    model: str,
    endpoint: str,
    max_new_tokens: int = 220,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Generate a counselor reply using the configured LLM endpoint."""

    LOGGER.debug("Generating reply via %s", endpoint)
    return _chat_completion(
        system_prompt,
        user_prompt,
        model,
        endpoint,
        max_new_tokens,
        temperature,
        top_p,
    )


def refine_with_judge(
    client_last: str,
    evidence: str,
    draft: str,
    judge_sys: str,
    judge_user_template: str,
    endpoint: str,
    model: str,
    min_pass: float,
    max_refine: int = 1,
    system_prompt: str | None = None,
    user_template: str | None = None,
    generation_kwargs: Optional[Dict] = None,
) -> str:
    """Evaluate and optionally refine a draft reply using a judge model."""

    generation_kwargs = generation_kwargs or {}

    def call_judge(reply: str) -> Dict:
        judge_prompt = judge_user_template.format(
            client_last=client_last,
            evidence=evidence,
            reply=reply,
        )
        content = _chat_completion(
            judge_sys,
            judge_prompt,
            model,
            endpoint,
            generation_kwargs.get("max_new_tokens", 220),
            generation_kwargs.get("temperature", 0.7),
            generation_kwargs.get("top_p", 0.95),
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            LOGGER.warning("Judge response is not valid JSON: %s", content)
            return {"scores": {}, "advice": ""}

    scores_payload = call_judge(draft)
    scores = scores_payload.get("scores", {})
    advice = scores_payload.get("advice", "")
    need_refine = any(float(value) < float(min_pass) for value in scores.values()) if scores else False

    if not need_refine or max_refine <= 0:
        return draft

    system_prompt = system_prompt or MC_SYSTEM
    base_prompt = (user_template or MC_USER_FMT).format(client_last=client_last, evidence=evidence)
    current_reply = draft

    for attempt in range(max_refine):
        refine_instruction = (
            "【督导建议】\n"
            f"{advice or '请在保持真诚共情的同时补充督导指出的要素。'}\n"
            "请根据督导建议重写你的回复，并保留上一版中已经做得好的部分。\n"
            "【上一版回复】\n"
            f"{current_reply}"
        )
        refined_prompt = f"{base_prompt}\n\n{refine_instruction}"
        current_reply = generate_reply(
            system_prompt,
            refined_prompt,
            model,
            endpoint,
            generation_kwargs.get("max_new_tokens", 220),
            generation_kwargs.get("temperature", 0.7),
            generation_kwargs.get("top_p", 0.95),
        )

        if attempt + 1 >= max_refine:
            break

        scores_payload = call_judge(current_reply)
        scores = scores_payload.get("scores", {})
        advice = scores_payload.get("advice", advice)
        need_refine = any(float(value) < float(min_pass) for value in scores.values()) if scores else False
        if not need_refine:
            break

    return current_reply

