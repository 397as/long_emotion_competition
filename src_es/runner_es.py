"""Runner script for the Emotion Summary (ES) pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import re
import traceback
from typing import Any, Dict, Iterable, List

import yaml

from src_mc.llm import generate_reply
from src_mc.retriever import Retriever
from src_mc.utils import read_jsonl, write_jsonl

from .prompts_es import ES_SYSTEM, ES_USER_FMT

LOGGER = logging.getLogger(__name__)

FIELDS = [
    "predicted_cause",
    "predicted_symptoms",
    "predicted_treatment_process",
    "predicted_illness_Characteristics",
    "predicted_treatment_effect",
]

FIELD_KEY_MAPPING = {
    "predicted_cause": ("cause",),
    "predicted_symptoms": ("symptoms",),
    "predicted_treatment_process": ("treatment_process",),
    "predicted_illness_Characteristics": ("illness_characteristics",),
    "predicted_treatment_effect": ("treatment_effect",),
}

DEFAULT_FILL = "未见明确描述"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Emotion Summary pipeline")
    parser.add_argument(
        "--config",
        default="src_es/config_es.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data",
        default="data/Emotion_Summary.jsonl",
        help="Input JSONL file containing counseling cases.",
    )
    parser.add_argument(
        "--output",
        default="outputs/Emotion_Summary_Result.jsonl",
        help="Where to write the generated summaries (JSONL).",
    )
    parser.add_argument(
        "--error_log",
        default="outputs/ES_errors.log",
        help="File to append errors encountered during processing.",
    )
    return parser


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stringify_chunks(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    parts.append(item)
            elif item is not None:
                parts.append(str(item).strip())
        return [p for p in parts if p]
    return [str(value).strip()]


def _merge_case(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.extend(_stringify_chunks(item.get("case_description")))
    parts.extend(_stringify_chunks(item.get("consultation_process")))
    parts.extend(_stringify_chunks(item.get("experience_and_reflection")))
    return "\n".join(part for part in parts if part)


def _sanitize(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r", " ")
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("\u200b", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace('\"', '"')
    return cleaned.strip()


def _parse_json_strict(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}


def _generate_structured_output(evidence: str, llm_cfg: Dict[str, Any], attempts: int = 2) -> Dict[str, Any]:
    for attempt in range(attempts):
        user_prompt = ES_USER_FMT.format(evidence=evidence)
        raw = generate_reply(
            ES_SYSTEM,
            user_prompt,
            llm_cfg.get("model", ""),
            llm_cfg.get("endpoint", ""),
            llm_cfg.get("max_new_tokens", 512),
            llm_cfg.get("temperature", 0.4),
            llm_cfg.get("top_p", 0.9),
        )
        parsed = _parse_json_strict(raw)
        if parsed:
            return parsed
        LOGGER.warning("Failed to parse LLM output on attempt %s: %s", attempt + 1, raw)
    return {}


def _build_retriever(cfg: Dict[str, Any], records: List[Dict[str, Any]]) -> Retriever:
    es_cfg = cfg.get("es", {})
    retriever = Retriever(
        embed_model=es_cfg.get("embed_model", "BAAI/bge-m3"),
        use_faiss=es_cfg.get("use_vector_store", True),
        index_path=es_cfg.get("index_path", "./indexes/es/faiss.index"),
        store_path=es_cfg.get("store_path", "./indexes/es/store.jsonl"),
        chunk_tokens=es_cfg.get("chunk_tokens", 128),
        overlap_tokens=es_cfg.get("overlap_tokens", 20),
        normalize=es_cfg.get("normalize", True),
    )
    dialogs = [
        {"id": item.get("id"), "conversation_history": _merge_case(item)}
        for item in records
    ]
    retriever.build_or_load(dialogs)
    return retriever


def _extract_field(parsed: Dict[str, Any], field: str) -> str:
    for key in FIELD_KEY_MAPPING.get(field, (field,)):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize(value)
    return ""


def process_case(
    item: Dict[str, Any],
    llm_cfg: Dict[str, Any],
    es_cfg: Dict[str, Any],
    retriever: Retriever | None,
    queries: Dict[str, str],
) -> Dict[str, Any]:
    case_id = item.get("id")
    case_id_str = str(case_id)
    merged_text = _merge_case(item)
    per_field: Dict[str, str] = {}

    if retriever is None:
        parsed = _generate_structured_output(merged_text, llm_cfg)
        for field in FIELDS:
            extracted = _extract_field(parsed, field)
            per_field[field] = extracted or DEFAULT_FILL
        return {"id": case_id, **per_field}

    top_k = int(es_cfg.get("retriever_topk", 4))
    search_k = max(int(es_cfg.get("retriever_search_k", top_k * 4)), top_k)
    for field in FIELDS:
        query = queries.get(field) if isinstance(queries, dict) else None
        query = (query or field).strip()
        hits = retriever.search(query, k=search_k, prefer_client=False)

        total_hits = len(hits)
        filtered_hits = [
            hit for hit in hits if str(hit.get("meta", {}).get("dialog_id")) == case_id_str
        ]
        filtered_count = len(filtered_hits)
        filtered_hits.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        filtered_hits = filtered_hits[:top_k]

        fallback_used = False
        if filtered_hits:
            evidence = "\n".join(hit["text"] for hit in filtered_hits)
        else:
            evidence = merged_text
            fallback_used = True

        LOGGER.debug(
            "Case %s field %s retriever hits: total=%d filtered=%d fallback=%s",
            case_id,
            field,
            total_hits,
            filtered_count,
            fallback_used,
        )
        parsed = _generate_structured_output(evidence, llm_cfg)
        extracted = _extract_field(parsed, field)
        per_field[field] = extracted or DEFAULT_FILL

    return {"id": case_id, **per_field}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = build_argument_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm_cfg = cfg.get("llm", {})
    es_cfg = cfg.get("es", {})
    queries = cfg.get("per_field_queries", {})

    LOGGER.info("Loading cases from %s", args.data)
    records = list(read_jsonl(args.data))

    retriever: Retriever | None = None
    if es_cfg.get("use_rag", True):
        LOGGER.info("Building / loading retriever for ES task (index=%s)", es_cfg.get("index_path"))
        retriever = _build_retriever(cfg, records)

    outputs: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for item in records:
        case_id = item.get("id")
        try:
            result = process_case(item, llm_cfg, es_cfg, retriever, queries)
            outputs.append(result)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to process case %s", case_id)
            errors.append(
                {
                    "id": case_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    write_jsonl(args.output, outputs)
    LOGGER.info("Saved %d summaries to %s", len(outputs), args.output)

    if errors:
        LOGGER.warning("Encountered %d errors; writing to %s", len(errors), args.error_log)
        write_jsonl(args.error_log, errors)


if __name__ == "__main__":
    main()
