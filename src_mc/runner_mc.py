"""MC 情绪关怀流水线的主入口脚本。

负责加载配置、检索上下文、调用大模型生成回复，并按需触发
督导模型进行自我审阅。"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List

import yaml

from .llm import generate_reply, refine_with_judge
from .prompts import JUDGE_SYSTEM, JUDGE_USER_FMT, MC_SYSTEM, MC_USER_FMT
from .retriever import Retriever
from .utils import last_client_turn, read_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML 配置文件并解析为字典。"""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器，便于 CLI 使用。"""

    parser = argparse.ArgumentParser(
        description="Run the MC emotional conversation pipeline"
    )
    parser.add_argument(
        "--config",
        default="src/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data",
        default="data/Conversations_Long.jsonl",
        help="Path to the input conversations JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/Emotion_Conversatin_Result.jsonl",
        help="Where to write the generated responses JSONL.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent LLM requests to process in parallel.",
    )
    return parser


def process_conversation(
    case: Dict[str, Any],
    llm_cfg: Dict[str, Any],
    judge_cfg: Dict[str, Any],
    max_retries: int,
) -> Dict[str, str]:
    """处理单个会话，生成回复并根据需要触发自审。"""

    attempts = max(max_retries, 0) + 1
    cid = case["id"]

    for attempt in range(1, attempts + 1):
        try:
            draft = generate_reply(
                MC_SYSTEM,
                case["user_prompt"],
                llm_cfg.get("model", "/data/zhangjingwei/LL-Doctor-qwen3-8b-Model"),
                llm_cfg.get("endpoint", "http://127.0.0.1:8000/v1/chat/completions"),
                llm_cfg.get("max_new_tokens", 220),
                llm_cfg.get("temperature", 0.7),
                llm_cfg.get("top_p", 0.95),
            )

            final = draft
            if judge_cfg.get("enable", False):
                generation_kwargs = {
                    "max_new_tokens": llm_cfg.get("max_new_tokens", 220),
                    "temperature": llm_cfg.get("temperature", 0.7),
                    "top_p": llm_cfg.get("top_p", 0.95),
                }
                final = refine_with_judge(
                    case["client_last"],
                    case["evidence"],
                    draft,
                    JUDGE_SYSTEM,
                    JUDGE_USER_FMT,
                    llm_cfg.get("endpoint", "http://127.0.0.1:8000/v1/chat/completions"),
                    llm_cfg.get("model", "/data/zhangjingwei/LL-Doctor-qwen3-8b-Model"),
                    judge_cfg.get("min_pass_score", 3.5),
                    judge_cfg.get("max_refine", 1),
                    MC_SYSTEM,
                    MC_USER_FMT,
                    generation_kwargs,
                )

            return {"id": cid, "predicted_response": final}
        except Exception:  # noqa: BLE001
            LOGGER.exception(
                "Processing conversation %s failed on attempt %d/%d", cid, attempt, attempts
            )
            if attempt < attempts:
                LOGGER.info("Retrying conversation %s", cid)

    LOGGER.error("Falling back to empty response for conversation %s", cid)
    return {"id": cid, "predicted_response": ""}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = build_argument_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = args.data
    output_path = args.output

    LOGGER.info("Loading dataset from %s", data_path)
    conversations = list(read_jsonl(data_path))

    retriever_cfg = cfg.get("mc", {})
    # 根据配置初始化检索器，用于召回历史上下文
    retriever = Retriever(
        embed_model=retriever_cfg.get("embed_model", "BAAI/bge-m3"),
        use_faiss=retriever_cfg.get("use_faiss", True),
        index_path=retriever_cfg.get("index_path", "./indexes/mc_faiss.index"),
        store_path=retriever_cfg.get("store_path", "./indexes/mc_store.jsonl"),
        chunk_tokens=retriever_cfg.get("chunk_tokens", 128),
        overlap_tokens=retriever_cfg.get("overlap_tokens", 20),
        normalize=retriever_cfg.get("normalize", True),
    )
    retriever.build_or_load(conversations)

    llm_cfg = cfg.get("llm", {})
    judge_cfg = cfg.get("self_judge", {})

    concurrency = args.concurrency or llm_cfg.get("concurrent_requests", 1)
    concurrency = max(int(concurrency or 1), 1)
    LOGGER.info("Using concurrency level: %d", concurrency)

    request_timeout = llm_cfg.get("request_timeout")
    timeout_value = float(request_timeout) if request_timeout is not None else None
    max_retries = int(llm_cfg.get("max_retries", 0))

    prepared_cases: List[Dict[str, Any]] = []
    results: Dict[int, Dict[str, str]] = {}

    for idx, item in enumerate(conversations):
        cid = item.get("id")
        dialog = item.get("conversation_history", "")
        client_last = last_client_turn(dialog)

        if not client_last:
            LOGGER.warning("Conversation %s has no client turn; skipping.", cid)
            results[idx] = {"id": cid, "predicted_response": ""}
            continue

        hits = retriever.search(client_last, k=retriever_cfg.get("retriever_topk", 4))
        evidence = "\n".join(hit["text"] for hit in hits) if hits else ""

        prepared_cases.append(
            {
                "index": idx,
                "id": cid,
                "client_last": client_last,
                "evidence": evidence,
                "user_prompt": MC_USER_FMT.format(
                    client_last=client_last, evidence=evidence
                ),
            }
        )

    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for case in prepared_cases:
            futures.append((case, executor.submit(process_conversation, case, llm_cfg, judge_cfg, max_retries)))

        for case, future in futures:
            try:
                if timeout_value is not None:
                    result = future.result(timeout=timeout_value)
                else:
                    result = future.result()
            except TimeoutError:
                LOGGER.error(
                    "Processing conversation %s timed out after %s seconds", case["id"], request_timeout
                )
                future.cancel()
                result = {"id": case["id"], "predicted_response": ""}
            except Exception:
                LOGGER.exception("Processing conversation %s failed", case["id"])
                result = {"id": case["id"], "predicted_response": ""}

            results[case["index"]] = result

    outputs = [
        results.get(
            idx,
            {
                "id": conversations[idx].get("id"),
                "predicted_response": "",
            },
        )
        for idx in range(len(conversations))
    ]

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    # 将全部结果写出为 JSONL，便于后续评估或提交
    write_jsonl(output_path, outputs)
    LOGGER.info("Saved %d responses to %s", len(outputs), output_path)


if __name__ == "__main__":
    main()
