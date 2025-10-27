"""Semantic retriever built on top of FAISS/Sentence-Transformers."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    meta: Dict


class Retriever:
    """Semantic retriever for long counseling conversations."""

    def __init__(
        self,
        embed_model: str,
        use_faiss: bool,
        index_path: str,
        store_path: str,
        chunk_tokens: int,
        overlap_tokens: int,
        normalize: bool = True,
    ) -> None:
        self.embed_model_name = embed_model
        self.use_faiss = bool(use_faiss)
        self.index_path = index_path
        self.store_path = store_path
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.normalize = normalize

        self._embedder: SentenceTransformer | None = None
        self._index = None
        self._chunks: List[Chunk] = []
        self._embeddings: np.ndarray | None = None

        if self.use_faiss and faiss is None:
            LOGGER.warning("FAISS not available, falling back to numpy search.")
            self.use_faiss = False

    # ------------------------------------------------------------------
    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            LOGGER.info("Loading embedding model %s", self.embed_model_name)
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    # ------------------------------------------------------------------
    def build_or_load(self, all_dialogs: Sequence[Dict]) -> None:
        """Build or load an index from cached assets."""

        index_exists = os.path.exists(self.index_path) if self.use_faiss else True
        store_exists = os.path.exists(self.store_path)
        emb_path = self._embedding_cache_path()
        emb_exists = os.path.exists(emb_path)

        if store_exists and index_exists and (emb_exists or self.use_faiss):
            LOGGER.info("Loading cached retriever assets from disk.")
            self._load_cached(index_exists, store_exists, emb_exists)
            return

        LOGGER.info("Building new retriever assets (faiss=%s).", self.use_faiss)
        chunks = self._build_chunks(all_dialogs)
        self._chunks = chunks

        if not chunks:
            LOGGER.warning("No chunks were generated from the provided dialogs.")
            self._embeddings = np.empty((0, 0), dtype="float32")
            self._persist_chunks(chunks)
            emb_path = self._embedding_cache_path()
            dir_name = os.path.dirname(emb_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            np.save(emb_path, self._embeddings)
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=self.normalize, convert_to_numpy=True
        )
        self._embeddings = embeddings.astype("float32")

        if self.use_faiss:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self._embeddings)
            faiss.write_index(index, self.index_path)
            self._index = index

        self._persist_chunks(chunks)
        dir_name = os.path.dirname(emb_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        np.save(emb_path, self._embeddings)

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 4, prefer_client: bool = True) -> List[Dict]:
        """Return a ranked list of context chunks."""

        if not self._chunks:
            raise RuntimeError("Retriever not initialised. Call build_or_load first.")

        query_emb = self.embedder.encode(
            [query], normalize_embeddings=self.normalize, convert_to_numpy=True
        ).astype("float32")

        if self.use_faiss and self._index is not None:
            scores, indices = self._index.search(query_emb, k)
            return self._gather_results(scores[0], indices[0], prefer_client)

        if self._embeddings is None or self._embeddings.size == 0:
            LOGGER.warning("Retriever embeddings are empty; returning no context.")
            return []

        similarities = np.dot(self._embeddings, query_emb[0])
        top_idx = np.argsort(similarities)[::-1][:k]
        scores = similarities[top_idx]
        return self._gather_results(scores, top_idx, prefer_client)

    # ------------------------------------------------------------------
    def _gather_results(self, scores: Sequence[float], indices: Sequence[int], prefer_client: bool) -> List[Dict]:
        results: List[Dict] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            bonus = 0.0
            if prefer_client and chunk.meta.get("contains_client"):
                bonus = 0.05
            results.append(
                {
                    "text": chunk.text,
                    "score": float(score + bonus),
                    "meta": chunk.meta,
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    def _build_chunks(self, dialogs: Sequence[Dict]) -> List[Chunk]:
        chunks: List[Chunk] = []
        token_pattern = re.compile(r"[\u4e00-\u9fff]|\w+|[^\s\w]")

        for dialog in dialogs:
            dialog_id = dialog.get("id")
            history = dialog.get("conversation_history", "")
            if not history:
                continue

            lines: List[tuple[str, int, bool]] = []
            for raw_line in history.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                tokens = token_pattern.findall(line)
                lines.append((line, len(tokens), line.startswith("Client:")))

            if not lines:
                continue

            buffer: List[tuple[str, int, bool]] = []
            buffer_tokens = 0
            chunk_idx = 0

            def flush_buffer() -> None:
                nonlocal buffer, buffer_tokens, chunk_idx
                if not buffer:
                    return
                text = "\n".join(item[0] for item in buffer)
                contains_client = any(item[2] for item in buffer)
                meta = {
                    "dialog_id": dialog_id,
                    "chunk_id": chunk_idx,
                    "token_length": sum(item[1] for item in buffer),
                    "contains_client": contains_client,
                }
                chunks.append(Chunk(text=text, meta=meta))
                chunk_idx += 1

                if self.overlap_tokens <= 0:
                    buffer = []
                    buffer_tokens = 0
                    return

                overlap: List[tuple[str, int, bool]] = []
                overlap_tokens = 0
                for item in reversed(buffer):
                    overlap.insert(0, item)
                    overlap_tokens += item[1]
                    if overlap_tokens >= self.overlap_tokens:
                        break
                buffer = overlap
                buffer_tokens = sum(item[1] for item in buffer)

            for line, token_count, is_client in lines:
                if buffer and buffer_tokens + token_count > self.chunk_tokens:
                    flush_buffer()
                buffer.append((line, token_count, is_client))
                buffer_tokens += token_count

            flush_buffer()

        LOGGER.info("Built %d chunks from %d dialogs.", len(chunks), len(dialogs))
        return chunks

    # ------------------------------------------------------------------
    def _persist_chunks(self, chunks: Sequence[Chunk]) -> None:
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps({"text": chunk.text, "meta": chunk.meta}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def _load_cached(self, index_exists: bool, store_exists: bool, emb_exists: bool) -> None:
        if store_exists:
            with open(self.store_path, "r", encoding="utf-8") as f:
                self._chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        if index_exists and self.use_faiss:
            self._index = faiss.read_index(self.index_path)
        emb_path = self._embedding_cache_path()
        if emb_exists:
            self._embeddings = np.load(emb_path).astype("float32")

    # ------------------------------------------------------------------
    def _embedding_cache_path(self) -> str:
        base, _ = os.path.splitext(self.index_path)
        return f"{base}_embeddings.npy"

