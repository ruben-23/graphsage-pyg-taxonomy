"""
features/embedder.py
────────────────────
Thin wrapper around the local Ollama REST API for text embedding.
Batches requests to avoid overwhelming the server.
"""

from __future__ import annotations

import logging
import time
import concurrent.futures
from typing import List

import requests

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDING_DIM

log = logging.getLogger(__name__)

_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"


class OllamaEmbedder:
    """
    Embeds a list of strings using the local Ollama `nomic-embed-text` model.
    Falls back to zero vectors on error (so the pipeline doesn't crash
    during development when Ollama isn't running).
    """

    # def __init__(
    #     self,
    #     model: str = OLLAMA_MODEL,
    #     batch_size: int = 32,
    #     retry_delay: float = 1.0,
    #     max_retries: int = 3,
    #     max_workers: int = 8,
    # ):
    #     self.model = model
    #     self.batch_size = batch_size
    #     self.retry_delay = retry_delay
    #     self.max_retries = max_retries
    #     self.max_workers = max_workers
    #     self._dim = EMBEDDING_DIM

    # # ── public ────────────────────────────────────────────────────────────────

    # def embed(self, texts: List[str]) -> List[List[float]]:
    #     """
    #     Embed a list of strings.  Returns a list of float vectors, one per text.
    #     """
    #     all_embeddings: List[List[float]] = []
    #     for i in range(0, len(texts), self.batch_size):
    #         batch = texts[i : i + self.batch_size]
    #         embeddings = self._embed_batch(batch)
    #         all_embeddings.extend(embeddings)
    #         if i % (self.batch_size * 10) == 0 and i > 0:
    #             log.info("  embedded %d / %d texts", i, len(texts))
    #     log.info("Embedded %d texts total", len(texts))
    #     return all_embeddings

    # def embed_one(self, text: str) -> List[float]:
    #     return self.embed([text])[0]

    # # ── private ───────────────────────────────────────────────────────────────

    # def _embed_batch(self, texts: List[str]) -> List[List[float]]:
    #     results: List[List[float]] = []
    #     # for text in texts:
    #     #     vec = self._embed_single_with_retry(text)
    #     #     results.append(vec)
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    #         # executor.map processes the texts in parallel and returns results
    #         # in the same order as the input.
    #         results = list(executor.map(self._embed_single_with_retry, texts))
    #     return results

    # def _embed_single_with_retry(self, text: str) -> List[float]:
    #     for attempt in range(self.max_retries):
    #         try:
    #             resp = requests.post(
    #                 _EMBED_URL,
    #                 json={"model": self.model, "prompt": text},
    #                 timeout=30,
    #             )
    #             resp.raise_for_status()
    #             vec = resp.json().get("embedding", [])
    #             if vec:
    #                 return vec
    #             log.warning("Empty embedding returned for text snippet")
    #             return [0.0] * self._dim
    #         except Exception as exc:
    #             log.warning(
    #                 "Ollama attempt %d/%d failed: %s",
    #                 attempt + 1, self.max_retries, exc,
    #             )
    #             if attempt < self.max_retries - 1:
    #                 time.sleep(self.retry_delay)
    #     log.error("All retries failed - returning zero vector")
    #     return [0.0] * self._dim

    # @property
    # def dim(self) -> int:
    #     return self._dim

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        batch_size: int = 32,
        retry_delay: float = 1.0,
        max_retries: int = 3,
        max_workers: int = 8,
    ):
        self.model = model
        self.batch_size = batch_size
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.max_workers = max_workers
        self._dim = EMBEDDING_DIM
        
        # Keep TCP connections alive between requests
        self.session = requests.Session()

    # ── public ────────────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings. Returns a list of float vectors, one per text.
        """
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
            if i % (self.batch_size * 10) == 0 and i > 0:
                log.info("  embedded %d / %d texts", i, len(texts))
        log.info("Embedded %d texts total", len(texts))
        return all_embeddings

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    # ── private ───────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Removed the redundant synchronous loop! 
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # executor.map processes the texts in parallel and returns results
            # in the same order as the input.
            results = list(executor.map(self._embed_single_with_retry, texts))
        return results

    def _embed_single_with_retry(self, text: str) -> List[float]:
        for attempt in range(self.max_retries):
            try:
                # Use the session instead of requests.post
                resp = self.session.post(
                    _EMBED_URL,
                    # json={"model": self.model, "prompt": text},
                    json={
                        "model": self.model,
                        "input": text,
                        "dimensions": self._dim,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                # vec = resp.json().get("embedding", [])
                # if vec:
                #     return vec
                vec = resp.json().get("embeddings", [])
                if vec:
                    # The response should be a flat list of floats. Some model/API
                    # versions might incorrectly wrap it, e.g., [[...]]. This
                    # check handles that case by unwrapping it.
                    if len(vec) > 0 and isinstance(vec[0], list):
                        return vec[0]  # We got [[...]], return the inner list.
                    return vec  # We got [...] as expected.


                log.warning("Empty embedding returned for text snippet")
                return [0.0] * self._dim
            except Exception as exc:
                log.warning(
                    "Ollama attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        log.error("All retries failed - returning zero vector")
        return [0.0] * self._dim

    @property
    def dim(self) -> int:
        return self._dim