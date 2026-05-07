"""
features/pipeline.py
─────────────────────
Orchestrates the full feature-building pipeline:

  1. Build semantic texts  (features/semantic.py)
  2. Embed with Ollama     (features/embedder.py)
  3. Build structured feats (features/structured.py)
  4. Concatenate [emb | struct] → final feature tensor per node type
  5. Optionally cache to disk (pickle) to avoid re-embedding on reruns
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from features.semantic import build_semantic_texts
from features.embedder import OllamaEmbedder
from features.structured import build_structured_features
from config.settings import EMBEDDING_DIM

log = logging.getLogger(__name__)

_CACHE_PATH = "outputs/feature_cache.pkl"


class FeaturePipeline:
    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        use_cache: bool = True,
    ):
        self.embedder = embedder or OllamaEmbedder()
        self.use_cache = use_cache

    # ── public API ────────────────────────────────────────────────────────────

    def build(
        self,
        nodes: Dict[str, List[Dict[str, Any]]],
        force_rebuild: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns {node_type: FloatTensor[N, F]} where
        F = EMBEDDING_DIM + STRUCTURED_DIM[node_type].
        """
        if self.use_cache and not force_rebuild and os.path.exists(_CACHE_PATH):
            log.info("Attempting to load feature cache from %s", _CACHE_PATH)
            try:
                with open(_CACHE_PATH, "rb") as f:
                    cached_tensors = pickle.load(f)

                # Validate cache against current node counts. If the number of
                # nodes for any type has changed, the cache is invalid.
                is_valid = True
                all_types = set(nodes.keys()) | set(cached_tensors.keys())
                for ntype in all_types:
                    num_nodes_current = len(nodes.get(ntype, []))
                    num_nodes_cached = cached_tensors.get(ntype, torch.empty(0)).shape[0]

                    if num_nodes_current != num_nodes_cached:
                        log.warning(
                            "Cache invalid for '%s': current data has %d nodes, cache has %d.",
                            ntype, num_nodes_current, num_nodes_cached
                        )
                        is_valid = False
                        break

                if is_valid:
                    log.info("Feature cache is valid and will be used.")
                    return cached_tensors
            except Exception as e:
                log.warning("Could not load or validate cache (%s). Rebuilding.", e)

        log.info("=== Building semantic texts ===")
        texts = build_semantic_texts(nodes)

        log.info("=== Embedding with Ollama ===")
        semantic_arrays = self._embed_all(texts)

        log.info("=== Building structured features ===")
        structured_arrays = build_structured_features(nodes)

        log.info("=== Concatenating feature buckets ===")
        feature_tensors = self._concat(semantic_arrays, structured_arrays)

        if self.use_cache:
            os.makedirs("outputs", exist_ok=True)
            with open(_CACHE_PATH, "wb") as f:
                pickle.dump(feature_tensors, f)
            log.info("Feature cache saved -> %s", _CACHE_PATH)

        return feature_tensors

    # ── private ───────────────────────────────────────────────────────────────

    def _embed_all(
        self,
        texts: Dict[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {}
        for ntype, txts in texts.items():
            if not txts:
                arrays[ntype] = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
                continue
            log.info("  Embedding %d %s nodes …", len(txts), ntype)
            vecs = self.embedder.embed(txts)
            arr = np.array(vecs, dtype=np.float32)
            # safety: clip / pad to expected dim
            if arr.shape[1] != EMBEDDING_DIM:
                log.warning(
                    "%s embeddings dim=%d, expected %d – padding",
                    ntype, arr.shape[1], EMBEDDING_DIM,
                )
                arr = _pad_or_clip(arr, EMBEDDING_DIM)
            arrays[ntype] = arr
            log.info("  %s semantic shape: %s", ntype, arr.shape)
        return arrays

    def _concat(
        self,
        semantic: Dict[str, np.ndarray],
        structured: Dict[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for ntype in semantic:
            sem = semantic[ntype]
            strt = structured.get(ntype, np.zeros((len(sem), 0), dtype=np.float32))

            if sem.shape[0] == 0:
                total_dim = sem.shape[1] + strt.shape[1]
                tensors[ntype] = torch.zeros((0, total_dim), dtype=torch.float)
                continue

            combined = np.concatenate([sem, strt], axis=1)
            tensors[ntype] = torch.tensor(combined, dtype=torch.float)
            log.info(
                "%s final features: sem%s + struct%s -> %s",
                ntype, sem.shape, strt.shape, combined.shape,
            )
        return tensors


# ── util ──────────────────────────────────────────────────────────────────────

def _pad_or_clip(arr: np.ndarray, target_dim: int) -> np.ndarray:
    cur = arr.shape[1]
    if cur < target_dim:
        pad = np.zeros((arr.shape[0], target_dim - cur), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)
    return arr[:, :target_dim]