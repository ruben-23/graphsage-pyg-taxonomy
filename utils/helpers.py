"""
utils/logging_setup.py  +  utils/helpers.py
────────────────────────────────────────────
Shared utilities for the pipeline.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from typing import Any, Dict

import torch


# ── logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        encoding='utf-8',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("outputs/pipeline.log", mode="w"),
        ],
    )


# ── checkpoints ──────────────────────────────────────────────────────────────

def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── summary printer ───────────────────────────────────────────────────────────

def print_graph_summary(
    nodes: Dict[str, list],
    edges: Dict[str, list],
    feature_tensors: Dict[str, torch.Tensor],
) -> None:
    log = logging.getLogger(__name__)
    log.info("-" * 60)
    log.info("GRAPH SUMMARY")
    log.info("-" * 60)
    total_nodes = sum(len(v) for v in nodes.values())
    total_edges = sum(len(v) for v in edges.values())
    log.info("Total nodes : %d", total_nodes)
    log.info("Total edges : %d", total_edges)
    log.info("")
    log.info("%-15s  %8s  %10s", "Node Type", "Count", "Feat Dim")
    log.info("%-15s  %8s  %10s", "-" * 15, "-" * 8, "-" * 10)
    for ntype, rows in nodes.items():
        ft = feature_tensors.get(ntype)
        fdim = ft.size(1) if ft is not None and ft.dim() > 1 else "?"
        log.info("%-15s  %8d  %10s", ntype, len(rows), fdim)
    log.info("")
    log.info("%-20s  %8s", "Relation Type", "Count")
    log.info("%-20s  %8s", "-" * 20, "-" * 8)
    for rel, rows in edges.items():
        log.info("%-20s  %8d", rel, len(rows))
    log.info("-" * 60)