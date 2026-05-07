"""
training/trainer.py
────────────────────
Self-supervised link-prediction trainer for HeteroGraphSAGE.

Objective
---------
For each relation type, sample positive (real) edges and
negative (random) edges.  Train with binary cross-entropy
so the model learns to produce embeddings where connected
nodes are closer than random ones.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling

from models.graphsage import HeteroGraphSAGE
from config.settings import GraphSAGEConfig, TrainingConfig, CHECKPOINT_PATH

log = logging.getLogger(__name__)


# ── edge split helper ─────────────────────────────────────────────────────────

def split_edges(
    data: HeteroData,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[HeteroData, Dict, Dict]:
    """
    Split every edge type into train / val / test sets.
    Returns (train_data, val_pos_edges, test_pos_edges)
    where *_pos_edges is {triple: edge_index}.
    The returned train_data contains only training edges.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    train_data = copy.deepcopy(data)
    val_pos:  Dict = {}
    test_pos: Dict = {}

    for triple, store in data.edge_items():
        src_type, rel, dst_type = triple
        # Skip reverse edges (they'll mirror the forward split)
        if rel.startswith("rev_"):
            continue
        ei = store.edge_index
        n  = ei.size(1)
        if n < 10:
            continue   # too few edges to split meaningfully

        perm = torch.randperm(n)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))

        test_idx  = perm[:n_test]
        val_idx   = perm[n_test : n_test + n_val]
        train_idx = perm[n_test + n_val :]

        train_data[triple].edge_index = ei[:, train_idx]
        val_pos[triple]  = ei[:, val_idx]
        test_pos[triple] = ei[:, test_idx]

        # Also update reverse
        rev_triple = (dst_type, f"rev_{rel}", src_type)
        if rev_triple in data.edge_types:
            train_data[rev_triple].edge_index = ei[:, train_idx].flip(0)

    return train_data, val_pos, test_pos


# ── loss helper ───────────────────────────────────────────────────────────────

def link_pred_loss(
    model: HeteroGraphSAGE,
    z_dict: Dict[str, torch.Tensor],
    pos_edge_dict: Dict,
    num_neg_samples: int = 1,
) -> torch.Tensor:
    """BCE loss across all relation types with at least one positive edge."""
    total_loss = torch.tensor(0.0, requires_grad=True)
    count = 0

    for triple, pos_ei in pos_edge_dict.items():
        src_type, _, dst_type = triple
        if src_type not in z_dict or dst_type not in z_dict:
            continue

        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]

        n_src = z_src.size(0)
        n_dst = z_dst.size(0)

        # Positive scores
        s_idx = pos_ei[0]
        d_idx = pos_ei[1]
        if s_idx.max() >= n_src or d_idx.max() >= n_dst:
            continue

        pos_score = model.decode(z_src[s_idx], z_dst[d_idx])

        # Negative samples
        num_neg = pos_ei.size(1) * num_neg_samples
        neg_s = torch.randint(0, n_src, (num_neg,), device=pos_ei.device)
        neg_d = torch.randint(0, n_dst, (num_neg,), device=pos_ei.device)
        neg_score = model.decode(z_src[neg_s], z_dst[neg_d])

        labels = torch.cat([
            torch.ones(pos_score.size(0)),
            torch.zeros(neg_score.size(0)),
        ]).to(pos_ei.device)
        scores = torch.cat([pos_score, neg_score])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        total_loss = total_loss + loss
        count += 1

    return total_loss / max(count, 1)


# ── AUC helper ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_auc(
    model: HeteroGraphSAGE,
    data: HeteroData,
    pos_edge_dict: Dict,
    device: torch.device,
) -> float:
    from sklearn.metrics import roc_auc_score
    import numpy as np

    model.eval()
    z_dict = model(data.x_dict, data.edge_index_dict)

    all_labels, all_scores = [], []

    for triple, pos_ei in pos_edge_dict.items():
        src_type, _, dst_type = triple
        if src_type not in z_dict or dst_type not in z_dict:
            continue

        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]
        n_src, n_dst = z_src.size(0), z_dst.size(0)

        s_idx = pos_ei[0].to(device)
        d_idx = pos_ei[1].to(device)
        if s_idx.max() >= n_src or d_idx.max() >= n_dst:
            continue

        pos_scores = model.decode(z_src[s_idx], z_dst[d_idx]).cpu().numpy()

        num_neg = len(s_idx)
        neg_s = torch.randint(0, n_src, (num_neg,))
        neg_d = torch.randint(0, n_dst, (num_neg,))
        neg_scores = model.decode(z_src[neg_s], z_dst[neg_d]).cpu().numpy()

        all_labels.extend([1] * len(pos_scores) + [0] * len(neg_scores))
        all_scores.extend(pos_scores.tolist() + neg_scores.tolist())

    if len(set(all_labels)) < 2:
        return 0.5
    return float(roc_auc_score(all_labels, all_scores))


# ── trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model:     HeteroGraphSAGE,
        cfg:       TrainingConfig,
        gnn_cfg:   GraphSAGEConfig,
    ):
        self.model   = model
        self.cfg     = cfg
        self.device  = torch.device(cfg.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=25, factor=0.5
        )

        # History for plotting
        self.train_losses: List[float] = []
        self.val_aucs:     List[float] = []
        self.val_losses:   List[float] = []

    def train(
        self,
        train_data:  HeteroData,
        val_pos:     Dict,
        test_pos:    Dict,
    ) -> Dict[str, Any]:
        """Full training loop. Returns result dict."""
        os.makedirs("outputs", exist_ok=True)
        best_val_auc   = 0.0
        patience_count = 0
        best_state     = None

        # Move all data to device
        train_data = train_data.to(self.device)
        val_pos_d  = {k: v.to(self.device) for k, v in val_pos.items()}
        test_pos_d = {k: v.to(self.device) for k, v in test_pos.items()}

        log.info("Starting training for %d epochs", self.cfg.epochs)

        for epoch in range(1, self.cfg.epochs + 1):
            # ── train step ──
            self.model.train()
            self.optimizer.zero_grad()
            z_dict = self.model(train_data.x_dict, train_data.edge_index_dict)

            # Build pos-edge dict from training data
            train_pos = {
                triple: store.edge_index
                for triple, store in train_data.edge_items()
                if not triple[1].startswith("rev_") and store.edge_index.size(1) > 0
            }
            loss = link_pred_loss(self.model, z_dict, train_pos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # ── val step ──
            val_auc  = evaluate_auc(self.model, train_data, val_pos_d, self.device)
            val_loss = self._val_loss(train_data, val_pos_d)

            self.train_losses.append(float(loss.detach()))
            self.val_aucs.append(val_auc)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    "Epoch %3d | loss %.4f | val_loss %.4f | val_AUC %.4f",
                    epoch, loss.item(), val_loss, val_auc,
                )

            # ── early stopping ──
            if val_auc > best_val_auc + 1e-4:
                best_val_auc   = val_auc
                patience_count = 0
                best_state     = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, CHECKPOINT_PATH)
            else:
                patience_count += 1
                if patience_count >= self.cfg.patience:
                    log.info("Early stopping at epoch %d", epoch)
                    break

        # ── final eval on best model ──
        if best_state is not None:
            self.model.load_state_dict(best_state)
        test_auc = evaluate_auc(self.model, train_data, test_pos_d, self.device)
        log.info("Test AUC: %.4f", test_auc)

        return {
            "train_losses": self.train_losses,
            "val_aucs":     self.val_aucs,
            "val_losses":   self.val_losses,
            "best_val_auc": best_val_auc,
            "test_auc":     test_auc,
            "epochs_run":   len(self.train_losses),
        }

    @torch.no_grad()
    def _val_loss(self, data: HeteroData, val_pos: Dict) -> float:
        self.model.eval()
        z = self.model(data.x_dict, data.edge_index_dict)
        return float(link_pred_loss(self.model, z, val_pos))

    @torch.no_grad()
    def get_embeddings(
        self,
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        data = data.to(self.device)
        return {k: v.cpu() for k, v in self.model(data.x_dict, data.edge_index_dict).items()}