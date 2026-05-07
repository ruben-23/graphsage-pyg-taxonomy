"""
evaluation/visualizer.py
─────────────────────────
Generates and saves all training/validation diagnostic charts:

  1. train_loss_curve        - loss over epochs
  2. val_auc_curve           - AUC-ROC over epochs
  3. embedding_tsne          - t-SNE of all node embeddings coloured by type
  4. embedding_umap          - UMAP (if umap-learn installed)
  5. per_type_similarity     - intra-type cosine-sim box plot
  6. degree_vs_embedding_norm - node degree vs ‖embedding‖₂
  7. edge_type_breakdown     - bar chart of edge counts per relation type
  8. feature_dim_summary     - bar chart of feature sizes per node type
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── 1 & 2: Loss / AUC curves ─────────────────────────────────────────────────

def plot_training_curves(
    result: Dict[str, Any],
    save_dir: str = "outputs/plots",
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _ensure_dir(save_dir)

    train_losses = result["train_losses"]
    val_losses   = result["val_losses"]
    val_aucs     = result["val_aucs"]
    epochs       = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GraphSAGE Training Diagnostics", fontsize=15, fontweight="bold")

    # Loss
    ax1.plot(epochs, train_losses, label="Train Loss", color="#2563EB", linewidth=2)
    ax1.plot(epochs, val_losses,   label="Val Loss",   color="#DC2626", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("BCE Loss")
    ax1.set_title("Link-Prediction Loss");  ax1.legend();  ax1.grid(alpha=0.3)

    # AUC
    ax2.plot(epochs, val_aucs, label="Val AUC", color="#16A34A", linewidth=2)
    ax2.axhline(result.get("test_auc", 0), color="#CA8A04", linewidth=1.5,
                linestyle=":", label=f"Test AUC = {result.get('test_auc', 0):.3f}")
    ax2.set_xlabel("Epoch");  ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Validation AUC-ROC");    ax2.legend();  ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved -> %s", path)


# ── 3: t-SNE embedding plot ───────────────────────────────────────────────────

def plot_tsne(
    embeddings: Dict[str, torch.Tensor],
    save_dir:   str = "outputs/plots",
    max_nodes:  int = 3000,
) -> None:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    _ensure_dir(save_dir)

    PALETTE = {
        "Student":    "#2563EB",
        "Job":        "#DC2626",
        "Skill":      "#16A34A",
        "Occupation": "#9333EA",
        "Company":    "#EA580C",
        "Project":    "#0891B2",
        "Course":     "#CA8A04",
        "Diploma":    "#DB2777",
    }

    all_vecs, all_labels = [], []
    for ntype, emb in embeddings.items():
        vecs = emb.numpy()
        if len(vecs) > max_nodes // len(embeddings):
            idx = np.random.choice(len(vecs), max_nodes // len(embeddings), replace=False)
            vecs = vecs[idx]
        all_vecs.append(vecs)
        all_labels.extend([ntype] * len(vecs))

    if not all_vecs:
        return

    X = np.vstack(all_vecs)
    log.info("Running t-SNE on %d nodes …", len(X))
    tsne = TSNE(n_components=2, perplexity=min(30, max(2, len(X) - 1)), random_state=42, max_iter=1000)
    Z = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 9))
    for ntype in dict.fromkeys(all_labels):   # preserve order
        mask = np.array(all_labels) == ntype
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=PALETTE.get(ntype, "#888"),
                   s=18, alpha=0.65, linewidths=0, label=ntype)

    ax.set_title("t-SNE of GraphSAGE Embeddings (coloured by node type)", fontsize=13)
    ax.legend(markerscale=2, framealpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])

    path = os.path.join(save_dir, "tsne_embeddings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved -> %s", path)


# ── 4: UMAP (optional) ───────────────────────────────────────────────────────

def plot_umap(
    embeddings: Dict[str, torch.Tensor],
    save_dir:   str = "outputs/plots",
    max_nodes:  int = 3000,
) -> None:
    try:
        import umap
    except ImportError:
        log.info("umap-learn not installed - skipping UMAP plot")
        return
    import matplotlib.pyplot as plt

    _ensure_dir(save_dir)
    PALETTE = ["#2563EB","#DC2626","#16A34A","#9333EA","#EA580C","#0891B2","#CA8A04","#DB2777"]

    all_vecs, all_labels = [], []
    for ntype, emb in embeddings.items():
        vecs = emb.numpy()
        if len(vecs) > max_nodes // max(len(embeddings), 1):
            idx = np.random.choice(len(vecs), max_nodes // len(embeddings), replace=False)
            vecs = vecs[idx]
        all_vecs.append(vecs)
        all_labels.extend([ntype] * len(vecs))

    if not all_vecs:
        return

    X = np.vstack(all_vecs)
    log.info("Running UMAP on %d nodes …", len(X))
    reducer = umap.UMAP(n_components=2, random_state=42)
    Z = reducer.fit_transform(X)

    types = list(dict.fromkeys(all_labels))
    fig, ax = plt.subplots(figsize=(12, 9))
    for i, ntype in enumerate(types):
        mask = np.array(all_labels) == ntype
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=PALETTE[i % len(PALETTE)],
                   s=18, alpha=0.65, linewidths=0, label=ntype)
    ax.set_title("UMAP of GraphSAGE Embeddings", fontsize=13)
    ax.legend(markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])

    path = os.path.join(save_dir, "umap_embeddings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved -> %s", path)


# ── 5: intra-type cosine similarity box plot ─────────────────────────────────

def plot_cosine_similarity_boxplot(
    embeddings: Dict[str, torch.Tensor],
    save_dir:   str = "outputs/plots",
    sample_n:   int = 200,
) -> None:
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    _ensure_dir(save_dir)

    data_by_type: Dict[str, List[float]] = {}
    for ntype, emb in embeddings.items():
        if emb.size(0) < 2:
            continue
        n = min(emb.size(0), sample_n)
        idx = torch.randperm(emb.size(0))[:n]
        e = F.normalize(emb[idx], dim=-1)
        sim_matrix = (e @ e.T).triu(diagonal=1)
        sims = sim_matrix[sim_matrix != 0].tolist()
        data_by_type[ntype] = sims

    if not data_by_type:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = list(data_by_type.keys())
    ax.boxplot([data_by_type[k] for k in labels],
               labels=labels, patch_artist=True,
               medianprops=dict(color="black", linewidth=2))
    ax.set_title("Intra-type Cosine Similarity Distribution", fontsize=13)
    ax.set_ylabel("Cosine Similarity")
    ax.set_xlabel("Node Type")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)

    path = os.path.join(save_dir, "cosine_similarity_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved -> %s", path)


# ── 6: embedding norm distribution ───────────────────────────────────────────

def plot_embedding_norms(
    embeddings: Dict[str, torch.Tensor],
    save_dir:   str = "outputs/plots",
) -> None:
    import matplotlib.pyplot as plt

    _ensure_dir(save_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    for ntype, emb in embeddings.items():
        norms = emb.norm(dim=-1).numpy()
        n_range = float(norms.max() - norms.min())
        if n_range < 1e-6:
            ax.bar([float(norms.mean())], [len(norms)], width=0.02, alpha=0.6, label=ntype)
        else:
            bins = min(30, max(2, len(norms)))
            ax.hist(norms, bins=bins, alpha=0.6, label=ntype)

    ax.set_title("Embedding ‖·‖₂ Norm Distribution per Node Type", fontsize=13)
    ax.set_xlabel("‖embedding‖₂")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    path = os.path.join(save_dir, "embedding_norms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved -> %s", path)


# ── 7: edge type breakdown ───────────────────────────────────────────────────

def plot_edge_counts(
    edges: Dict[str, list],
    save_dir: str = "outputs/plots",
) -> None:
    import matplotlib.pyplot as plt

    _ensure_dir(save_dir)

    rel_types = [k for k in edges if not k.startswith("rev_")]
    counts    = [len(edges[k]) for k in rel_types]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(rel_types, counts, color="#2563EB", alpha=0.8)
    ax.bar_label(bars, padding=3)
    ax.set_title("Edge Count per Relation Type", fontsize=13)
    ax.set_xlabel("Number of Edges")
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(save_dir, "edge_counts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved -> %s", path)


# ── 8: feature dimension summary ─────────────────────────────────────────────

def plot_feature_dims(
    feature_tensors: Dict[str, torch.Tensor],
    save_dir: str = "outputs/plots",
) -> None:
    import matplotlib.pyplot as plt

    _ensure_dir(save_dir)

    types = list(feature_tensors.keys())
    dims  = [feature_tensors[t].size(1) if feature_tensors[t].dim() > 1 else 0
             for t in types]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(types, dims, color="#16A34A", alpha=0.85)
    ax.bar_label(bars, padding=2)
    ax.set_title("Input Feature Dimension per Node Type", fontsize=13)
    ax.set_ylabel("Feature Dimension")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)

    path = os.path.join(save_dir, "feature_dims.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved -> %s", path)


# ── 9: Node count summary ─────────────────────────────────────────────────────

def plot_node_counts(
    nodes: Dict[str, list],
    save_dir: str = "outputs/plots",
) -> None:
    import matplotlib.pyplot as plt

    _ensure_dir(save_dir)

    types  = list(nodes.keys())
    counts = [len(nodes[t]) for t in types]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(types, counts, color="#9333EA", alpha=0.85)
    ax.bar_label(bars, padding=2)
    ax.set_title("Node Count per Type", fontsize=13)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)

    path = os.path.join(save_dir, "node_counts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved -> %s", path)


# ── master runner ─────────────────────────────────────────────────────────────

def run_all_plots(
    result:          Dict[str, Any],
    embeddings:      Dict[str, torch.Tensor],
    nodes:           Dict[str, list],
    edges:           Dict[str, list],
    feature_tensors: Dict[str, torch.Tensor],
    save_dir:        str = "outputs/plots",
) -> None:
    log.info("=== Generating evaluation plots ===")
    plot_training_curves(result,       save_dir)
    plot_tsne(embeddings,              save_dir)
    plot_umap(embeddings,              save_dir)
    plot_cosine_similarity_boxplot(embeddings, save_dir)
    plot_embedding_norms(embeddings,   save_dir)
    plot_edge_counts(edges,            save_dir)
    plot_feature_dims(feature_tensors, save_dir)
    plot_node_counts(nodes,            save_dir)
    log.info("All plots saved to %s/", save_dir)