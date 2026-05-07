# #!/usr/bin/env python3
# """
# main.py
# ────────
# Entry point for the Heterogeneous GraphSAGE pipeline.

# Usage
# ─────
#   # Full run (load Neo4j → embed → train → evaluate → write back)
#   python main.py

#   # Skip Neo4j loading (use cached feature tensors)
#   python main.py --use-cache

#   # Force re-embed even if cache exists
#   python main.py --force-rebuild

#   # Skip writing embeddings back to Neo4j
#   python main.py --no-write-back

#   # Dry-run (no Neo4j needed: generate synthetic data)
#   python main.py --dry-run
# """

# from __future__ import annotations

# import argparse
# import logging
# import os
# import sys
# import time

# import torch

# # ── ensure project root is on sys.path ───────────────────────────────────────
# sys.path.insert(0, os.path.dirname(__file__))

# from config.settings import (
#     GraphSAGEConfig, TrainingConfig,
#     CHECKPOINT_PATH, EMBEDDINGS_PATH, PLOTS_DIR,
#     feature_dim,
# )
# from data.neo4j_loader import Neo4jLoader
# from data.graph_builder import GraphBuilder, NODE_ID_FIELDS
# from features.pipeline import FeaturePipeline
# from features.embedder import OllamaEmbedder
# from models.graphsage import HeteroGraphSAGE
# from training.trainer import Trainer, split_edges
# from evaluation.visualizer import run_all_plots
# from utils.helpers import setup_logging, save_pickle, print_graph_summary


# # ── CLI ───────────────────────────────────────────────────────────────────────

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Heterogeneous GraphSAGE for Jobs Graph")
#     p.add_argument("--use-cache",      action="store_true", help="Use cached feature tensors")
#     p.add_argument("--force-rebuild",  action="store_true", help="Force re-embed texts")
#     p.add_argument("--no-write-back",  action="store_true", help="Skip writing embeddings to Neo4j")
#     p.add_argument("--dry-run",        action="store_true", help="Use synthetic data (no Neo4j)")
#     p.add_argument("--epochs",         type=int,   default=None)
#     p.add_argument("--lr",             type=float, default=None)
#     p.add_argument("--hidden",         type=int,   default=None)
#     p.add_argument("--out",            type=int,   default=None)
#     p.add_argument("--device",         type=str,   default=None)
#     return p.parse_args()


# # ── synthetic data for dry-run ────────────────────────────────────────────────

# def make_synthetic_data():
#     """Generates minimal in-memory data so the pipeline can be tested
#     without a live Neo4j instance."""
#     import random, string

#     def rand_id(prefix, n=8):
#         return prefix + "_" + "".join(random.choices(string.ascii_lowercase, k=n))

#     nodes = {
#         "Student": [
#             {"id": rand_id("s"), "student_id": rand_id("s"),
#              "name": f"Student {i}", "major": random.choice(["CS", "DS", "IT"]),
#              "graduation_year": random.randint(2023, 2026),
#              "current_year_of_study": random.randint(1, 4),
#              "degree_level": random.choice(["bachelor", "master"])}
#             for i in range(30)
#         ],
#         "Job": [
#             {"id": rand_id("j"), "job_id": rand_id("j"),
#              "title": f"Developer {i}", "description": f"Build software {i}",
#              "experience_level": random.choice(["junior", "mid", "senior"]),
#              "job_type": "full-time",
#              "salary": random.randint(30000, 120000),
#              "remote": random.choice([True, False]),
#              "location": "Remote"}
#             for i in range(20)
#         ],
#         "Company": [
#             {"id": rand_id("c"), "company_id": rand_id("c"),
#              "name": f"Company {i}", "industry": random.choice(["Tech", "Finance"]),
#              "location": "EU", "size": random.choice(["small", "medium", "large"])}
#             for i in range(10)
#         ],
#         "Skill": [
#             {"id": rand_id("sk"), "skill_id": rand_id("sk"),
#              "name": sk, "layer": 3, "type": "Specific Skill",
#              "parent": "Programming & Scripting",
#              "group_name": "Programming & Scripting",
#              "category_name": "Technical Competencies (Hard Skills)"}
#             for sk in ["Python", "Java", "SQL", "Docker", "React", "TypeScript", "Neo4j"]
#         ],
#         "Occupation": [
#             {"id": rand_id("o"), "occupation_id": rand_id("o"),
#              "name": occ, "layer": 3, "type": "Specialized Role",
#              "parent": "Software Engineer",
#              "parent_name": "Software Engineer",
#              "grandparent_name": "Engineering"}
#             for occ in ["Backend Developer", "Frontend Developer", "Data Engineer"]
#         ],
#         "Project": [
#             {"id": rand_id("p"), "project_id": rand_id("p"),
#              "title": f"Project {i}", "description": f"A cool project {i}"}
#             for i in range(15)
#         ],
#         "Course": [
#             {"id": rand_id("co"), "course_id": rand_id("co"),
#              "title": f"Course {i}", "description": f"Learn stuff {i}", "provider": "Coursera"}
#             for i in range(8)
#         ],
#         "Diploma": [
#             {"id": rand_id("d"), "diploma_id": rand_id("d"),
#              "title": f"BSc {i}", "description": f"Bachelor degree {i}", "issuer": "Uni"}
#             for i in range(5)
#         ],
#     }

#     # Fix id fields to match NODE_ID_FIELDS
#     for ntype, rows in nodes.items():
#         id_field = NODE_ID_FIELDS[ntype]
#         for r in rows:
#             if id_field not in r:
#                 r[id_field] = r["id"]

#     # Create simple edges
#     def sample_edges(src_type, dst_type, n=20):
#         src_ids = [r[NODE_ID_FIELDS[src_type]] for r in nodes[src_type]]
#         dst_ids = [r[NODE_ID_FIELDS[dst_type]] for r in nodes[dst_type]]
#         return [
#             {"src": random.choice(src_ids), "dst": random.choice(dst_ids),
#              "src_type": src_type, "dst_type": dst_type}
#             for _ in range(n)
#         ]

#     edges = {
#         "KNOWS":    sample_edges("Student", "Skill",   40),
#         "REQUIRES": sample_edges("Job",     "Skill",   30),
#         "POSTS":    sample_edges("Company", "Job",     15),
#         "CREATED":  sample_edges("Student", "Project", 20),
#         "BUILT_WITH": sample_edges("Project", "Skill", 25),
#         "COMPLETED":  sample_edges("Student", "Course",15),
#         "EARNED":     sample_edges("Student", "Diploma",10),
#         # "COVERS":     sample_edges("Course",  "Skill",  12),
#         "CERTIFIES":  sample_edges("Diploma", "Skill",   8),
#     }

#     return nodes, edges


# # ── main pipeline ─────────────────────────────────────────────────────────────

# def main():
#     os.makedirs("outputs/plots", exist_ok=True)
#     setup_logging()
#     log = logging.getLogger(__name__)
#     args = parse_args()

#     # Override config from CLI
#     gnn_cfg   = GraphSAGEConfig()
#     train_cfg = TrainingConfig()
#     if args.epochs:  train_cfg.epochs          = args.epochs
#     if args.lr:      train_cfg.lr              = args.lr
#     if args.hidden:  gnn_cfg.hidden_channels   = args.hidden
#     if args.out:     gnn_cfg.out_channels      = args.out
#     if args.device:  train_cfg.device          = args.device

#     torch.manual_seed(train_cfg.seed)

#     # ── Step 1: Load data ─────────────────────────────────────────────────────
#     if args.dry_run:
#         log.info("=== DRY-RUN: using synthetic data ===")
#         nodes, edges = make_synthetic_data()
#         loader = None
#     else:
#         log.info("=== Loading data from Neo4j ===")
#         loader = Neo4jLoader()
#         nodes  = loader.load_nodes()
#         edges  = loader.load_edges()

#     # ── Step 2: Build features ────────────────────────────────────────────────
#     log.info("=== Building feature tensors ===")
#     pipeline = FeaturePipeline(
#         embedder=OllamaEmbedder(),
#         use_cache=args.use_cache or not args.force_rebuild,
#     )
#     feature_tensors = pipeline.build(
#         nodes,
#         force_rebuild=args.force_rebuild,
#     )

#     # ── Step 3: Build graph ───────────────────────────────────────────────────
#     log.info("=== Building heterogeneous graph ===")
#     builder = GraphBuilder(edges)
#     data    = builder.build(feature_tensors, nodes)

#     print_graph_summary(nodes, edges, feature_tensors)

#     # ── Step 4: Build model ───────────────────────────────────────────────────
#     log.info("=== Constructing HeteroGraphSAGE ===")
#     in_channels_dict = {
#         ntype: feat.size(1)
#         for ntype, feat in feature_tensors.items()
#         if feat.size(0) > 0 and feat.dim() > 1
#     }
#     model = HeteroGraphSAGE(
#         in_channels_dict  = in_channels_dict,
#         hidden_channels   = gnn_cfg.hidden_channels,
#         out_channels      = gnn_cfg.out_channels,
#         num_layers        = gnn_cfg.num_layers,
#         dropout           = gnn_cfg.dropout,
#         metadata          = data.metadata(),
#     )
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     log.info("Model parameters: %d", n_params)

#     # ── Step 5: Split edges ───────────────────────────────────────────────────
#     log.info("=== Splitting edges ===")
#     train_data, val_pos, test_pos = split_edges(
#         data,
#         val_ratio  = train_cfg.val_ratio,
#         test_ratio = train_cfg.test_ratio,
#         seed       = train_cfg.seed,
#     )

#     # ── Step 6: Train ─────────────────────────────────────────────────────────
#     log.info("=== Training ===")
#     t0 = time.time()
#     trainer = Trainer(model, train_cfg, gnn_cfg)
#     result  = trainer.train(train_data, val_pos, test_pos)
#     elapsed = time.time() - t0
#     log.info("Training finished in %.1f s | Test AUC: %.4f", elapsed, result["test_auc"])

#     # ── Step 7: Extract embeddings ────────────────────────────────────────────
#     log.info("=== Extracting final embeddings ===")
#     embeddings = trainer.get_embeddings(data)
#     save_pickle(embeddings, EMBEDDINGS_PATH)
#     log.info("Embeddings saved -> %s", EMBEDDINGS_PATH)

#     # ── Step 8: Write back to Neo4j ───────────────────────────────────────────
#     if loader is not None and not args.no_write_back:
#         log.info("=== Writing embeddings to Neo4j ===")
#         for ntype, emb_tensor in embeddings.items():
#             if ntype not in nodes or not nodes[ntype]:
#                 continue
#             id_field = NODE_ID_FIELDS.get(ntype)
#             if id_field is None:
#                 continue
#             id_to_emb = {
#                 row[id_field]: emb_tensor[idx].tolist()
#                 for idx, row in enumerate(nodes[ntype])
#                 if idx < emb_tensor.size(0)
#             }
#             loader.write_embeddings(ntype, id_field, id_to_emb)

#     if loader:
#         loader.close()

#     # ── Step 9: Plots ─────────────────────────────────────────────────────────
#     log.info("=== Generating plots ===")
#     run_all_plots(
#         result          = result,
#         embeddings      = embeddings,
#         nodes           = nodes,
#         edges           = edges,
#         feature_tensors = feature_tensors,
#         save_dir        = PLOTS_DIR,
#     )

#     # ── Final summary ─────────────────────────────────────────────────────────
#     log.info("=" * 60)
#     log.info("PIPELINE COMPLETE")
#     log.info("  Test AUC       : %.4f", result["test_auc"])
#     log.info("  Best val AUC   : %.4f", result["best_val_auc"])
#     log.info("  Epochs run     : %d",   result["epochs_run"])
#     log.info("  Model saved    : %s",   CHECKPOINT_PATH)
#     log.info("  Embeddings     : %s",   EMBEDDINGS_PATH)
#     log.info("  Plots          : %s/",  PLOTS_DIR)
#     log.info("=" * 60)


# if __name__ == "__main__":
#     main()



# v1


#!/usr/bin/env python3
"""
main.py
────────
Entry point for the Heterogeneous GraphSAGE pipeline.

Usage
─────
  # Full run (load Neo4j → embed → train → evaluate → write back)
  python main.py

  # Skip Neo4j loading (use cached feature tensors)
  python main.py --use-cache

  # Force re-embed even if cache exists
  python main.py --force-rebuild

  # Skip writing embeddings back to Neo4j
  python main.py --no-write-back

  # Dry-run (no Neo4j needed: generate synthetic data)
  python main.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch

# ── ensure project root is on sys.path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import (
    GraphSAGEConfig, TrainingConfig,
    CHECKPOINT_PATH, EMBEDDINGS_PATH, PLOTS_DIR,
    feature_dim,
)
from data.neo4j_loader import Neo4jLoader
from data.graph_builder import GraphBuilder, NODE_ID_FIELDS
from features.pipeline import FeaturePipeline
from features.embedder import OllamaEmbedder
from models.graphsage import HeteroGraphSAGE
from training.trainer import Trainer, split_edges
from evaluation.visualizer import run_all_plots
from utils.helpers import setup_logging, save_pickle, print_graph_summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heterogeneous GraphSAGE for Jobs Graph")
    p.add_argument("--use-cache",      action="store_true", help="Use cached feature tensors")
    p.add_argument("--force-rebuild",  action="store_true", help="Force re-embed texts")
    p.add_argument("--no-write-back",  action="store_true", help="Skip writing embeddings to Neo4j")
    p.add_argument("--dry-run",        action="store_true", help="Use synthetic data (no Neo4j)")
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--hidden",         type=int,   default=None)
    p.add_argument("--out",            type=int,   default=None)
    p.add_argument("--device",         type=str,   default=None)
    return p.parse_args()


# ── synthetic data for dry-run ────────────────────────────────────────────────

def make_synthetic_data():
    """Generates minimal in-memory data so the pipeline can be tested
    without a live Neo4j instance."""
    import random, string

    def rand_id(prefix, n=8):
        return prefix + "_" + "".join(random.choices(string.ascii_lowercase, k=n))

    nodes = {
        "Student": [
            {"id": rand_id("s"), "student_id": rand_id("s"),
             "name": f"Student {i}", "major": random.choice(["CS", "DS", "IT"]),
             "graduation_year": random.randint(2023, 2026),
             "current_year_of_study": random.randint(1, 4),
             "degree_level": random.choice(["bachelor", "master"])}
            for i in range(30)
        ],
        "Job": [
            {"id": rand_id("j"), "job_id": rand_id("j"),
             "title": f"Developer {i}", "description": f"Build software {i}",
             "experience_level": random.choice(["junior", "mid", "senior"]),
             "job_type": "full-time",
             "salary": random.randint(30000, 120000),
             "remote": random.choice([True, False]),
             "location": "Remote"}
            for i in range(20)
        ],
        "Company": [
            {"id": rand_id("c"), "company_id": rand_id("c"),
             "name": f"Company {i}", "industry": random.choice(["Tech", "Finance"]),
             "location": "EU", "size": random.choice(["small", "medium", "large"])}
            for i in range(10)
        ],
        # Skill hierarchy: L1 (categories) → L2 (groups) → L3 (specific skills)
        "Skill_L1": [
            {"skill_id": f"skl1_{i}", "name": cat, "layer": 1, "type": "Category",
             "parent": None, "group_name": None, "category_name": None}
            for i, cat in enumerate(["Technical Competencies", "Soft Skills"])
        ],
        "Skill_L2": [
            {"skill_id": f"skl2_{i}", "name": grp, "layer": 2, "type": "Group",
             "parent": cat, "group_name": None, "category_name": cat}
            for i, (grp, cat) in enumerate([
                ("Programming & Scripting", "Technical Competencies"),
                ("Data & Databases",        "Technical Competencies"),
                ("Communication",           "Soft Skills"),
            ])
        ],
        "Skill_L3": [
            {"skill_id": rand_id("sk"), "name": sk, "layer": 3, "type": "Specific Skill",
             "parent": grp,
             "group_name": grp,
             "category_name": "Technical Competencies"}
            for sk, grp in [
                ("Python",     "Programming & Scripting"),
                ("Java",       "Programming & Scripting"),
                ("TypeScript", "Programming & Scripting"),
                ("SQL",        "Data & Databases"),
                ("Neo4j",      "Data & Databases"),
                ("Docker",     "Programming & Scripting"),
                ("React",      "Programming & Scripting"),
            ]
        ],
        # Occupation hierarchy: L1 (families) → L2 (groups) → L3 (specialized roles)
        "Occupation_L1": [
            {"occupation_id": f"ol1_{i}", "name": fam, "layer": 1, "type": "Family",
             "parent": None, "parent_name": None, "grandparent_name": None}
            for i, fam in enumerate(["Engineering", "Data & Analytics"])
        ],
        "Occupation_L2": [
            {"occupation_id": f"ol2_{i}", "name": grp, "layer": 2, "type": "Group",
             "parent": fam, "parent_name": None, "grandparent_name": fam}
            for i, (grp, fam) in enumerate([
                ("Software Engineering", "Engineering"),
                ("Data Engineering",     "Data & Analytics"),
            ])
        ],
        "Occupation_L3": [
            {"occupation_id": rand_id("o"), "name": occ, "layer": 3, "type": "Specialized Role",
             "parent": grp, "parent_name": grp, "grandparent_name": fam}
            for occ, grp, fam in [
                ("Backend Developer",  "Software Engineering", "Engineering"),
                ("Frontend Developer", "Software Engineering", "Engineering"),
                ("Data Engineer",      "Data Engineering",     "Data & Analytics"),
            ]
        ],
        "Project": [
            {"id": rand_id("p"), "project_id": rand_id("p"),
             "title": f"Project {i}", "description": f"A cool project {i}"}
            for i in range(15)
        ],
        "Course": [
            {"id": rand_id("co"), "course_id": rand_id("co"),
             "title": f"Course {i}", "description": f"Learn stuff {i}", "provider": "Coursera"}
            for i in range(8)
        ],
        "Diploma": [
            {"id": rand_id("d"), "diploma_id": rand_id("d"),
             "title": f"BSc {i}", "description": f"Bachelor degree {i}", "issuer": "Uni"}
            for i in range(5)
        ],
    }

    # Fix id fields to match NODE_ID_FIELDS
    for ntype, rows in nodes.items():
        id_field = NODE_ID_FIELDS[ntype]
        for r in rows:
            if id_field not in r:
                r[id_field] = r["id"]

    # Create simple edges
    def sample_edges(src_type, dst_type, n=20):
        src_ids = [r[NODE_ID_FIELDS[src_type]] for r in nodes[src_type]]
        dst_ids = [r[NODE_ID_FIELDS[dst_type]] for r in nodes[dst_type]]
        return [
            {"src": random.choice(src_ids), "dst": random.choice(dst_ids),
             "src_type": src_type, "dst_type": dst_type}
            for _ in range(n)
        ]

    edges = {
        "KNOWS":    sample_edges("Student",  "Skill_L3",   40),
        "REQUIRES": sample_edges("Job",      "Skill_L3",   30),
        "POSTS":    sample_edges("Company",  "Job",        15),
        "CREATED":  sample_edges("Student",  "Project",    20),
        "BUILT_WITH": sample_edges("Project","Skill_L3",   25),
        "COMPLETED":  sample_edges("Student","Course",     15),
        "EARNED":     sample_edges("Student","Diploma",    10),
        # "COVERS":   sample_edges("Course",  "Skill_L3",   12),
        "CERTIFIES":  sample_edges("Diploma","Skill_L3",    8),
        # Skill hierarchy edges
        "SKILL_SUBCLASS_L3_L2": [
            {"src": sk["skill_id"], "dst": f"skl2_{i % 2}",
             "src_type": "Skill_L3", "dst_type": "Skill_L2"}
            for i, sk in enumerate(nodes["Skill_L3"])
        ],
        "SKILL_SUBCLASS_L2_L1": [
            {"src": f"skl2_{i}", "dst": f"skl1_{i % 2}",
             "src_type": "Skill_L2", "dst_type": "Skill_L1"}
            for i in range(len(nodes["Skill_L2"]))
        ],
        # Occupation hierarchy edges
        "OCC_SUBCLASS_L3_L2": [
            {"src": o["occupation_id"], "dst": f"ol2_{i % 2}",
             "src_type": "Occupation_L3", "dst_type": "Occupation_L2"}
            for i, o in enumerate(nodes["Occupation_L3"])
        ],
        "OCC_SUBCLASS_L2_L1": [
            {"src": f"ol2_{i}", "dst": f"ol1_{i % 2}",
             "src_type": "Occupation_L2", "dst_type": "Occupation_L1"}
            for i in range(len(nodes["Occupation_L2"]))
        ],
    }

    return nodes, edges


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs/plots", exist_ok=True)
    setup_logging()
    log = logging.getLogger(__name__)
    args = parse_args()

    # Override config from CLI
    gnn_cfg   = GraphSAGEConfig()
    train_cfg = TrainingConfig()
    if args.epochs:  train_cfg.epochs          = args.epochs
    if args.lr:      train_cfg.lr              = args.lr
    if args.hidden:  gnn_cfg.hidden_channels   = args.hidden
    if args.out:     gnn_cfg.out_channels      = args.out
    if args.device:  train_cfg.device          = args.device

    torch.manual_seed(train_cfg.seed)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    if args.dry_run:
        log.info("=== DRY-RUN: using synthetic data ===")
        nodes, edges = make_synthetic_data()
        loader = None
    else:
        log.info("=== Loading data from Neo4j ===")
        loader = Neo4jLoader()
        nodes  = loader.load_nodes()
        edges  = loader.load_edges()

    # ── Step 2: Build features ────────────────────────────────────────────────
    log.info("=== Building feature tensors ===")
    pipeline = FeaturePipeline(
        embedder=OllamaEmbedder(),
        use_cache=args.use_cache or not args.force_rebuild,
    )
    feature_tensors = pipeline.build(
        nodes,
        force_rebuild=args.force_rebuild,
    )

    # ── Step 3: Build graph ───────────────────────────────────────────────────
    log.info("=== Building heterogeneous graph ===")
    builder = GraphBuilder(edges)
    data    = builder.build(feature_tensors, nodes)

    print_graph_summary(nodes, edges, feature_tensors)

    # ── Step 4: Build model ───────────────────────────────────────────────────
    log.info("=== Constructing HeteroGraphSAGE ===")
    in_channels_dict = {
        ntype: feat.size(1)
        for ntype, feat in feature_tensors.items()
        if feat.size(0) > 0 and feat.dim() > 1
    }
    model = HeteroGraphSAGE(
        in_channels_dict  = in_channels_dict,
        hidden_channels   = gnn_cfg.hidden_channels,
        out_channels      = gnn_cfg.out_channels,
        num_layers        = gnn_cfg.num_layers,
        dropout           = gnn_cfg.dropout,
        metadata          = data.metadata(),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d", n_params)

    # ── Step 5: Split edges ───────────────────────────────────────────────────
    log.info("=== Splitting edges ===")
    train_data, val_pos, test_pos = split_edges(
        data,
        val_ratio  = train_cfg.val_ratio,
        test_ratio = train_cfg.test_ratio,
        seed       = train_cfg.seed,
    )

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    log.info("=== Training ===")
    t0 = time.time()
    trainer = Trainer(model, train_cfg, gnn_cfg)
    result  = trainer.train(train_data, val_pos, test_pos)
    elapsed = time.time() - t0
    log.info("Training finished in %.1f s | Test AUC: %.4f", elapsed, result["test_auc"])

    # ── Step 7: Extract embeddings ────────────────────────────────────────────
    log.info("=== Extracting final embeddings ===")
    embeddings = trainer.get_embeddings(data)
    save_pickle(embeddings, EMBEDDINGS_PATH)
    log.info("Embeddings saved -> %s", EMBEDDINGS_PATH)

    # ── Step 8: Write back to Neo4j ───────────────────────────────────────────
    if loader is not None and not args.no_write_back:
        log.info("=== Writing embeddings to Neo4j ===")
        for ntype, emb_tensor in embeddings.items():
            if ntype not in nodes or not nodes[ntype]:
                continue
            id_field = NODE_ID_FIELDS.get(ntype)
            if id_field is None:
                continue
            id_to_emb = {
                row[id_field]: emb_tensor[idx].tolist()
                for idx, row in enumerate(nodes[ntype])
                if idx < emb_tensor.size(0)
            }
            loader.write_embeddings(ntype, id_field, id_to_emb)

    if loader:
        loader.close()

    # ── Step 9: Plots ─────────────────────────────────────────────────────────
    log.info("=== Generating plots ===")
    run_all_plots(
        result          = result,
        embeddings      = embeddings,
        nodes           = nodes,
        edges           = edges,
        feature_tensors = feature_tensors,
        save_dir        = PLOTS_DIR,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("  Test AUC       : %.4f", result["test_auc"])
    log.info("  Best val AUC   : %.4f", result["best_val_auc"])
    log.info("  Epochs run     : %d",   result["epochs_run"])
    log.info("  Model saved    : %s",   CHECKPOINT_PATH)
    log.info("  Embeddings     : %s",   EMBEDDINGS_PATH)
    log.info("  Plots          : %s/",  PLOTS_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()