# """
# Central configuration for the GraphSAGE training pipeline.
# """

# from dataclasses import dataclass, field
# from typing import List, Optional


# # ── Neo4j ────────────────────────────────────────────────────────────────────
# NEO4J_URI      = "bolt://localhost:7687"
# NEO4J_USER     = "neo4j"
# NEO4J_PASSWORD = "Test1234"

# # ── Ollama ───────────────────────────────────────────────────────────────────
# OLLAMA_BASE_URL   = "http://localhost:11434"
# OLLAMA_MODEL      = "nomic-embed-text"
# EMBEDDING_DIM     = 768          # nomic-embed-text output size

# # ── Feature dimensions ───────────────────────────────────────────────────────
# STRUCTURED_DIM: dict[str, int] = {
#     "Student":   3,   # graduation_year, current_year_of_study (normalised), degree_level_enc
#     "Job":       4,   # salary (norm), remote, experience_level_enc, job_type_enc
#     "Company":   1,   # size_enc
#     "Skill":     2,   # layer_norm, type_enc
#     "Occupation":2,   # layer_norm, type_enc
#     "Project":   0,
#     "Course":    0,
#     "Diploma":   0,
# }

# # Final feature dim per node type = EMBEDDING_DIM + STRUCTURED_DIM[type]
# def feature_dim(node_type: str) -> int:
#     return EMBEDDING_DIM + STRUCTURED_DIM.get(node_type, 0)


# # ── GraphSAGE ────────────────────────────────────────────────────────────────
# @dataclass
# class GraphSAGEConfig:
#     hidden_channels:   int   = 128
#     out_channels:      int   = 64
#     num_layers:        int   = 2
#     dropout:           float = 0.5
#     aggregator:        str   = "pool"       # mean | lstm | max
#     # Which node types to produce embeddings for
#     target_node_types: List[str] = field(default_factory=lambda: [
#         "Student", "Job", "Skill", "Occupation", "Company",
#         "Project", "Course", "Diploma",
#     ])


# # ── Training ─────────────────────────────────────────────────────────────────
# @dataclass
# class TrainingConfig:
#     epochs:        int   = 200
#     lr:            float = 1e-3
#     weight_decay:  float = 5e-3
#     val_ratio:     float = 0.15
#     test_ratio:    float = 0.15
#     batch_size:    int   = 32
#     num_neighbors: List[int] = field(default_factory=lambda: [10, 10])
#     patience:      int   = 40          # early-stopping
#     seed:          int   = 42
#     device:        str   = "cpu"       # "cuda" if available


# # ── Output ───────────────────────────────────────────────────────────────────
# OUTPUT_DIR        = "outputs"
# CHECKPOINT_PATH   = f"{OUTPUT_DIR}/best_model.pt"
# EMBEDDINGS_PATH   = f"{OUTPUT_DIR}/embeddings.pkl"
# PLOTS_DIR         = f"{OUTPUT_DIR}/plots"



# v1 - no mlp for structured features
# """
# Central configuration for the GraphSAGE training pipeline.
# """

# from dataclasses import dataclass, field
# from typing import List, Optional


# # ── Neo4j ────────────────────────────────────────────────────────────────────
# NEO4J_URI      = "bolt://localhost:7687"
# NEO4J_USER     = "neo4j"
# NEO4J_PASSWORD = "Test1234"

# # ── Ollama ───────────────────────────────────────────────────────────────────
# OLLAMA_BASE_URL   = "http://localhost:11434"
# OLLAMA_MODEL      = "nomic-embed-text"
# EMBEDDING_DIM     = 768          # nomic-embed-text output size

# # ── Feature dimensions ───────────────────────────────────────────────────────
# STRUCTURED_DIM: dict[str, int] = {
#     "Student":      3,   # graduation_year, current_year_of_study (normalised), degree_level_enc
#     "Job":          4,   # salary (norm), remote, experience_level_enc, job_type_enc
#     "Company":      1,   # size_enc
#     "Skill_L1":     2,   # layer_norm (=1/3), type_enc
#     "Skill_L2":     2,   # layer_norm (=2/3), type_enc
#     "Skill_L3":     2,   # layer_norm (=3/3), type_enc
#     "Occupation_L1":2,   # layer_norm (=1/3), type_enc
#     "Occupation_L2":2,   # layer_norm (=2/3), type_enc
#     "Occupation_L3":2,   # layer_norm (=3/3), type_enc
#     "Project":      0,
#     "Course":       0,
#     "Diploma":      0,
# }

# # Final feature dim per node type = EMBEDDING_DIM + STRUCTURED_DIM[type]
# def feature_dim(node_type: str) -> int:
#     return EMBEDDING_DIM + STRUCTURED_DIM.get(node_type, 0)


# # ── GraphSAGE ────────────────────────────────────────────────────────────────
# @dataclass
# class GraphSAGEConfig:
#     hidden_channels:   int   = 64
#     out_channels:      int   = 32
#     num_layers:        int   = 2
#     dropout:           float = 0.5
#     aggregator:        str   = "mean"       # mean | lstm | max
#     # Which node types to produce embeddings for
#     target_node_types: List[str] = field(default_factory=lambda: [
#         "Student", "Job", "Company",
#         "Skill_L1", "Skill_L2", "Skill_L3",
#         "Occupation_L1", "Occupation_L2", "Occupation_L3",
#         "Project", "Course", "Diploma",
#     ])


# # ── Training ─────────────────────────────────────────────────────────────────
# @dataclass
# class TrainingConfig:
#     epochs:        int   = 500
#     lr:            float = 1e-3
#     weight_decay:  float = 5e-3
#     val_ratio:     float = 0.15
#     test_ratio:    float = 0.15
#     batch_size:    int   = 32
#     num_neighbors: List[int] = field(default_factory=lambda: [10, 10])
#     patience:      int   = 40          # early-stopping
#     seed:          int   = 42
#     device:        str   = "cpu"       # "cuda" if available


# # ── Output ───────────────────────────────────────────────────────────────────
# OUTPUT_DIR        = "outputs"
# CHECKPOINT_PATH   = f"{OUTPUT_DIR}/best_model.pt"
# EMBEDDINGS_PATH   = f"{OUTPUT_DIR}/embeddings.pkl"
# PLOTS_DIR         = f"{OUTPUT_DIR}/plots"



"""
Central configuration for the GraphSAGE training pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ── Neo4j ────────────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "Test1234"

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_MODEL      = "qwen3-embedding:8b"
RAW_EMBEDDING_DIM = 256          # nomic-embed-text output size
EMBEDDING_DIM     = 256        # Dimension used in the model (after truncation)

# ── Feature dimensions ───────────────────────────────────────────────────────
STRUCTURED_DIM: dict[str, int] = {
    "Student":      3,   # graduation_year, current_year_of_study (normalised), degree_level_enc
    "Job":          4,   # salary (norm), remote, experience_level_enc, job_type_enc
    # "Company":      1,   # size_enc
    "Skill_L1":     2,   # layer_norm (=1/3), type_enc
    "Skill_L2":     2,   # layer_norm (=2/3), type_enc
    "Skill_L3":     2,   # layer_norm (=3/3), type_enc
    "Occupation_L1":2,   # layer_norm (=1/3), type_enc
    "Occupation_L2":2,   # layer_norm (=2/3), type_enc
    "Occupation_L3":2,   # layer_norm (=3/3), type_enc
    "Project":      0,
    "Course":       0,
    "Diploma":      0,
}

# Final feature dim per node type = EMBEDDING_DIM + STRUCTURED_DIM[type]
# (before the structured projection layer; used by the feature pipeline)
def feature_dim(node_type: str) -> int:
    return EMBEDDING_DIM + STRUCTURED_DIM.get(node_type, 0)

# Feature dim as seen by the first GNN layer = EMBEDDING_DIM + structured_proj_dim
# (for types with zero structured features the raw dim is unchanged)
def projected_feature_dim(node_type: str, structured_proj_dim: int) -> int:
    sdim = STRUCTURED_DIM.get(node_type, 0)
    if sdim == 0:
        return EMBEDDING_DIM
    return EMBEDDING_DIM + structured_proj_dim


# ── GraphSAGE ────────────────────────────────────────────────────────────────
@dataclass
class GraphSAGEConfig:
    hidden_channels:     int   = 64
    out_channels:        int   = 32
    num_layers:          int   = 2
    dropout:             float = 0.5
    aggregator:          str   = "mean"       # mean | lstm | max
    # Structured features are projected from their raw dim to this size
    # before being concatenated with the semantic embedding.
    # Set to 0 to disable projection and concatenate raw structured features.
    structured_proj_dim: int   = 16
    # Which node types to produce embeddings for
    target_node_types: List[str] = field(default_factory=lambda: [
        "Student", "Job", 
        # "Company",
        "Skill_L1", "Skill_L2", 
        "Skill_L3",
        "Occupation_L1", "Occupation_L2", 
        "Occupation_L3",
        "Project", "Course", "Diploma",
    ])


# ── Training ─────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    epochs:        int   = 200
    lr:            float = 5e-4
    weight_decay:  float = 5e-3
    val_ratio:     float = 0.15
    test_ratio:    float = 0.15
    batch_size:    int   = 128
    num_neighbors: List[int] = field(default_factory=lambda: [10, 5])
    patience:      int   = 30          # early-stopping
    seed:          int   = 42
    device:        str   = "cpu"       # "cuda" if available


# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR        = "outputs"
CHECKPOINT_PATH   = f"{OUTPUT_DIR}/best_model.pt"
EMBEDDINGS_PATH   = f"{OUTPUT_DIR}/embeddings.pkl"
PLOTS_DIR         = f"{OUTPUT_DIR}/plots"