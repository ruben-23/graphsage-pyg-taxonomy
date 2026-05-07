# Heterogeneous GraphSAGE — Jobs Knowledge Graph

A modular, production-ready pipeline that trains a **Heterogeneous GraphSAGE** model
on a Neo4j knowledge graph of students, jobs, companies, skills, occupations,
courses, diplomas, and projects.

---

## Architecture

```
graphsage_jobs/
├── config/
│   └── settings.py          ← all hyper-parameters & Neo4j / Ollama config
├── data/
│   ├── neo4j_loader.py      ← loads nodes & relationships from Neo4j
│   └── graph_builder.py     ← converts raw dicts → PyG HeteroData
├── features/
│   ├── semantic.py          ← builds text sentences per node type
│   ├── embedder.py          ← calls Ollama nomic-embed-text
│   ├── structured.py        ← normalises numeric / categorical features
│   └── pipeline.py          ← orchestrates semantic + structured → tensor
├── models/
│   └── graphsage.py         ← HeteroGraphSAGE (to_hetero wrapping SAGEConv)
├── training/
│   └── trainer.py           ← self-supervised link-prediction trainer
├── evaluation/
│   └── visualizer.py        ← 9 diagnostic plots
├── utils/
│   └── helpers.py           ← logging, pickling, summary printer
├── main.py                  ← full pipeline entry point
└── requirements.txt
```

---

## Feature Engineering

### Two-bucket strategy

| Bucket | Content | Method | Output |
|--------|---------|--------|--------|
| **Semantic** | Titles, descriptions, majors, taxonomy paths | Ollama `nomic-embed-text` | 768-d vector |
| **Structured** | Salary, graduation year, remote flag, etc. | MinMax / ordinal encoding | N-d vector |

These are concatenated: `[768 ‖ N]` → final input feature vector.

### Taxonomy paths

`Skill` and `Occupation` nodes use **only layer-3 leaves**.
Their semantic text is the full taxonomy path:

```
Technical Competencies (Hard Skills) -> Programming & Scripting -> Python
Engineering -> Software Engineer -> Backend Developer
```

---

## Node types & feature dims

| Node Type  | Semantic text source                    | Structured features              | Total dim |
|------------|-----------------------------------------|----------------------------------|-----------|
| Student    | name, major, degree_level               | grad_year, year_of_study, degree | 768 + 3   |
| Job        | title, description, exp_level, location | salary, remote, exp_enc, type    | 768 + 4   |
| Company    | name, industry, location, size          | size_enc                         | 768 + 1   |
| Skill      | category → group → name                 | layer_norm, type_enc             | 768 + 2   |
| Occupation | family → core → name                    | layer_norm, type_enc             | 768 + 2   |
| Project    | title, description                      | –                                | 768       |
| Course     | title, description, provider            | –                                | 768       |
| Diploma    | title, description, issuer              | –                                | 768       |

---

## Training

- **Objective**: Self-supervised link prediction (binary cross-entropy)
- **Negative sampling**: random node pairs per relation type
- **Split**: 70 / 15 / 15 (train / val / test) on edges
- **Early stopping**: patience = 15 epochs on val AUC
- **Optimiser**: Adam + ReduceLROnPlateau

---

## Output plots (saved to `outputs/plots/`)

| File | Content |
|------|---------|
| `training_curves.png` | Train/val loss + val/test AUC over epochs |
| `tsne_embeddings.png` | t-SNE of all embeddings coloured by node type |
| `umap_embeddings.png` | UMAP projection (requires `umap-learn`) |
| `cosine_similarity_boxplot.png` | Intra-type cosine similarity distributions |
| `embedding_norms.png` | ‖embedding‖₂ histograms per type |
| `edge_counts.png` | Bar chart of edges per relation type |
| `feature_dims.png` | Input feature dimension per node type |
| `node_counts.png` | Node count per type |

---

## Quick start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
# Optional: pip install umap-learn
```

### 2 — Configure

Edit `config/settings.py`:

```python
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"
OLLAMA_BASE_URL = "http://localhost:11434"
```

### 3 — Run (full)

```bash
cd graphsage_jobs
python main.py
```

### 4 — Dry run (no Neo4j / Ollama needed)

```bash
python main.py --dry-run
```
Generates synthetic data so you can verify the pipeline end-to-end.

### 5 — Re-use cached embeddings

```bash
python main.py --use-cache          # skip Ollama calls, load from disk
python main.py --force-rebuild      # force re-embed even if cache exists
```

### 6 — Skip writing back to Neo4j

```bash
python main.py --no-write-back
```

### 7 — Override hyper-parameters

```bash
python main.py --epochs 200 --lr 5e-4 --hidden 512 --out 256 --device cuda
```

---

## Neo4j write-back

After training, each node gets a `graphsage_embedding` property (Float array):

```cypher
MATCH (s:Student)
RETURN s.student_id, s.graphsage_embedding
LIMIT 5
```

---

## Extending the model

- **New relation type**: add a Cypher query to `_REL_QUERIES` in `neo4j_loader.py`.
- **New node type**: add node query to `_NODE_QUERIES`, a text builder to `semantic.py`,
  and structured feature builder to `structured.py`.
- **Supervised task** (e.g., student-job matching): attach a task head after
  the GraphSAGE encoder and add a supervised loss to `trainer.py`.