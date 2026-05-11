# """
# data/graph_builder.py
# ─────────────────────
# Converts raw node/edge dicts (from Neo4jLoader) into a
# torch_geometric.data.HeteroData object.

# After feature matrices are built (see features/), this module also
# attaches them and constructs edge_index tensors.
# """

# from __future__ import annotations

# import logging
# from typing import Any, Dict, List, Tuple

# import torch
# from torch_geometric.data import HeteroData

# log = logging.getLogger(__name__)


# # ── id-field lookup (matches Neo4j property name used as @Id) ────────────────
# NODE_ID_FIELDS: Dict[str, str] = {
#     "Student":    "student_id",
#     "Job":        "job_id",
#     "Company":    "company_id",
#     "Skill":      "skill_id",
#     "Occupation": "occupation_id",
#     "Project":    "project_id",
#     "Course":     "course_id",
#     "Diploma":    "diploma_id",
# }


# class GraphBuilder:
#     """Builds a PyG HeteroData graph from raw Neo4j data + feature tensors."""

#     def __init__(self, edges: Dict[str, List[Dict[str, Any]]]):
#         self.edges = edges

#     # ── public ────────────────────────────────────────────────────────────────

#     def build(
#         self,
#         feature_matrices: Dict[str, torch.Tensor],
#         nodes: Dict[str, List[Dict[str, Any]]],
#     ) -> HeteroData:
#         """
#         Parameters
#         ----------
#         feature_matrices : {node_type: FloatTensor [N, F]}
#         nodes : {node_type: [row_dict, ...]}
#             The raw node data, used to build the ID-to-index mapping. This
#             must be the same data used to generate `feature_matrices` to
#             ensure consistency.

#         Returns
#         -------
#         HeteroData with .x and edge_index for every type present.
#         """
#         data = HeteroData()

#         # Build per-type id -> integer index maps. This is done here, not in
#         # __init__, to guarantee the maps are built from the exact same `nodes`
#         # dict that was used to create the `feature_matrices`.
#         id_maps: Dict[str, Dict[str, int]] = {}
#         for ntype, rows in nodes.items():
#             id_field = NODE_ID_FIELDS[ntype]
#             if rows:
#                 id_maps[ntype] = {row[id_field]: idx for idx, row in enumerate(rows)}
#             else:
#                 id_maps[ntype] = {}
#         log.info("ID maps built for: %s", list(id_maps.keys()))

#         # Set up node stores, attaching features and explicitly setting num_nodes
#         for ntype, n_rows in nodes.items():
#             if ntype in feature_matrices:
#                 feat = feature_matrices[ntype]
#                 data[ntype].x = feat
#                 data[ntype].num_nodes = feat.size(0)
#                 log.info("%s  ->  x shape %s", ntype, tuple(feat.shape))
#             else:
#                 # This case should not happen with the current feature pipeline,
#                 # but as a safeguard, set num_nodes from the raw node list.
#                 data[ntype].num_nodes = len(n_rows)

#         # Attach edges
#         for rel_type, rows in self.edges.items():
#             if not rows:
#                 continue
#             src_type = rows[0]["src_type"]
#             dst_type = rows[0]["dst_type"]

#             # Skip if either endpoint has no nodes loaded
#             if not id_maps.get(src_type) or not id_maps.get(dst_type):
#                 log.warning(
#                     "Skipping relation '%s': node type '%s' or '%s' has no nodes or id_map.",
#                     rel_type, src_type, dst_type
#                 )
#                 continue

#             src_map = id_maps[src_type]
#             dst_map = id_maps[dst_type]

#             srcs, dsts = [], []
#             for r in rows:
#                 s = src_map.get(r["src"])
#                 d = dst_map.get(r["dst"])
#                 if s is not None and d is not None:
#                     srcs.append(s)
#                     dsts.append(d)

#             if not srcs:
#                 log.warning("No valid edges for %s", rel_type)
#                 continue

#             edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
#             triple = (src_type, rel_type, dst_type)
#             data[triple].edge_index = edge_index
#             log.info(
#                 "Edge %s -> %d edges [%s->%s]",
#                 rel_type, len(srcs), src_type, dst_type,
#             )

#             # Also add reverse edges for undirected message passing
#             rev_triple = (dst_type, f"rev_{rel_type}", src_type)
#             data[rev_triple].edge_index = edge_index.flip(0)

#         # Final check for graph validity. This will raise an error if any
#         # edge_index contains an out-of-bounds node index.
#         data.validate()

#         return data



# v1

"""
data/graph_builder.py
─────────────────────
Converts raw node/edge dicts (from Neo4jLoader) into a
torch_geometric.data.HeteroData object.

After feature matrices are built (see features/), this module also
attaches them and constructs edge_index tensors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch
from torch_geometric.data import HeteroData

log = logging.getLogger(__name__)


# ── id-field lookup (matches Neo4j property name used as @Id) ────────────────
NODE_ID_FIELDS: Dict[str, str] = {
    "Student":      "student_id",
    "Job":          "job_id",
    "Company":      "company_id",
    "Skill_L1":     "skill_id",
    "Skill_L2":     "skill_id",
    "Skill_L3":     "skill_id",
    "Occupation_L1":"occupation_id",
    "Occupation_L2":"occupation_id",
    "Occupation_L3":"occupation_id",
    "Project":      "project_id",
    "Course":       "course_id",
    "Diploma":      "diploma_id",
}


class GraphBuilder:
    """Builds a PyG HeteroData graph from raw Neo4j data + feature tensors."""

    def __init__(self, edges: Dict[str, List[Dict[str, Any]]]):
        self.edges = edges

    # ── public ────────────────────────────────────────────────────────────────

    def build(
        self,
        feature_matrices: Dict[str, torch.Tensor],
        nodes: Dict[str, List[Dict[str, Any]]],
    ) -> HeteroData:
        """
        Parameters
        ----------
        feature_matrices : {node_type: FloatTensor [N, F]}
        nodes : {node_type: [row_dict, ...]}
            The raw node data, used to build the ID-to-index mapping. This
            must be the same data used to generate `feature_matrices` to
            ensure consistency.

        Returns
        -------
        HeteroData with .x and edge_index for every type present.
        """
        data = HeteroData()

        # Build per-type id -> integer index maps. This is done here, not in
        # __init__, to guarantee the maps are built from the exact same `nodes`
        # dict that was used to create the `feature_matrices`.
        id_maps: Dict[str, Dict[str, int]] = {}
        for ntype, rows in nodes.items():
            id_field = NODE_ID_FIELDS[ntype]
            if rows:
                id_maps[ntype] = {row[id_field]: idx for idx, row in enumerate(rows)}
            else:
                id_maps[ntype] = {}
        log.info("ID maps built for: %s", list(id_maps.keys()))

        # Set up node stores, attaching features and explicitly setting num_nodes
        for ntype, n_rows in nodes.items():
            if ntype in feature_matrices:
                feat = feature_matrices[ntype]
                data[ntype].x = feat
                data[ntype].num_nodes = feat.size(0)
                log.info("%s  ->  x shape %s", ntype, tuple(feat.shape))
            else:
                # This case should not happen with the current feature pipeline,
                # but as a safeguard, set num_nodes from the raw node list.
                data[ntype].num_nodes = len(n_rows)

        # Attach edges
        for rel_type, rows in self.edges.items():
            if not rows:
                continue
            src_type = rows[0]["src_type"]
            dst_type = rows[0]["dst_type"]

            # Skip if either endpoint has no nodes loaded
            if not id_maps.get(src_type) or not id_maps.get(dst_type):
                log.warning(
                    "Skipping relation '%s': node type '%s' or '%s' has no nodes or id_map.",
                    rel_type, src_type, dst_type
                )
                continue

            src_map = id_maps[src_type]
            dst_map = id_maps[dst_type]

            srcs, dsts = [], []
            for r in rows:
                s = src_map.get(r["src"])
                d = dst_map.get(r["dst"])
                if s is not None and d is not None:
                    srcs.append(s)
                    dsts.append(d)

            if not srcs:
                log.warning("No valid edges for %s", rel_type)
                continue

            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
            triple = (src_type, rel_type, dst_type)
            data[triple].edge_index = edge_index
            log.info(
                "Edge %s -> %d edges [%s->%s]",
                rel_type, len(srcs), src_type, dst_type,
            )

            # Also add reverse edges for undirected message passing
            rev_triple = (dst_type, f"rev_{rel_type}", src_type)
            data[rev_triple].edge_index = edge_index.flip(0)

        # Final check for graph validity. This will raise an error if any
        # edge_index contains an out-of-bounds node index.
        data.validate()

        return data