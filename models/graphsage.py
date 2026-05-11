# """
# models/graphsage.py
# ────────────────────
# Heterogeneous GraphSAGE using PyG's HeteroConv.
# """

# from __future__ import annotations
# from typing import Dict, Tuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv, HeteroConv


# class NodeEncoder(nn.Module):
#     def __init__(self, in_channels_dict: Dict[str, int], hidden_channels: int):
#         super().__init__()
#         self.projections = nn.ModuleDict({
#             ntype: nn.Sequential(
#                 nn.Linear(in_c, hidden_channels, bias=True),
#                 nn.LayerNorm(hidden_channels),
#             )
#             for ntype, in_c in in_channels_dict.items()
#             if in_c > 0
#         })

#     def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         return {
#             ntype: self.projections[ntype](x) if ntype in self.projections else x
#             for ntype, x in x_dict.items()
#         }


# class HeteroGraphSAGE(nn.Module):
#     """
#     Parameters
#     ----------
#     in_channels_dict : {node_type: feature_dim}
#     hidden_channels  : internal width (used for ALL SAGEConv layers)
#     out_channels     : final embedding dimension
#     num_layers       : number of message-passing layers
#     dropout          : dropout probability
#     metadata         : HeteroData.metadata() -> (node_types, edge_types)
#     """

#     def __init__(
#         self,
#         in_channels_dict: Dict[str, int],
#         hidden_channels:  int,
#         out_channels:     int,
#         num_layers:       int,
#         dropout:          float,
#         metadata: Tuple,
#     ):
#         super().__init__()
#         _, edge_types = metadata

#         self.encoder = NodeEncoder(in_channels_dict, hidden_channels)
#         self._node_types = list(in_channels_dict.keys())
#         self.dropout = dropout
#         self.num_layers = num_layers
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels

#         # All intermediate layers use hidden_channels -> hidden_channels.
#         # A final projection head maps hidden -> out.
#         self.layers: nn.ModuleList = nn.ModuleList()
#         self.norms:  nn.ModuleList = nn.ModuleList()

#         for _ in range(num_layers):
#             conv = HeteroConv(
#                 {et: SAGEConv(hidden_channels, hidden_channels, aggr="mean")
#                  for et in edge_types},
#                 aggr="sum",
#             )
#             self.layers.append(conv)
#             self.norms.append(nn.ModuleDict({
#                 ntype: nn.LayerNorm(hidden_channels) for ntype in self._node_types
#             }))

#         # Final projection to output dimension
#         self.head = nn.ModuleDict({
#             ntype: nn.Linear(hidden_channels, out_channels)
#             for ntype in self._node_types
#         })

#     def forward(
#         self,
#         x_dict: Dict[str, torch.Tensor],
#         edge_index_dict: Dict,
#     ) -> Dict[str, torch.Tensor]:
#         h = self.encoder(x_dict)

#         for i, layer in enumerate(self.layers):
#             h_new = layer(h, edge_index_dict)
#             norm_dict = self.norms[i]

#             for ntype in h:
#                 if ntype not in h_new or h_new[ntype] is None:
#                     h_new[ntype] = h[ntype]  # keep previous if no messages

#             for ntype in h_new:
#                 if ntype in norm_dict:
#                     h_new[ntype] = norm_dict[ntype](h_new[ntype])
#                 h_new[ntype] = F.relu(h_new[ntype])
#                 h_new[ntype] = F.dropout(
#                     h_new[ntype], p=self.dropout, training=self.training
#                 )
#             h = h_new

#         # Project to out_channels and L2-normalise
#         out = {}
#         for ntype, h_n in h.items():
#             if ntype in self.head:
#                 out[ntype] = F.normalize(self.head[ntype](h_n), p=2, dim=-1)
#             else:
#                 out[ntype] = F.normalize(h_n, p=2, dim=-1)
#         return out

#     def decode(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
#         return (z_src * z_dst).sum(dim=-1)




# v1

"""
models/graphsage.py
───────────────────
Heterogeneous GraphSAGE with a per-node-type structured-feature projection.

Architecture
────────────
For each node type that has structured features (STRUCTURED_DIM > 0):

    x  =  [ semantic (768-d) | structured (s-d) ]   ← raw node feature tensor

    x_sem  = x[:, :EMBEDDING_DIM]                    ← semantic slice
    x_str  = x[:, EMBEDDING_DIM:]                    ← structured slice

    x_str_proj = ReLU( Linear(s -> structured_proj_dim)(x_str) )

    x_in   = cat([x_sem, x_str_proj], dim=-1)        ← (768 + proj_dim)-d

For node types with no structured features (STRUCTURED_DIM == 0):
    x_in = x  (768-d, unchanged)

x_in is then passed through `num_layers` SAGEConv layers in the standard way.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData

from config.settings import EMBEDDING_DIM, STRUCTURED_DIM

log = logging.getLogger(__name__)


class StructuredProjector(nn.Module):
    """
    Projects the structured slice of a node's feature vector from its raw
    dimension (s) to `proj_dim` using a single Linear + ReLU + LayerNorm.

    Parameters
    ----------
    in_dim   : raw structured feature size (≥ 1)
    proj_dim : target projection dimension
    """

    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(F.relu(self.proj(x)))


class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE.

    Parameters
    ----------
    in_channels_dict     : {node_type: raw_feature_dim}  (768 + structured_dim)
    hidden_channels      : SAGEConv hidden width
    out_channels         : final embedding dimension
    num_layers           : number of SAGEConv layers
    dropout              : dropout probability
    metadata             : HeteroData.metadata()  -> (node_types, edge_types)
    structured_proj_dim  : project structured slice to this width before concat.
                           Pass 0 to skip projection entirely.
    """

    def __init__(
        self,
        in_channels_dict:    Dict[str, int],
        hidden_channels:     int,
        out_channels:        int,
        num_layers:          int,
        dropout:             float,
        metadata:            Tuple,
        structured_proj_dim: int = 128,
    ):
        super().__init__()

        self.embedding_dim      = EMBEDDING_DIM
        self.structured_proj_dim = structured_proj_dim
        self.dropout             = dropout
        self.num_layers          = num_layers

        # ── Structured feature projectors (one per type that has them) ────────
        # Keyed by node type; absent if structured_dim == 0 or proj disabled.
        self.struct_projectors = nn.ModuleDict()

        # After optional projection the effective input dim per type:
        effective_in: Dict[str, int] = {}

        for ntype, raw_dim in in_channels_dict.items():
            sdim = STRUCTURED_DIM.get(ntype, 0)
            if sdim > 0 and structured_proj_dim > 0:
                self.struct_projectors[ntype] = StructuredProjector(sdim, structured_proj_dim)
                effective_in[ntype] = EMBEDDING_DIM + structured_proj_dim
                log.info(
                    "Structured projector for %-14s: %d -> %d  "
                    "(effective input: %d)",
                    ntype, sdim, structured_proj_dim, effective_in[ntype],
                )
            else:
                effective_in[ntype] = raw_dim   # no projection

        # ── SAGEConv layers ───────────────────────────────────────────────────
        node_types, edge_types = metadata

        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            # The input to the first SAGEConv layer is `hidden_channels` because
            # of the `self.input_lins` projections. Subsequent layers also take
            # `hidden_channels` as input.
            in_ch  = hidden_channels
            out_ch = out_channels if layer_idx == num_layers - 1 else hidden_channels

            conv = HeteroConv(
                {
                    etype: SAGEConv(in_ch, out_ch, aggr="mean")
                    for etype in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        # Input linear projections: effective_in[ntype] -> hidden_channels
        # (needed for the first layer when not using lazy init, and to unify
        #  the varied effective_in dimensions into a common hidden width)
        self.input_lins = nn.ModuleDict({
            ntype: nn.Linear(eff_dim, hidden_channels)
            for ntype, eff_dim in effective_in.items()
        })

    # ── helpers ───────────────────────────────────────────────────────────────

    def _project_features(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        For each node type: split x into semantic / structured slices,
        project the structured part, re-concatenate, and pass through the
        input linear to get a uniform hidden_channels-wide representation.
        """
        out: Dict[str, Tensor] = {}
        for ntype, x in x_dict.items():
            if ntype in self.struct_projectors:
                x_sem  = x[:, :self.embedding_dim]
                x_str  = x[:, self.embedding_dim:]
                x_str_proj = self.struct_projectors[ntype](x_str)
                x_combined = torch.cat([x_sem, x_str_proj], dim=-1)
            else:
                x_combined = x

            if ntype in self.input_lins:
                out[ntype] = F.relu(self.input_lins[ntype](x_combined))
            else:
                out[ntype] = x_combined

        return out

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_dict:        Dict[str, Tensor],
        edge_index_dict: Dict[Tuple, Tensor],
    ) -> Dict[str, Tensor]:
        # Step 1: project structured features + input linear
        h = self._project_features(x_dict)

        # Step 2: SAGEConv message passing
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            if i < self.num_layers - 1:
                h = {k: F.relu(v) for k, v in h.items()}
                h = {k: F.dropout(v, p=self.dropout, training=self.training)
                     for k, v in h.items()}

        return h

    def decode(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """Computes the dot product for link prediction."""
        return (z_src * z_dst).sum(dim=-1)