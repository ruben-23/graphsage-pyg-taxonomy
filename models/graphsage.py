"""
models/graphsage.py
────────────────────
Heterogeneous GraphSAGE using PyG's HeteroConv.
"""

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv


class NodeEncoder(nn.Module):
    def __init__(self, in_channels_dict: Dict[str, int], hidden_channels: int):
        super().__init__()
        self.projections = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(in_c, hidden_channels, bias=True),
                nn.LayerNorm(hidden_channels),
            )
            for ntype, in_c in in_channels_dict.items()
            if in_c > 0
        })

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            ntype: self.projections[ntype](x) if ntype in self.projections else x
            for ntype, x in x_dict.items()
        }


class HeteroGraphSAGE(nn.Module):
    """
    Parameters
    ----------
    in_channels_dict : {node_type: feature_dim}
    hidden_channels  : internal width (used for ALL SAGEConv layers)
    out_channels     : final embedding dimension
    num_layers       : number of message-passing layers
    dropout          : dropout probability
    metadata         : HeteroData.metadata() -> (node_types, edge_types)
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels:  int,
        out_channels:     int,
        num_layers:       int,
        dropout:          float,
        metadata: Tuple,
    ):
        super().__init__()
        _, edge_types = metadata

        self.encoder = NodeEncoder(in_channels_dict, hidden_channels)
        self._node_types = list(in_channels_dict.keys())
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # All intermediate layers use hidden_channels -> hidden_channels.
        # A final projection head maps hidden -> out.
        self.layers: nn.ModuleList = nn.ModuleList()
        self.norms:  nn.ModuleList = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {et: SAGEConv(hidden_channels, hidden_channels, aggr="mean")
                 for et in edge_types},
                aggr="sum",
            )
            self.layers.append(conv)
            self.norms.append(nn.ModuleDict({
                ntype: nn.LayerNorm(hidden_channels) for ntype in self._node_types
            }))

        # Final projection to output dimension
        self.head = nn.ModuleDict({
            ntype: nn.Linear(hidden_channels, out_channels)
            for ntype in self._node_types
        })

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, torch.Tensor]:
        h = self.encoder(x_dict)

        for i, layer in enumerate(self.layers):
            h_new = layer(h, edge_index_dict)
            norm_dict = self.norms[i]

            for ntype in h:
                if ntype not in h_new or h_new[ntype] is None:
                    h_new[ntype] = h[ntype]  # keep previous if no messages

            for ntype in h_new:
                if ntype in norm_dict:
                    h_new[ntype] = norm_dict[ntype](h_new[ntype])
                h_new[ntype] = F.relu(h_new[ntype])
                h_new[ntype] = F.dropout(
                    h_new[ntype], p=self.dropout, training=self.training
                )
            h = h_new

        # Project to out_channels and L2-normalise
        out = {}
        for ntype, h_n in h.items():
            if ntype in self.head:
                out[ntype] = F.normalize(self.head[ntype](h_n), p=2, dim=-1)
            else:
                out[ntype] = F.normalize(h_n, p=2, dim=-1)
        return out

    def decode(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return (z_src * z_dst).sum(dim=-1)