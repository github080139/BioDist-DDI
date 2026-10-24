import torch
import torch.nn as nn

from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
)


class MolecularGINEEncoder(nn.Module):

    def __init__(
        self,
        atom_dim=36,
        bond_dim=12,
        hidden_dim=256,
        num_layers=3,
        dropout=0.3,
    ):
        super().__init__()

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.ReLU(),
        )

        self.bond_encoder = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):

            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(
                GINEConv(
                    nn=mlp,
                    edge_dim=hidden_dim,
                )
            )

            self.norms.append(
                nn.LayerNorm(hidden_dim)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
    ):

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        for conv, norm in zip(
            self.convs,
            self.norms,
        ):

            residual = x

            x = conv(
                x,
                edge_index,
                edge_attr,
            )

            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

            x = x + residual

        z = global_mean_pool(
            x,
            batch,
        )

        return z


class StructureOnlyDDI(nn.Module):

    def __init__(
        self,
        atom_dim=36,
        bond_dim=12,
        hidden_dim=256,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = MolecularGINEEncoder(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim * 3,
                256,
            ),

            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                128,
                1,
            ),
        )

    def encode_graph_batch(
        self,
        batch,
    ):
        return self.encoder(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

    def score_pair_embeddings(
        self,
        za,
        zb,
    ):

        pair = torch.cat(
            [
                za,
                zb,
                torch.abs(za - zb),
            ],
            dim=-1,
        )

        return (
            self.classifier(pair)
            .squeeze(-1)
        )

    def forward(
        self,
        batch_a,
        batch_b,
    ):

        za = self.encode_graph_batch(
            batch_a
        )

        zb = self.encode_graph_batch(
            batch_b
        )

        return self.score_pair_embeddings(
            za,
            zb,
        )
