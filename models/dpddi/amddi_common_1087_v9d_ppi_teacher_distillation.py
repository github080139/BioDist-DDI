#!/usr/bin/env python3

"""
AM-DDI Common-1087 — V9-D

V9-A-compatible privileged biological distillation.

DEPLOYABLE DDI PATH
-------------------
Identical architecture to V9-A:

    molecular graph -> GINE-256 -> graph pair classifier
    Morgan-SVD128               -> Morgan pair classifier

    DDI logit =
        graph_logit + beta * Morgan_logit

Biological information is NEVER an inference input.

TRAINING-ONLY BIOLOGICAL TRANSFER
---------------------------------
Frozen biological teacher:

    base =
        clean DTI-target aggregated ESM2

    PPI residual =
        PPI-refined ESM2 - base ESM2

    [base, residual]
        -> frozen biological teacher
        -> 128-D teacher representation

Student biological predictor:

    GINE graph embedding 256
        -> predictor
        -> 128-D predicted biological representation

Auxiliary loss:
    cosine(student_predicted_biology,
           frozen_teacher_biology)

Only inner-training drugs participate in this auxiliary loss.

Validation/inference uses only the V9-A DDI path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.dpddi.amddi_common_1087_structure_gine import (
    MolecularGINEEncoder,
)


GRAPH_DIM = 256
MORGAN_DIM = 128
BIO_DIM = 128

VALID_BETAS = (
    0.05,
    0.10,
    0.25,
)

VALID_LAMBDAS = (
    0.00,
    0.01,
    0.03,
    0.05,
    0.10,
)


class PairMLP(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                3 * embedding_dim,
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

    def forward(
        self,
        za: Tensor,
        zb: Tensor,
    ) -> Tensor:

        pair = torch.cat(
            [
                za,
                zb,
                torch.abs(
                    za - zb
                ),
            ],
            dim=-1,
        )

        return self.net(
            pair
        ).squeeze(-1)


class V9DStudent(nn.Module):
    """
    Deployable model.

    DDI prediction path is architecturally equivalent to V9-A.
    """

    def __init__(
        self,
        beta: float = 0.05,
        atom_dim: int = 36,
        bond_dim: int = 12,
        graph_hidden_dim: int = GRAPH_DIM,
        morgan_dim: int = MORGAN_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()

        beta = float(beta)

        if beta not in VALID_BETAS:
            raise ValueError(
                f"Invalid beta={beta}; "
                f"allowed={VALID_BETAS}"
            )

        if graph_hidden_dim != GRAPH_DIM:
            raise ValueError(
                f"graph_hidden_dim must be {GRAPH_DIM}"
            )

        if morgan_dim != MORGAN_DIM:
            raise ValueError(
                f"morgan_dim must be {MORGAN_DIM}"
            )

        self.beta = beta

        # ------------------------------------------------------
        # Exact V9-A deployable chemistry path
        # ------------------------------------------------------

        self.graph_encoder = MolecularGINEEncoder(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=3,
            dropout=dropout,
        )

        self.graph_classifier = PairMLP(
            embedding_dim=graph_hidden_dim,
            dropout=dropout,
        )

        self.morgan_classifier = PairMLP(
            embedding_dim=morgan_dim,
            dropout=dropout,
        )

        # ------------------------------------------------------
        # Training-time biological prediction head.
        #
        # This head is NOT used by DDI validation/inference.
        # ------------------------------------------------------

        self.bio_predictor = nn.Sequential(
            nn.Linear(
                GRAPH_DIM,
                256,
            ),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                256,
                BIO_DIM,
            ),
        )

    def encode_graph_batch(
        self,
        batch,
    ) -> Tensor:

        return self.graph_encoder(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

    def score_graph_embeddings(
        self,
        za,
        zb,
    ):

        return self.graph_classifier(
            za,
            zb,
        )

    def score_morgan_embeddings(
        self,
        ma,
        mb,
    ):

        return self.morgan_classifier(
            ma,
            mb,
        )

    def fuse_logits(
        self,
        graph_logit,
        morgan_logit,
    ):

        return (
            graph_logit
            +
            self.beta
            *
            morgan_logit
        )

    def predict_biology(
        self,
        graph_z,
    ):

        return self.bio_predictor(
            graph_z
        )


class V9DBiologicalTeacher(nn.Module):
    """
    Training-only biological teacher.

    Input 1:
        clean DTI-target aggregated ESM2.

    Input 2:
        PPI-refined ESM2 context.

    Internally:
        residual = PPI context - base context.
    """

    def __init__(
        self,
        input_dim=128,
        bio_dim=BIO_DIM,
        dropout=0.3,
    ):
        super().__init__()

        self.base_encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                bio_dim,
            ),
            nn.LayerNorm(
                bio_dim
            ),
            nn.ReLU(),
        )

        self.ppi_residual_encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                bio_dim,
            ),
            nn.LayerNorm(
                bio_dim
            ),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                bio_dim * 2,
                256,
            ),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                256,
                bio_dim,
            ),
        )

    def forward(
        self,
        base,
        ppi,
    ):

        base = base.float()
        ppi = ppi.float()

        if base.shape != ppi.shape:
            raise ValueError(
                "base/PPI shape mismatch"
            )

        residual = (
            ppi - base
        )

        hb = self.base_encoder(
            base
        )

        hr = self.ppi_residual_encoder(
            residual
        )

        return self.fusion(
            torch.cat(
                [
                    hb,
                    hr,
                ],
                dim=-1,
            )
        )


def freeze_teacher(
    teacher,
):

    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad_(False)

    return teacher


def biological_distillation_loss(
    student_pred,
    teacher_target,
):

    if student_pred.shape != teacher_target.shape:
        raise ValueError(
            "student/teacher biological shape mismatch"
        )

    if (
        student_pred.ndim != 2
        or student_pred.shape[1] != BIO_DIM
    ):
        raise ValueError(
            f"Expected [N,{BIO_DIM}] biological tensors"
        )

    teacher_target = (
        teacher_target.detach()
    )

    student_pred = F.normalize(
        student_pred,
        p=2,
        dim=-1,
        eps=1e-12,
    )

    teacher_target = F.normalize(
        teacher_target,
        p=2,
        dim=-1,
        eps=1e-12,
    )

    cosine = torch.sum(
        student_pred
        *
        teacher_target,
        dim=-1,
    )

    return (
        1.0 - cosine
    ).mean()


V9D_METADATA = {

    "version":
        "V9-D",

    "chemistry_backbone":
        "V9-A-compatible-late-logit-fusion",

    "deployable_inputs": [
        "molecular_graph",
        "Morgan-SVD128",
    ],

    "beta":
        0.05,

    "biological_teacher": {
        "training_only":
            True,

        "base":
            "clean-DTI-target-ESM2",

        "ppi_signal":
            "PPI-refined-ESM2-minus-base",

        "teacher_pretrained":
            True,

        "validation_input":
            False,

        "inference_input":
            False,
    },

    "student_transfer": {
        "source":
            "GINE-256",

        "predictor":
            "256-to-256-to-128",

        "loss":
            "cosine",

        "lambda_candidates":
            list(
                VALID_LAMBDAS
            ),
    },

    "outer_test":
        "SEALED",
}


__all__ = [
    "V9DStudent",
    "V9DBiologicalTeacher",
    "freeze_teacher",
    "biological_distillation_loss",
    "VALID_BETAS",
    "VALID_LAMBDAS",
    "V9D_METADATA",
]
