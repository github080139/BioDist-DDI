#!/usr/bin/env python3

"""
AM-DDI Common-1087 — V9-D biological teacher pretraining.

STRICT DEVELOPMENT PROTOCOL
---------------------------
Teacher inputs:
    base = clean DTI-target aggregated ESM2
    ppi  = PPI-refined ESM2
    residual = ppi - base

Teacher supervision:
    S1 inner-training DDI pairs ONLY.

Checkpoint selection:
    deterministic monitor subset drawn ONLY from inner-training pairs.

NOT USED:
    official inner-validation DDI pairs
    inner-validation biological features
    outer-test pairs
    outer-test labels

The pretrained teacher is training-only privileged information.
"""

from pathlib import Path
import argparse
import json
import random
import hashlib

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from models.dpddi.amddi_common_1087_v9d_ppi_teacher_distillation import (
    V9DBiologicalTeacher,
)


NUM_DRUGS = 1087
BIO_DIM = 128

SPLIT_NAME = "cold_start_S1"

SPLIT_ROOT = Path(
    "data/amddi_common_1087_coldstart_dev_splits_v2"
)

FEATURE_ROOT = Path(
    "data/amddi_common_1087_leakage_free_features"
)

OUT_ROOT = Path(
    "results/amddi_common_1087_v9d_teacher"
)

MODEL_ROOT = Path(
    "checkpoints/amddi_common_1087_v9d_teacher"
)


class TeacherPairHead(nn.Module):

    def __init__(
        self,
        dim=BIO_DIM,
        dropout=0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dim * 3,
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
        za,
        zb,
    ):

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


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def normalize_pairs(x):

    x = x.long()

    if x.ndim != 2:
        raise RuntimeError(
            f"Bad pair shape {tuple(x.shape)}"
        )

    if x.shape[1] == 2:
        return x.contiguous()

    if x.shape[0] == 2:
        return x.t().contiguous()

    raise RuntimeError(
        f"Cannot interpret pair tensor {tuple(x.shape)}"
    )


def load_data():

    split_dir = (
        SPLIT_ROOT
        /
        SPLIT_NAME
    )

    train_pos = normalize_pairs(
        torch.load(
            split_dir / "train_pos.pt",
            map_location="cpu",
            weights_only=True,
        )
    )

    train_neg = normalize_pairs(
        torch.load(
            split_dir / "train_neg.pt",
            map_location="cpu",
            weights_only=True,
        )
    )

    inner_train = torch.load(
        split_dir / "inner_train_drugs.pt",
        map_location="cpu",
        weights_only=True,
    ).long().flatten()

    base = torch.load(
        FEATURE_ROOT
        /
        f"{SPLIT_NAME}_v9d_clean_dti_esm2_context_128.pt",
        map_location="cpu",
        weights_only=True,
    ).float()

    ppi = torch.load(
        FEATURE_ROOT
        /
        f"{SPLIT_NAME}_ppi_esm2_context_128.pt",
        map_location="cpu",
        weights_only=True,
    ).float()

    if tuple(base.shape) != (
        NUM_DRUGS,
        BIO_DIM,
    ):
        raise RuntimeError(
            f"Bad base shape {tuple(base.shape)}"
        )

    if tuple(ppi.shape) != (
        NUM_DRUGS,
        BIO_DIM,
    ):
        raise RuntimeError(
            f"Bad PPI shape {tuple(ppi.shape)}"
        )

    allowed = torch.zeros(
        NUM_DRUGS,
        dtype=torch.bool,
    )

    allowed[inner_train] = True

    endpoints = torch.cat(
        [
            train_pos.flatten(),
            train_neg.flatten(),
        ]
    )

    if not bool(
        allowed[endpoints].all()
    ):
        raise RuntimeError(
            "Teacher training pairs contain "
            "non-inner-training drugs"
        )

    base_nz = (
        base.abs().sum(dim=1) > 0
    )

    ppi_nz = (
        ppi.abs().sum(dim=1) > 0
    )

    if int(base_nz.sum()) != 685:
        raise RuntimeError(
            "Base context must contain exactly "
            "685 nonzero drug rows"
        )

    if int(ppi_nz.sum()) != 685:
        raise RuntimeError(
            "PPI context must contain exactly "
            "685 nonzero drug rows"
        )

    if not bool(
        base_nz[inner_train].all()
    ):
        raise RuntimeError(
            "Missing base biology for inner-training drug"
        )

    if not bool(
        ppi_nz[inner_train].all()
    ):
        raise RuntimeError(
            "Missing PPI biology for inner-training drug"
        )

    outside = torch.ones(
        NUM_DRUGS,
        dtype=torch.bool,
    )

    outside[inner_train] = False

    if bool(
        base_nz[outside].any()
    ):
        raise RuntimeError(
            "Base biology exposure outside inner training"
        )

    if bool(
        ppi_nz[outside].any()
    ):
        raise RuntimeError(
            "PPI biology exposure outside inner training"
        )

    return (
        train_pos,
        train_neg,
        inner_train,
        base,
        ppi,
    )


def make_teacher_split(
    pos,
    neg,
    seed,
    monitor_fraction=0.10,
):
    """
    Deterministic class-stratified partition of the OFFICIAL
    inner-training pairs.

    This does not touch official inner-validation pairs.
    """

    g = torch.Generator()
    g.manual_seed(seed)

    pos_perm = torch.randperm(
        pos.shape[0],
        generator=g,
    )

    neg_perm = torch.randperm(
        neg.shape[0],
        generator=g,
    )

    n_pos_monitor = max(
        1,
        int(
            round(
                pos.shape[0]
                *
                monitor_fraction
            )
        ),
    )

    n_neg_monitor = max(
        1,
        int(
            round(
                neg.shape[0]
                *
                monitor_fraction
            )
        ),
    )

    monitor_pos = pos[
        pos_perm[:n_pos_monitor]
    ]

    fit_pos = pos[
        pos_perm[n_pos_monitor:]
    ]

    monitor_neg = neg[
        neg_perm[:n_neg_monitor]
    ]

    fit_neg = neg[
        neg_perm[n_neg_monitor:]
    ]

    fit_pairs = torch.cat(
        [
            fit_pos,
            fit_neg,
        ],
        dim=0,
    )

    fit_labels = torch.cat(
        [
            torch.ones(
                fit_pos.shape[0]
            ),
            torch.zeros(
                fit_neg.shape[0]
            ),
        ]
    ).float()

    monitor_pairs = torch.cat(
        [
            monitor_pos,
            monitor_neg,
        ],
        dim=0,
    )

    monitor_labels = torch.cat(
        [
            torch.ones(
                monitor_pos.shape[0]
            ),
            torch.zeros(
                monitor_neg.shape[0]
            ),
        ]
    ).float()

    # Deterministically shuffle fit set.
    fit_perm = torch.randperm(
        fit_pairs.shape[0],
        generator=g,
    )

    fit_pairs = fit_pairs[
        fit_perm
    ]

    fit_labels = fit_labels[
        fit_perm
    ]

    return (
        fit_pairs,
        fit_labels,
        monitor_pairs,
        monitor_labels,
    )


def symmetric_pair_logits(
    head,
    za,
    zb,
):

    ab = head(
        za,
        zb,
    )

    ba = head(
        zb,
        za,
    )

    return 0.5 * (
        ab + ba
    )


@torch.no_grad()
def evaluate(
    teacher,
    head,
    base,
    ppi,
    pairs,
    labels,
    device,
    batch_size,
):

    teacher.eval()
    head.eval()

    probs = []

    for start in range(
        0,
        pairs.shape[0],
        batch_size,
    ):

        batch_pairs = pairs[
            start:
            start + batch_size
        ]

        a = batch_pairs[:, 0]
        b = batch_pairs[:, 1]

        za = teacher(
            base[a].to(device),
            ppi[a].to(device),
        )

        zb = teacher(
            base[b].to(device),
            ppi[b].to(device),
        )

        logits = symmetric_pair_logits(
            head,
            za,
            zb,
        )

        probs.append(
            torch.sigmoid(
                logits
            ).cpu()
        )

    probs = torch.cat(
        probs
    ).numpy()

    y = labels.numpy()

    return {
        "auroc":
            float(
                roc_auc_score(
                    y,
                    probs,
                )
            ),

        "aupr":
            float(
                average_precision_score(
                    y,
                    probs,
                )
            ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--monitor-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        args.device
    )

    (
        pos,
        neg,
        inner_train,
        base,
        ppi,
    ) = load_data()

    (
        fit_pairs,
        fit_labels,
        monitor_pairs,
        monitor_labels,
    ) = make_teacher_split(
        pos,
        neg,
        args.seed,
        args.monitor_fraction,
    )

    print("=" * 88)
    print("V9-D BIOLOGICAL TEACHER PRETRAINING")
    print("=" * 88)
    print("split                 :", SPLIT_NAME)
    print("seed                  :", args.seed)
    print("inner-training drugs  :", inner_train.numel())
    print("official train pos    :", pos.shape[0])
    print("official train neg    :", neg.shape[0])
    print("teacher fit pairs     :", fit_pairs.shape[0])
    print("teacher monitor pairs :", monitor_pairs.shape[0])
    print("monitor fraction      :", args.monitor_fraction)
    print("official inner-val    : NOT LOADED")
    print("outer test pairs      : NOT LOADED / SEALED")
    print("=" * 88)

    teacher = V9DBiologicalTeacher(
        dropout=0.3,
    ).to(device)

    head = TeacherPairHead(
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(
            teacher.parameters()
        )
        +
        list(
            head.parameters()
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    g = torch.Generator()
    g.manual_seed(
        args.seed
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        MODEL_ROOT
        /
        f"s1_seed_{args.seed}.pt"
    )

    result_path = (
        OUT_ROOT
        /
        f"s1_seed_{args.seed}.json"
    )

    best_auroc = -float("inf")
    best_epoch = -1
    stale = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        teacher.train()
        head.train()

        perm = torch.randperm(
            fit_pairs.shape[0],
            generator=g,
        )

        total_loss = 0.0
        total_n = 0

        for start in range(
            0,
            fit_pairs.shape[0],
            args.batch_size,
        ):

            idx = perm[
                start:
                start + args.batch_size
            ]

            pairs = fit_pairs[
                idx
            ]

            labels = fit_labels[
                idx
            ].to(device)

            a = pairs[:, 0]
            b = pairs[:, 1]

            optimizer.zero_grad(
                set_to_none=True
            )

            za = teacher(
                base[a].to(device),
                ppi[a].to(device),
            )

            zb = teacher(
                base[b].to(device),
                ppi[b].to(device),
            )

            logits = symmetric_pair_logits(
                head,
                za,
                zb,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(
                    teacher.parameters()
                )
                +
                list(
                    head.parameters()
                ),
                5.0,
            )

            optimizer.step()

            n = labels.shape[0]

            total_loss += (
                float(
                    loss.item()
                )
                *
                n
            )

            total_n += n

        metrics = evaluate(
            teacher,
            head,
            base,
            ppi,
            monitor_pairs,
            monitor_labels,
            device,
            args.batch_size,
        )

        train_loss = (
            total_loss
            /
            max(
                total_n,
                1,
            )
        )

        print(
            f"epoch {epoch:03d} | "
            f"loss {train_loss:.6f} | "
            f"monitor AUROC "
            f"{metrics['auroc']:.6f} | "
            f"AUPR "
            f"{metrics['aupr']:.6f}"
        )

        # Same strict AUROC-improvement convention
        # used by the frozen GINE development workflow.
        if (
            metrics["auroc"]
            >
            best_auroc
        ):

            best_auroc = (
                metrics["auroc"]
            )

            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "version":
                        "V9-D-biological-teacher",

                    "split":
                        SPLIT_NAME,

                    "seed":
                        args.seed,

                    "epoch":
                        epoch,

                    "monitor_auroc":
                        metrics["auroc"],

                    "monitor_aupr":
                        metrics["aupr"],

                    "teacher_state_dict":
                        teacher.state_dict(),

                    "head_state_dict":
                        head.state_dict(),

                    "teacher_input":
                        (
                            "clean-DTI-target-ESM2 "
                            "+ PPI-residual"
                        ),

                    "official_inner_validation_loaded":
                        False,

                    "outer_test_pairs_loaded":
                        False,
                },
                checkpoint_path,
            )

        else:
            stale += 1

        if stale >= args.patience:

            print(
                f"Early stopping at epoch "
                f"{epoch}"
            )

            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    result = {
        "version":
            "V9-D-biological-teacher",

        "split":
            SPLIT_NAME,

        "seed":
            args.seed,

        "inner_training_drugs":
            int(
                inner_train.numel()
            ),

        "official_train_positive_pairs":
            int(
                pos.shape[0]
            ),

        "official_train_negative_pairs":
            int(
                neg.shape[0]
            ),

        "teacher_fit_pairs":
            int(
                fit_pairs.shape[0]
            ),

        "teacher_monitor_pairs":
            int(
                monitor_pairs.shape[0]
            ),

        "monitor_fraction":
            args.monitor_fraction,

        "best_epoch":
            int(
                checkpoint["epoch"]
            ),

        "best_monitor_auroc":
            float(
                checkpoint[
                    "monitor_auroc"
                ]
            ),

        "best_monitor_aupr":
            float(
                checkpoint[
                    "monitor_aupr"
                ]
            ),

        "official_inner_validation_loaded":
            False,

        "outer_test_pairs_loaded":
            False,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_sha256":
            sha256(
                checkpoint_path
            ),
    }

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
        +
        "\n"
    )

    print()
    print("=" * 88)
    print("TEACHER PRETRAINING COMPLETE")
    print("=" * 88)
    print(
        "best epoch       :",
        result["best_epoch"],
    )
    print(
        "best monitor AUROC:",
        f"{result['best_monitor_auroc']:.6f}",
    )
    print(
        "best monitor AUPR :",
        f"{result['best_monitor_aupr']:.6f}",
    )
    print(
        "checkpoint SHA256 :",
        result["checkpoint_sha256"],
    )
    print(
        "official inner-val: NOT LOADED"
    )
    print(
        "outer test pairs  : NOT LOADED / SEALED"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
