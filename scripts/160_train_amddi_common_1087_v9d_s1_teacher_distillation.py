#!/usr/bin/env python3

"""
AM-DDI Common-1087 — V9-D

V9-A-compatible chemistry model with TRAINING-ONLY biological distillation.

DDI inference:
    GINE pair logit + beta * Morgan-SVD pair logit

Biological transfer:
    molecular GINE representation
        -> bio predictor
        -> frozen biological-teacher representation

Biological tensors:
    used ONLY for inner-training drug-level auxiliary batches.

Validation:
    molecular graph + Morgan only.

Outer test:
    NEVER loaded during development.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from torch_geometric.data import Batch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from models.dpddi.amddi_common_1087_v9d_ppi_teacher_distillation import (
    V9DStudent,
    V9DBiologicalTeacher,
    freeze_teacher,
    biological_distillation_loss,
    VALID_LAMBDAS,
)


NUM_DRUGS = 1087
MORGAN_DIM = 128

VALID_SEEDS = (
    42,
    123,
    456,
    789,
    2026,
)

VALID_SPLITS = (
    "cold_start_S1",
    "cold_start_S2",
)

GRAPH_FILE = (
    ROOT
    / "data/amddi_common_1087_structure"
    / "molecular_graphs.pt"
)

MORGAN_ROOT = (
    ROOT
    / "data/amddi_common_1087_v9a_morgan_svd"
)

SPLIT_ROOT = (
    ROOT
    / "data/amddi_common_1087_coldstart_dev_splits_v2"
)

BIO_ROOT = (
    ROOT
    / "data/amddi_common_1087_leakage_free_features"
)

TEACHER_ROOT = (
    ROOT
    / "checkpoints/amddi_common_1087_v9d_teacher"
)

RESULT_ROOT = (
    ROOT
    / "results_final/amddi_common_1087"
    / "v9D_ppi_teacher_distillation"
)

CHECKPOINT_ROOT = (
    ROOT
    / "checkpoints/amddi_common_1087"
    / "v9D_ppi_teacher_distillation"
)


BATCH_SIZE = 256
BIO_BATCH_SIZE = 64

DROPOUT = 0.3

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

GRAD_CLIP = 5.0

MAX_EPOCHS = 200
PATIENCE = 30
SMOKE_EPOCHS = 5

# V9-A's selected conservative Morgan residual.
FIXED_BETA = 0.05


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def atomic_json_save(
    obj,
    path,
):

    tmp = str(path) + ".tmp"

    with open(tmp, "w") as f:
        json.dump(
            obj,
            f,
            indent=2,
        )

    os.replace(
        tmp,
        path,
    )


def atomic_torch_save(
    obj,
    path,
):

    tmp = str(path) + ".tmp"

    torch.save(
        obj,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


class DDIPairDataset(Dataset):

    def __init__(
        self,
        pos_pairs,
        neg_pairs,
    ):

        self.samples = []

        for pair in pos_pairs:

            self.samples.append(
                (
                    int(pair[0]),
                    int(pair[1]),
                    1.0,
                )
            )

        for pair in neg_pairs:

            self.samples.append(
                (
                    int(pair[0]),
                    int(pair[1]),
                    0.0,
                )
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx,
    ):

        return self.samples[idx]


def load_graphs():

    graphs = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if len(graphs) != NUM_DRUGS:

        raise RuntimeError(
            "Unexpected molecular graph count"
        )

    return graphs


def load_morgan_svd(
    split_name,
):

    path = (
        MORGAN_ROOT
        /
        f"{split_name}_morgan_svd128.pt"
    )

    x = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if tuple(x.shape) != (
        NUM_DRUGS,
        MORGAN_DIM,
    ):

        raise RuntimeError(
            f"Unexpected Morgan shape {tuple(x.shape)}"
        )

    if torch.isnan(x).any():

        raise RuntimeError(
            "NaN in Morgan tensor"
        )

    return (
        x.float()
        .contiguous()
    )


def load_split(
    split_name,
):

    split_dir = (
        SPLIT_ROOT
        /
        split_name
    )

    out = {}

    # Intentionally NO test_pos/test_neg.
    for key in (
        "train_pos",
        "train_neg",
        "val_pos",
        "val_neg",
    ):

        out[key] = torch.load(
            split_dir
            /
            f"{key}.pt",
            map_location="cpu",
            weights_only=True,
        )

    out["inner_train_drugs"] = torch.load(
        split_dir
        /
        "inner_train_drugs.pt",
        map_location="cpu",
        weights_only=True,
    ).long().flatten()

    return out


def load_biology(
    split_name,
    inner_train,
):

    base_path = (
        BIO_ROOT
        /
        f"{split_name}_v9d_clean_dti_esm2_context_128.pt"
    )

    ppi_path = (
        BIO_ROOT
        /
        f"{split_name}_ppi_esm2_context_128.pt"
    )

    base = torch.load(
        base_path,
        map_location="cpu",
        weights_only=True,
    ).float()

    ppi = torch.load(
        ppi_path,
        map_location="cpu",
        weights_only=True,
    ).float()

    for name, x in (
        ("base", base),
        ("ppi", ppi),
    ):

        if tuple(x.shape) != (
            NUM_DRUGS,
            128,
        ):

            raise RuntimeError(
                f"{name}: bad shape {tuple(x.shape)}"
            )

        if torch.isnan(x).any():

            raise RuntimeError(
                f"{name}: NaN detected"
            )

        nz = (
            x.abs().sum(dim=1) > 0
        )

        if int(nz.sum()) != 685:

            raise RuntimeError(
                f"{name}: expected 685 nonzero rows, "
                f"got {int(nz.sum())}"
            )

        if not bool(
            nz[inner_train].all()
        ):

            raise RuntimeError(
                f"{name}: missing inner-training biology"
            )

        outside = torch.ones(
            NUM_DRUGS,
            dtype=torch.bool,
        )

        outside[inner_train] = False

        if bool(
            nz[outside].any()
        ):

            raise RuntimeError(
                f"{name}: biology exists outside inner training"
            )

    return (
        base.contiguous(),
        ppi.contiguous(),
    )


class GraphSVDPairCollator:

    def __init__(
        self,
        graphs,
        morgan,
    ):

        self.graphs = graphs
        self.morgan = morgan

    def __call__(
        self,
        batch,
    ):

        ids_a = []
        ids_b = []
        labels = []

        for a,b,y in batch:

            ids_a.append(a)
            ids_b.append(b)
            labels.append(y)

        graph_a = Batch.from_data_list(
            [
                self.graphs[i]
                for i in ids_a
            ]
        )

        graph_b = Batch.from_data_list(
            [
                self.graphs[i]
                for i in ids_b
            ]
        )

        ia = torch.tensor(
            ids_a,
            dtype=torch.long,
        )

        ib = torch.tensor(
            ids_b,
            dtype=torch.long,
        )

        morgan_a = self.morgan[
            ia
        ]

        morgan_b = self.morgan[
            ib
        ]

        labels = torch.tensor(
            labels,
            dtype=torch.float32,
        )

        return (
            graph_a,
            graph_b,
            morgan_a,
            morgan_b,
            labels,
        )


def symmetric_logits(
    model,
    batch_a,
    batch_b,
    morgan_a,
    morgan_b,
):

    za = model.encode_graph_batch(
        batch_a
    )

    zb = model.encode_graph_batch(
        batch_b
    )

    graph_ab = (
        model.score_graph_embeddings(
            za,
            zb,
        )
    )

    graph_ba = (
        model.score_graph_embeddings(
            zb,
            za,
        )
    )

    morgan_ab = (
        model.score_morgan_embeddings(
            morgan_a,
            morgan_b,
        )
    )

    morgan_ba = (
        model.score_morgan_embeddings(
            morgan_b,
            morgan_a,
        )
    )

    logit_ab = model.fuse_logits(
        graph_ab,
        morgan_ab,
    )

    logit_ba = model.fuse_logits(
        graph_ba,
        morgan_ba,
    )

    return 0.5 * (
        logit_ab
        +
        logit_ba
    )


def compute_metrics(
    labels,
    probs,
):

    labels_np = np.array(
        labels
    )

    probs_np = np.array(
        probs
    )

    pred = (
        probs_np >= 0.5
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            labels_np,
            pred,
            labels=[0,1],
        ).ravel()
    )

    return {

        "auroc":
            float(
                roc_auc_score(
                    labels_np,
                    probs_np,
                )
            ),

        "aupr":
            float(
                average_precision_score(
                    labels_np,
                    probs_np,
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    labels_np,
                    pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    labels_np,
                    pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    labels_np,
                    pred,
                    zero_division=0,
                )
            ),

        "specificity":
            float(
                tn
                /
                max(
                    tn + fp,
                    1,
                )
            ),

        "f1":
            float(
                f1_score(
                    labels_np,
                    pred,
                    zero_division=0,
                )
            ),

        "mcc":
            float(
                matthews_corrcoef(
                    labels_np,
                    pred,
                )
            ),
    }


def make_bio_batch(
    inner_train,
    graphs,
    generator,
):

    n = min(
        BIO_BATCH_SIZE,
        inner_train.numel(),
    )

    perm = torch.randperm(
        inner_train.numel(),
        generator=generator,
    )[:n]

    ids = inner_train[
        perm
    ]

    graph_batch = (
        Batch.from_data_list(
            [
                graphs[
                    int(i)
                ]
                for i in ids
            ]
        )
    )

    return (
        ids,
        graph_batch,
    )


def train_epoch(
    model,
    teacher,
    loader,
    optimizer,
    criterion,
    device,
    graphs,
    inner_train,
    base,
    ppi,
    lambda_align,
    bio_generator,
):

    model.train()

    # Frozen teacher stays in eval mode.
    teacher.eval()

    total = 0.0
    total_ddi = 0.0
    total_bio = 0.0
    total_n = 0

    for (
        batch_a,
        batch_b,
        morgan_a,
        morgan_b,
        labels,
    ) in loader:

        batch_a = batch_a.to(
            device
        )

        batch_b = batch_b.to(
            device
        )

        morgan_a = morgan_a.to(
            device
        )

        morgan_b = morgan_b.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------
        # Ordinary V9-A DDI objective
        # --------------------------------------------------

        logits = symmetric_logits(
            model,
            batch_a,
            batch_b,
            morgan_a,
            morgan_b,
        )

        ddi_loss = criterion(
            logits,
            labels,
        )

        # --------------------------------------------------
        # Separate drug-level biological transfer batch
        # --------------------------------------------------

        (
            bio_ids,
            bio_graph_batch,
        ) = make_bio_batch(
            inner_train,
            graphs,
            bio_generator,
        )

        bio_graph_batch = (
            bio_graph_batch.to(
                device
            )
        )

        student_graph_z = (
            model.encode_graph_batch(
                bio_graph_batch
            )
        )

        student_bio = (
            model.predict_biology(
                student_graph_z
            )
        )

        with torch.no_grad():

            teacher_bio = teacher(
                base[bio_ids].to(
                    device
                ),
                ppi[bio_ids].to(
                    device
                ),
            )

        bio_loss = (
            biological_distillation_loss(
                student_bio,
                teacher_bio,
            )
        )

        loss = (
            ddi_loss
            +
            float(lambda_align)
            *
            bio_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        n = labels.shape[0]

        total += (
            float(
                loss.item()
            )
            *
            n
        )

        total_ddi += (
            float(
                ddi_loss.item()
            )
            *
            n
        )

        total_bio += (
            float(
                bio_loss.item()
            )
            *
            n
        )

        total_n += n

    return {
        "total":
            total / total_n,

        "ddi":
            total_ddi / total_n,

        "bio":
            total_bio / total_n,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    labels_all = []
    probs_all = []

    total_loss = 0.0
    total_n = 0

    # IMPORTANT:
    # no teacher and no biological tensors appear here.

    for (
        batch_a,
        batch_b,
        morgan_a,
        morgan_b,
        labels,
    ) in loader:

        batch_a = batch_a.to(
            device
        )

        batch_b = batch_b.to(
            device
        )

        morgan_a = morgan_a.to(
            device
        )

        morgan_b = morgan_b.to(
            device
        )

        labels = labels.to(
            device
        )

        logits = symmetric_logits(
            model,
            batch_a,
            batch_b,
            morgan_a,
            morgan_b,
        )

        loss = criterion(
            logits,
            labels,
        )

        probs = torch.sigmoid(
            logits
        )

        labels_all.extend(
            labels.cpu().tolist()
        )

        probs_all.extend(
            probs.cpu().tolist()
        )

        n = labels.shape[0]

        total_loss += (
            float(
                loss.item()
            )
            *
            n
        )

        total_n += n

    metrics = compute_metrics(
        labels_all,
        probs_all,
    )

    metrics["loss"] = (
        total_loss
        /
        total_n
    )

    return metrics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=VALID_SPLITS,
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "smoke",
            "development",
        ),
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=VALID_SEEDS,
    )

    parser.add_argument(
        "--lambda-align",
        required=True,
        type=float,
        choices=VALID_LAMBDAS,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.split != "cold_start_S2":

        raise RuntimeError(
            "Initial V9-D development is locked to "
            "cold_start_S2."
        )

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    device = torch.device(
        args.device
    )

    result_dir = (
        RESULT_ROOT
        /
        "runs"
        /
        args.stage
        /
        args.split
        /
        f"seed_{args.seed}"
        /
        f"lambda_{args.lambda_align:.2f}"
    )

    checkpoint_dir = (
        CHECKPOINT_ROOT
        /
        args.stage
        /
        args.split
        /
        f"seed_{args.seed}"
        /
        f"lambda_{args.lambda_align:.2f}"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_file = (
        result_dir
        /
        "validation_metrics.json"
    )

    best_file = (
        checkpoint_dir
        /
        "best.pt"
    )

    if metrics_file.exists():

        print(
            "PASS: already complete",
            metrics_file,
        )

        return

    graphs = load_graphs()

    split = load_split(
        args.split
    )

    morgan = load_morgan_svd(
        args.split
    )

    inner_train = (
        split[
            "inner_train_drugs"
        ]
    )

    base, ppi = load_biology(
        args.split,
        inner_train,
    )

    # --------------------------------------------------------
    # Teacher checkpoint
    # --------------------------------------------------------

    teacher_file = (
        TEACHER_ROOT
        /
        f"s2_seed_{args.seed}.pt"
    )

    if not teacher_file.exists():

        raise FileNotFoundError(
            f"Missing teacher checkpoint: "
            f"{teacher_file}"
        )

    teacher_checkpoint = torch.load(
        teacher_file,
        map_location="cpu",
        weights_only=True,
    )

    teacher = (
        V9DBiologicalTeacher(
            dropout=DROPOUT
        )
    )

    teacher.load_state_dict(
        teacher_checkpoint[
            "teacher_state_dict"
        ],
        strict=True,
    )

    freeze_teacher(
        teacher
    )

    teacher = teacher.to(
        device
    )

    # --------------------------------------------------------
    # DDI loaders
    # --------------------------------------------------------

    train_dataset = DDIPairDataset(
        split["train_pos"],
        split["train_neg"],
    )

    val_dataset = DDIPairDataset(
        split["val_pos"],
        split["val_neg"],
    )

    collator = GraphSVDPairCollator(
        graphs,
        morgan,
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        args.seed
    )

    bio_generator = (
        torch.Generator()
    )

    bio_generator.manual_seed(
        args.seed + 100000
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    model = V9DStudent(
        beta=FIXED_BETA,
        dropout=DROPOUT,
    ).to(
        device
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    epochs = (
        SMOKE_EPOCHS
        if args.stage == "smoke"
        else MAX_EPOCHS
    )

    print("=" * 96)
    print("AM-DDI COMMON-1087 V9-D")
    print("=" * 96)
    print(
        "split                :",
        args.split,
    )
    print(
        "stage                :",
        args.stage,
    )
    print(
        "seed                 :",
        args.seed,
    )
    print(
        "beta                 :",
        FIXED_BETA,
    )
    print(
        "lambda_align         :",
        args.lambda_align,
    )
    print(
        "DDI batch size       :",
        BATCH_SIZE,
    )
    print(
        "bio drug batch size  :",
        BIO_BATCH_SIZE,
    )
    print(
        "inner-training drugs :",
        inner_train.numel(),
    )
    print(
        "teacher epoch        :",
        teacher_checkpoint[
            "epoch"
        ],
    )
    print(
        "teacher monitor AUROC:",
        teacher_checkpoint[
            "monitor_auroc"
        ],
    )
    print(
        "teacher frozen       :",
        all(
            not p.requires_grad
            for p in teacher.parameters()
        ),
    )
    print(
        "validation biology   : NOT USED"
    )
    print(
        "outer test           : SEALED / NOT LOADED"
    )
    print("=" * 96)

    best_auc = -float("inf")
    best_epoch = -1
    stale = 0

    for epoch in range(
        1,
        epochs + 1,
    ):

        losses = train_epoch(
            model,
            teacher,
            train_loader,
            optimizer,
            criterion,
            device,
            graphs,
            inner_train,
            base,
            ppi,
            args.lambda_align,
            bio_generator,
        )

        metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"total {losses['total']:.4f} | "
            f"DDI {losses['ddi']:.4f} | "
            f"bio {losses['bio']:.4f} | "
            f"val AUROC {metrics['auroc']:.6f} | "
            f"AUPR {metrics['aupr']:.6f}"
        )

        if (
            metrics["auroc"]
            >
            best_auc
        ):

            best_auc = (
                metrics["auroc"]
            )

            best_epoch = epoch
            stale = 0

            atomic_torch_save(
                {
                    "version":
                        "AM-DDI-common-1087-V9-D",

                    "model_state_dict":
                        model.state_dict(),

                    "best_epoch":
                        best_epoch,

                    "best_val_auroc":
                        best_auc,

                    "lambda_align":
                        args.lambda_align,

                    "beta":
                        FIXED_BETA,

                    "seed":
                        args.seed,

                    "teacher_checkpoint":
                        str(
                            teacher_file
                        ),

                    "teacher_checkpoint_sha256":
                        sha256_file(
                            teacher_file
                        ),

                    "outer_test_loaded":
                        False,
                },
                best_file,
            )

        else:

            stale += 1

        if (
            args.stage == "development"
            and stale >= PATIENCE
        ):

            print(
                f"Early stopping at epoch "
                f"{epoch}; best epoch "
                f"{best_epoch}"
            )

            break

    checkpoint = torch.load(
        best_file,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    final_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    result = {

        "version":
            "AM-DDI-common-1087-V9-D",

        "stage":
            args.stage,

        "split":
            args.split,

        "seed":
            args.seed,

        "beta":
            FIXED_BETA,

        "lambda_align":
            args.lambda_align,

        "architecture":
            (
                "V9-A GINE + Morgan-SVD128 "
                "late-logit-fusion with "
                "training-only PPI/ESM "
                "biological distillation"
            ),

        "bio_batch_size":
            BIO_BATCH_SIZE,

        "teacher_checkpoint":
            str(
                teacher_file
            ),

        "teacher_checkpoint_sha256":
            sha256_file(
                teacher_file
            ),

        "teacher_best_epoch":
            int(
                teacher_checkpoint[
                    "epoch"
                ]
            ),

        "teacher_monitor_auroc":
            float(
                teacher_checkpoint[
                    "monitor_auroc"
                ]
            ),

        "validation_biology_used":
            False,

        "outer_test_status":
            "SEALED",

        "outer_test_loaded":
            False,

        "best_epoch":
            best_epoch,

        "validation":
            final_metrics,
    }

    atomic_json_save(
        result,
        metrics_file,
    )

    print()
    print("=" * 96)
    print("BEST VALIDATION RESULT")
    print("=" * 96)
    print(
        "Best epoch:",
        best_epoch,
    )

    for k,v in (
        final_metrics.items()
    ):

        print(
            f"{k:12s}: "
            f"{v:.6f}"
        )

    print(
        "Validation biology : NOT USED"
    )
    print(
        "Outer test         : SEALED / NOT LOADED"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
