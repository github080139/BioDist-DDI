#!/usr/bin/env python3

"""
AM-DDI Common-1087 V9-D
ONE-TIME FINAL OUTER-TEST EVALUATION

FROZEN PROTOCOL
---------------
Model:
    V9-D chemistry student
    GINE + Morgan-SVD128 late logit fusion

Inference:
    Chemistry only.
    Biological teacher/context is NEVER used.

Frozen hyperparameters:
    beta   = 0.05
    lambda = 0.01

Frozen seeds:
    42, 123, 456, 789, 2026

Final-evaluation policy:
    - checkpoint-only evaluation
    - NO retraining
    - NO model selection
    - NO threshold tuning
    - NO hyperparameter tuning
    - NO checkpoint replacement
    - fixed threshold = 0.5
    - AB/BA symmetric logit averaging
    - outer test loaded only after explicit double unlock
    - all 10 frozen checkpoint SHA256 hashes verified BEFORE test access

This script is intended to be audited and hashed BEFORE its
single final execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from models.dpddi.amddi_common_1087_v9d_ppi_teacher_distillation import (
    V9DStudent,
)


ROOT = Path(__file__).resolve().parents[1]

SPLIT_ROOT = (
    ROOT
    / "data/amddi_common_1087_coldstart_dev_splits_v2"
)

MORGAN_ROOT = (
    ROOT
    / "data/amddi_common_1087_v9a_morgan_svd"
)

GRAPH_FILE = (
    ROOT
    / "data/amddi_common_1087_structure"
    / "molecular_graphs.pt"
)

CHECKPOINT_ROOT = (
    ROOT
    / "checkpoints/amddi_common_1087"
    / "v9D_ppi_teacher_distillation"
    / "development"
)

RESULT_ROOT = (
    ROOT
    / "results_final/amddi_common_1087"
    / "v9D_ppi_teacher_distillation"
    / "FINAL_OUTER_TEST"
)

NUM_DRUGS = 1087
MORGAN_DIM = 128
BATCH_SIZE = 256

FIXED_BETA = 0.05
FIXED_LAMBDA = 0.01
FIXED_THRESHOLD = 0.5

SEEDS = [42, 123, 456, 789, 2026]

SPLITS = [
    "cold_start_S1",
    "cold_start_S2",
]


EXPECTED_CHECKPOINT_SHA256 = {

    ("cold_start_S1", 42):
        "75bc501794e64ecdff1293d3db0421352bf715a591f890ef52e4646f4596cfaa",

    ("cold_start_S1", 123):
        "e209fdb0c3ac5b77e38656e914232bdf8f0f5e93fb5fbcfc443eba52bf3dae5b",

    ("cold_start_S1", 456):
        "6057840423735bb137b115e0d62835b92deaeae130befde7c24d3d8899d75802",

    ("cold_start_S1", 789):
        "c793bfa7297efc006aaab517d9ac6333b328c0c9485f9cf4806a976d69b0f96b",

    ("cold_start_S1", 2026):
        "b1b4ee44f5c467725c84fa731eab63bbfb79689821c8a68832fb82de0ea394f7",

    ("cold_start_S2", 42):
        "64461ea8871920e65e170effb4e34a77934a6f5473219dc801c086d50c7e988f",

    ("cold_start_S2", 123):
        "cb4a866ae42b6b1e5a5bf8a09c7106b4c989bb4a696a941a68d2efbdf722bc27",

    ("cold_start_S2", 456):
        "3014f9aa71ff581abef4b638df28550033981628f984d212dbd18d0d159a6306",

    ("cold_start_S2", 789):
        "af8dcf685afbe8786d2b221059af8348139b52ac5f1b8d75a1b5b582b55e9259",

    ("cold_start_S2", 2026):
        "5dc64570ae109a066cf32bb50b49608a3083ce494eba90290b2277243660470d",
}


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def checkpoint_path(
    split_name: str,
    seed: int,
) -> Path:

    return (
        CHECKPOINT_ROOT
        / split_name
        / f"seed_{seed}"
        / f"lambda_{FIXED_LAMBDA:.2f}"
        / "best.pt"
    )


def verify_all_checkpoints():

    print("=" * 100)
    print("FROZEN CHECKPOINT IMMUTABILITY AUDIT")
    print("=" * 100)

    for split_name in SPLITS:

        for seed in SEEDS:

            path = checkpoint_path(
                split_name,
                seed,
            )

            if not path.is_file():

                raise RuntimeError(
                    f"Missing frozen checkpoint: {path}"
                )

            observed = sha256_file(path)

            expected = (
                EXPECTED_CHECKPOINT_SHA256[
                    (split_name, seed)
                ]
            )

            if observed != expected:

                raise RuntimeError(
                    "Frozen checkpoint SHA256 mismatch:\n"
                    f"  split    = {split_name}\n"
                    f"  seed     = {seed}\n"
                    f"  expected = {expected}\n"
                    f"  observed = {observed}"
                )

            print(
                f"PASS {split_name} seed={seed}: "
                f"{observed}"
            )

    print()
    print("PASS: all 10 frozen checkpoints verified.")


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

        return len(self.samples)

    def __getitem__(
        self,
        idx,
    ):

        return self.samples[idx]


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

        for a, b, y in batch:

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

        morgan_a = self.morgan[ia]
        morgan_b = self.morgan[ib]

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


def load_graphs():

    graphs = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if len(graphs) != NUM_DRUGS:

        raise RuntimeError(
            f"Expected {NUM_DRUGS} molecular graphs, "
            f"found {len(graphs)}"
        )

    return graphs


def load_morgan_svd(
    split_name,
):

    path = (
        MORGAN_ROOT
        / f"{split_name}_morgan_svd128.pt"
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
            f"Unexpected Morgan shape: {tuple(x.shape)}"
        )

    if torch.isnan(x).any():

        raise RuntimeError(
            "NaN detected in Morgan-SVD tensor"
        )

    return x.float().contiguous()


def normalize_pairs(
    x,
    name,
):

    if not torch.is_tensor(x):

        raise RuntimeError(
            f"{name} is not a tensor"
        )

    x = x.detach().cpu()

    if x.ndim != 2:

        raise RuntimeError(
            f"{name}: expected rank-2 tensor"
        )

    if x.shape[1] == 2:

        out = x

    elif x.shape[0] == 2:

        out = x.t()

    else:

        raise RuntimeError(
            f"{name}: expected [N,2] or [2,N], "
            f"found {tuple(x.shape)}"
        )

    return (
        out.long()
        .contiguous()
    )


def load_outer_test(
    split_name,
):

    # THIS IS THE ONLY FUNCTION IN THIS SCRIPT
    # THAT LOADS PAIR-LEVEL OUTER-TEST DATA.

    split_dir = (
        SPLIT_ROOT
        / split_name
    )

    test_pos = normalize_pairs(
        torch.load(
            split_dir / "test_pos.pt",
            map_location="cpu",
            weights_only=True,
        ),
        f"{split_name}/test_pos",
    )

    test_neg = normalize_pairs(
        torch.load(
            split_dir / "test_neg.pt",
            map_location="cpu",
            weights_only=True,
        ),
        f"{split_name}/test_neg",
    )

    return test_pos, test_neg


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

    labels_np = np.array(labels)
    probs_np = np.array(probs)

    pred = (
        probs_np >= FIXED_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            labels_np,
            pred,
            labels=[0, 1],
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

    # Deliberately chemistry-only:
    # no teacher, DTI, PPI or ESM tensor is accepted here.

    for (
        batch_a,
        batch_b,
        morgan_a,
        morgan_b,
        labels,
    ) in loader:

        batch_a = batch_a.to(device)
        batch_b = batch_b.to(device)

        morgan_a = morgan_a.to(device)
        morgan_b = morgan_b.to(device)

        labels = labels.to(device)

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
            float(loss.item())
            * n
        )

        total_n += n

    metrics = compute_metrics(
        labels_all,
        probs_all,
    )

    metrics["loss"] = (
        total_loss
        / total_n
    )

    metrics["n"] = int(total_n)

    return metrics


def verify_checkpoint_metadata(
    checkpoint,
    split_name,
    seed,
):

    required = [
        "version",
        "model_state_dict",
        "best_epoch",
        "best_val_auroc",
        "lambda_align",
        "beta",
        "seed",
        "outer_test_loaded",
    ]

    for key in required:

        if key not in checkpoint:

            raise RuntimeError(
                f"{split_name} seed={seed}: "
                f"checkpoint missing {key}"
            )

    if int(checkpoint["seed"]) != seed:

        raise RuntimeError(
            f"{split_name} seed={seed}: seed mismatch"
        )

    if (
        abs(
            float(checkpoint["beta"])
            - FIXED_BETA
        )
        > 1e-12
    ):

        raise RuntimeError(
            f"{split_name} seed={seed}: beta mismatch"
        )

    if (
        abs(
            float(checkpoint["lambda_align"])
            - FIXED_LAMBDA
        )
        > 1e-12
    ):

        raise RuntimeError(
            f"{split_name} seed={seed}: lambda mismatch"
        )

    if checkpoint["outer_test_loaded"] is not False:

        raise RuntimeError(
            f"{split_name} seed={seed}: "
            "development checkpoint does not certify "
            "outer_test_loaded=False"
        )


def aggregate(
    seed_results,
):

    metric_names = [
        "auroc",
        "aupr",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "loss",
    ]

    out = {}

    for metric in metric_names:

        values = np.array(
            [
                x["metrics"][metric]
                for x in seed_results
            ],
            dtype=np.float64,
        )

        out[metric] = {
            "mean":
                float(values.mean()),

            "sample_sd":
                float(
                    values.std(ddof=1)
                ),
        }

    return out


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "ONE-TIME frozen V9-D final outer-test evaluation"
        )
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
    )

    parser.add_argument(
        "--unlock-outer-test",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 100)
    print("AM-DDI COMMON-1087 V9-D")
    print("ONE-TIME FINAL OUTER-TEST EVALUATOR")
    print("=" * 100)

    print("Retraining           : NO")
    print("Model selection      : NO")
    print("Threshold tuning     : NO")
    print("Biology at inference : NO")
    print(f"Fixed beta           : {FIXED_BETA}")
    print(f"Fixed lambda         : {FIXED_LAMBDA}")
    print(f"Fixed threshold      : {FIXED_THRESHOLD}")
    print(f"Seeds                : {SEEDS}")
    print()

    # ------------------------------------------------------------
    # HARD DOUBLE-LOCK
    # ------------------------------------------------------------

    if (
        args.evaluate_test
        or args.unlock_outer_test
    ) and not args.eval_only:

        raise RuntimeError(
            "Outer test can only be accessed with --eval-only."
        )

    if (
        args.evaluate_test
        != args.unlock_outer_test
    ):

        raise RuntimeError(
            "Final outer-test evaluation requires BOTH "
            "--evaluate-test and --unlock-outer-test."
        )

    # ------------------------------------------------------------
    # CHECKPOINT IMMUTABILITY BEFORE ANY TEST ACCESS
    # ------------------------------------------------------------

    verify_all_checkpoints()

    if not args.evaluate_test:

        print()
        print("=" * 100)
        print("PREFLIGHT COMPLETE")
        print("=" * 100)
        print("Outer-test pair tensors loaded : NO")
        print("Outer-test status              : SEALED")
        print()
        print(
            "This was a preflight-only execution. "
            "No final metrics were produced."
        )

        return

    # ------------------------------------------------------------
    # FROM THIS POINT FORWARD, FINAL OUTER TEST IS UNLOCKED.
    # ------------------------------------------------------------

    print()
    print("!" * 100)
    print("FINAL OUTER TEST IS NOW BEING UNLOCKED")
    print("THIS IS THE PREDEFINED ONE-TIME EVALUATION EVENT")
    print("!" * 100)
    print()

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs = load_graphs()

    criterion = nn.BCEWithLogitsLoss()

    all_results = {}

    for split_name in SPLITS:

        print()
        print("=" * 100)
        print(split_name)
        print("=" * 100)

        morgan = load_morgan_svd(
            split_name
        )

        # Pair-level outer test is loaded exactly here,
        # after all ten checkpoint hashes passed.
        test_pos, test_neg = load_outer_test(
            split_name
        )

        print(
            f"Outer-test positives : {len(test_pos)}"
        )

        print(
            f"Outer-test negatives : {len(test_neg)}"
        )

        test_dataset = DDIPairDataset(
            test_pos,
            test_neg,
        )

        collator = GraphSVDPairCollator(
            graphs,
            morgan,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )

        split_results = []

        for seed in SEEDS:

            ckpt_path = checkpoint_path(
                split_name,
                seed,
            )

            # SHA already verified before outer-test loading.
            checkpoint = torch.load(
                ckpt_path,
                map_location=device,
                weights_only=True,
            )

            verify_checkpoint_metadata(
                checkpoint,
                split_name,
                seed,
            )

            model = V9DStudent(
                beta=FIXED_BETA,
                dropout=0.30,
            ).to(device)

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ],
                strict=True,
            )

            metrics = evaluate(
                model,
                test_loader,
                criterion,
                device,
            )

            row = {
                "seed":
                    seed,

                "checkpoint_sha256":
                    EXPECTED_CHECKPOINT_SHA256[
                        (split_name, seed)
                    ],

                "best_epoch":
                    int(
                        checkpoint[
                            "best_epoch"
                        ]
                    ),

                "best_inner_val_auroc":
                    float(
                        checkpoint[
                            "best_val_auroc"
                        ]
                    ),

                "metrics":
                    metrics,
            }

            split_results.append(
                row
            )

            print(
                f"seed={seed:4d} | "
                f"AUROC={metrics['auroc']:.6f} | "
                f"AUPR={metrics['aupr']:.6f} | "
                f"ACC={metrics['accuracy']:.6f} | "
                f"F1={metrics['f1']:.6f} | "
                f"MCC={metrics['mcc']:.6f}"
            )

            del model
            del checkpoint

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary = aggregate(
            split_results
        )

        all_results[split_name] = {
            "seeds":
                split_results,

            "aggregate":
                summary,
        }

        print()
        print("AGGREGATE mean +/- sample SD")

        for metric in [
            "auroc",
            "aupr",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "mcc",
        ]:

            x = summary[metric]

            print(
                f"{metric:12s}: "
                f"{x['mean']:.6f} +/- "
                f"{x['sample_sd']:.6f}"
            )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "version":
            "AM-DDI-common-1087-V9-D-FINAL",

        "evaluation_protocol":
            "checkpoint_only_no_retraining",

        "outer_test_evaluated":
            True,

        "outer_test_used_for_selection":
            False,

        "retraining":
            False,

        "biology_at_inference":
            False,

        "beta":
            FIXED_BETA,

        "lambda_align":
            FIXED_LAMBDA,

        "threshold":
            FIXED_THRESHOLD,

        "seeds":
            SEEDS,

        "checkpoint_sha256":
            {
                f"{split}_{seed}":
                    EXPECTED_CHECKPOINT_SHA256[
                        (split, seed)
                    ]
                for split in SPLITS
                for seed in SEEDS
            },

        "results":
            all_results,
    }

    out_file = (
        RESULT_ROOT
        / "final_outer_test_metrics.json"
    )

    if out_file.exists():

        raise RuntimeError(
            "FINAL result file already exists. "
            "Refusing to overwrite one-time outer-test results."
        )

    tmp_file = out_file.with_suffix(
        ".json.tmp"
    )

    tmp_file.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    tmp_file.replace(
        out_file
    )

    print()
    print("=" * 100)
    print("FINAL OUTER-TEST EVALUATION COMPLETE")
    print("=" * 100)
    print(f"Result: {out_file}")
    print("Retraining                 : NO")
    print("Biology at inference       : NO")
    print("Outer test used selection  : NO")
    print("Post-test tuning permitted : NO")


if __name__ == "__main__":
    main()
