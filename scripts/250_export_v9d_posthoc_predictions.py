#!/usr/bin/env python3

"""
POST-HOC V9-D PREDICTION EXPORTER

Purpose:
    Export pair-level validation and already-consumed outer-test
    predictions from the ORIGINAL frozen V9-D checkpoints.

IMPORTANT:
    - NO retraining
    - NO checkpoint selection
    - NO hyperparameter changes
    - NO beta/lambda changes
    - NO threshold tuning
    - NO model modification
    - Original final evaluator is NOT modified
    - Outer-test access is for secondary/post-hoc analysis only

Primary results remain unchanged.
"""

from pathlib import Path
import importlib.util
import json
import sys
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]

# Ensure project-local packages such as `models` are importable
# when the frozen evaluator is loaded dynamically.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL = (
    ROOT
    / "scripts/162_final_evaluate_amddi_common_1087_v9d.py"
)

OUT_ROOT = (
    ROOT
    / "results_final/amddi_common_1087"
    / "v9D_ppi_teacher_distillation"
    / "POSTHOC_PREDICTIONS"
)

# ---------------------------------------------------------
# Load original frozen evaluator as a module.
# We reuse its exact architecture, checkpoint paths,
# SHA hashes, graph loading, Morgan loading,
# collator, and symmetric logit function.
# ---------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "v9d_frozen_evaluator",
    ORIGINAL,
)

v9d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9d)


def load_validation_pairs(split_name):
    split_dir = v9d.SPLIT_ROOT / split_name

    val_pos = v9d.normalize_pairs(
        torch.load(
            split_dir / "val_pos.pt",
            map_location="cpu",
            weights_only=True,
        ),
        f"{split_name}/val_pos",
    )

    val_neg = v9d.normalize_pairs(
        torch.load(
            split_dir / "val_neg.pt",
            map_location="cpu",
            weights_only=True,
        ),
        f"{split_name}/val_neg",
    )

    return val_pos, val_neg


@torch.no_grad()
def predict_loader(
    model,
    loader,
    dataset,
    criterion,
    device,
):
    model.eval()

    logits_all = []
    probs_all = []
    labels_all = []

    total_loss = 0.0
    total_n = 0

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

        logits = v9d.symmetric_logits(
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

        probs = torch.sigmoid(logits)

        logits_all.extend(
            logits.detach().cpu().numpy().tolist()
        )

        probs_all.extend(
            probs.detach().cpu().numpy().tolist()
        )

        labels_all.extend(
            labels.detach().cpu().numpy().tolist()
        )

        n = labels.shape[0]

        total_loss += float(loss.item()) * n
        total_n += n

    if len(dataset.samples) != len(probs_all):
        raise RuntimeError(
            "Prediction/sample count mismatch: "
            f"{len(dataset.samples)} vs {len(probs_all)}"
        )

    rows = []

    for idx, ((a, b, y), z, p) in enumerate(
        zip(
            dataset.samples,
            logits_all,
            probs_all,
        )
    ):
        rows.append(
            {
                "pair_index": idx,
                "drug_a": int(a),
                "drug_b": int(b),
                "label": int(y),
                "logit": float(z),
                "probability": float(p),
                "prediction_at_0.5": int(
                    float(p) >= v9d.FIXED_THRESHOLD
                ),
            }
        )

    metrics = v9d.compute_metrics(
        labels_all,
        probs_all,
    )

    metrics["loss"] = (
        total_loss / max(total_n, 1)
    )

    metrics["n"] = int(total_n)

    return pd.DataFrame(rows), metrics


def run_one_role(
    split_name,
    seed,
    role,
    pos_pairs,
    neg_pairs,
    graphs,
    morgan,
    device,
):
    dataset = v9d.DDIPairDataset(
        pos_pairs,
        neg_pairs,
    )

    collator = v9d.GraphSVDPairCollator(
        graphs,
        morgan,
    )

    loader = DataLoader(
        dataset,
        batch_size=v9d.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    ckpt_path = v9d.checkpoint_path(
        split_name,
        seed,
    )

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=True,
    )

    v9d.verify_checkpoint_metadata(
        checkpoint,
        split_name,
        seed,
    )

    model = v9d.V9DStudent(
        beta=v9d.FIXED_BETA,
        dropout=0.30,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    criterion = nn.BCEWithLogitsLoss()

    df, metrics = predict_loader(
        model,
        loader,
        dataset,
        criterion,
        device,
    )

    df["seed"] = seed
    df["split"] = split_name
    df["dataset_role"] = role
    df["beta"] = v9d.FIXED_BETA
    df["lambda_align"] = v9d.FIXED_LAMBDA
    df["checkpoint_sha256"] = (
        v9d.EXPECTED_CHECKPOINT_SHA256[
            (split_name, seed)
        ]
    )

    out_dir = (
        OUT_ROOT
        / split_name
        / f"seed_{seed}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        out_dir
        / f"{role}_predictions.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    meta = {
        "analysis_type":
            "posthoc_secondary_prediction_export",

        "primary_model_changed":
            False,

        "retraining":
            False,

        "checkpoint_reselection":
            False,

        "threshold_tuning":
            False,

        "outer_test_used_for_model_selection":
            False,

        "split":
            split_name,

        "seed":
            seed,

        "dataset_role":
            role,

        "beta":
            v9d.FIXED_BETA,

        "lambda_align":
            v9d.FIXED_LAMBDA,

        "fixed_primary_threshold":
            v9d.FIXED_THRESHOLD,

        "checkpoint_sha256":
            v9d.EXPECTED_CHECKPOINT_SHA256[
                (split_name, seed)
            ],

        "metrics_recomputed_from_export":
            metrics,
    }

    meta_path = (
        out_dir
        / f"{role}_metadata.json"
    )

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
        )
    )

    print(
        f"{split_name:<14} "
        f"seed={seed:<4} "
        f"{role:<10} "
        f"N={len(df):<6} "
        f"AUROC={metrics['auroc']:.6f} "
        f"AUPR={metrics['aupr']:.6f} "
        f"F1={metrics['f1']:.6f}"
    )

    del model
    del checkpoint

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 100)
    print("V9-D POST-HOC PREDICTION EXPORT")
    print("=" * 100)
    print("Device           :", device)
    print("Beta             :", v9d.FIXED_BETA)
    print("Lambda           :", v9d.FIXED_LAMBDA)
    print("Primary threshold:", v9d.FIXED_THRESHOLD)
    print("Retraining       : NO")
    print("Model selection  : NO")
    print()

    # Critical: verify all 10 original checkpoint hashes.
    v9d.verify_all_checkpoints()

    graphs = v9d.load_graphs()

    summary = {}

    for split_name in v9d.SPLITS:

        print()
        print("=" * 100)
        print(split_name)
        print("=" * 100)

        morgan = v9d.load_morgan_svd(
            split_name
        )

        # Validation data
        val_pos, val_neg = (
            load_validation_pairs(
                split_name
            )
        )

        # Already-consumed outer test:
        # used only for secondary post-hoc analysis.
        test_pos, test_neg = (
            v9d.load_outer_test(
                split_name
            )
        )

        summary[split_name] = {}

        for seed in v9d.SEEDS:

            val_metrics = run_one_role(
                split_name,
                seed,
                "validation",
                val_pos,
                val_neg,
                graphs,
                morgan,
                device,
            )

            outer_metrics = run_one_role(
                split_name,
                seed,
                "outer",
                test_pos,
                test_neg,
                graphs,
                morgan,
                device,
            )

            summary[split_name][str(seed)] = {
                "validation": val_metrics,
                "outer": outer_metrics,
            }

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUT_ROOT
        / "posthoc_export_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 100)
    print("EXPORT COMPLETE")
    print("=" * 100)
    print("Saved under:")
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
