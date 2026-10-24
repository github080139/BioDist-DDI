#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
)

ROOT = Path(
    "results_final/amddi_common_1087/"
    "v9D_ppi_teacher_distillation/"
    "POSTHOC_PREDICTIONS"
)

OUT = ROOT / "ANALYSIS"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = [
    "cold_start_S1",
    "cold_start_S2",
]

SEEDS = [
    42,
    123,
    456,
    789,
    2026,
]


def ece(y, p, n_bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    total = len(y)
    score = 0.0

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (p >= lo)
                & (p <= hi)
            )
        else:
            mask = (
                (p >= lo)
                & (p < hi)
            )

        n = int(mask.sum())

        if n == 0:
            continue

        confidence = float(
            p[mask].mean()
        )

        observed = float(
            y[mask].mean()
        )

        score += (
            n / total
        ) * abs(
            observed - confidence
        )

    return float(score)


def threshold_metrics(
    y,
    p,
    threshold,
):
    pred = (
        p >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0, 1],
        ).ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else np.nan
    )

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def select_threshold_for_sensitivity(
    y,
    p,
    target,
):
    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0]),
                np.asarray(p, dtype=float),
                np.array([1.0]),
            ]
        )
    )

    best = None

    for t in thresholds:
        m = threshold_metrics(
            y,
            p,
            t,
        )

        if (
            m["sensitivity"]
            >= target
        ):
            candidate = (
                m["specificity"],
                float(t),
                m,
            )

            if (
                best is None
                or candidate[0] > best[0]
                or (
                    candidate[0]
                    == best[0]
                    and candidate[1] > best[1]
                )
            ):
                best = candidate

    if best is None:
        raise RuntimeError(
            f"No validation threshold reaches "
            f"sensitivity {target}"
        )

    return (
        float(best[1]),
        best[2],
    )


all_results = []

for split in SPLITS:

    print("\n" + "=" * 100)
    print(split)
    print("=" * 100)

    for seed in SEEDS:

        base = (
            ROOT
            / split
            / f"seed_{seed}"
        )

        val_file = (
            base
            / "validation_predictions.csv"
        )

        outer_file = (
            base
            / "outer_predictions.csv"
        )

        if not val_file.is_file():
            raise FileNotFoundError(
                f"Missing: {val_file}"
            )

        if not outer_file.is_file():
            raise FileNotFoundError(
                f"Missing: {outer_file}"
            )

        val = pd.read_csv(
            val_file
        )

        outer = pd.read_csv(
            outer_file
        )

        yv = (
            val["label"]
            .to_numpy()
            .astype(int)
        )

        zv = (
            val["logit"]
            .to_numpy()
            .astype(float)
        )

        pv = (
            val["probability"]
            .to_numpy()
            .astype(float)
        )

        yo = (
            outer["label"]
            .to_numpy()
            .astype(int)
        )

        zo = (
            outer["logit"]
            .to_numpy()
            .astype(float)
        )

        po = (
            outer["probability"]
            .to_numpy()
            .astype(float)
        )

        # -------------------------------------------------
        # Raw outer calibration
        # -------------------------------------------------

        raw_brier = (
            brier_score_loss(
                yo,
                po,
            )
        )

        raw_ece = ece(
            yo,
            po,
            n_bins=10,
        )

        # -------------------------------------------------
        # Platt scaling
        # FIT ON VALIDATION ONLY.
        # Applied unchanged to outer test.
        # -------------------------------------------------

        platt = LogisticRegression(
            solver="lbfgs",
            C=1e6,
            max_iter=5000,
        )

        platt.fit(
            zv.reshape(-1, 1),
            yv,
        )

        po_cal = (
            platt.predict_proba(
                zo.reshape(-1, 1)
            )[:, 1]
        )

        cal_brier = (
            brier_score_loss(
                yo,
                po_cal,
            )
        )

        cal_ece = ece(
            yo,
            po_cal,
            n_bins=10,
        )

        # -------------------------------------------------
        # Sensitivity-oriented operating points
        # thresholds selected on VALIDATION ONLY
        # -------------------------------------------------

        print(
            f"seed={seed:<4} | "
            f"raw Brier={raw_brier:.6f} | "
            f"raw ECE={raw_ece:.6f} | "
            f"Platt Brier={cal_brier:.6f} | "
            f"Platt ECE={cal_ece:.6f}"
        )

        for target in [
            0.80,
            0.90,
        ]:

            threshold, val_op = (
                select_threshold_for_sensitivity(
                    yv,
                    pv,
                    target,
                )
            )

            outer_op = (
                threshold_metrics(
                    yo,
                    po,
                    threshold,
                )
            )

            print(
                f"    target={target:.2f} | "
                f"val threshold={threshold:.6f} | "
                f"val sens={val_op['sensitivity']:.6f} | "
                f"val spec={val_op['specificity']:.6f} | "
                f"outer sens={outer_op['sensitivity']:.6f} | "
                f"outer spec={outer_op['specificity']:.6f} | "
                f"outer prec={outer_op['precision']:.6f}"
            )

            all_results.append(
                {
                    "split":
                        split,

                    "seed":
                        seed,

                    "target_validation_sensitivity":
                        target,

                    "validation_selected_threshold":
                        threshold,

                    "validation_sensitivity":
                        val_op["sensitivity"],

                    "validation_specificity":
                        val_op["specificity"],

                    "outer_sensitivity":
                        outer_op["sensitivity"],

                    "outer_specificity":
                        outer_op["specificity"],

                    "outer_precision":
                        outer_op["precision"],

                    "outer_raw_brier":
                        float(raw_brier),

                    "outer_raw_ece10":
                        float(raw_ece),

                    "outer_platt_brier":
                        float(cal_brier),

                    "outer_platt_ece10":
                        float(cal_ece),

                    "platt_coefficient":
                        float(
                            platt.coef_[0, 0]
                        ),

                    "platt_intercept":
                        float(
                            platt.intercept_[0]
                        ),
                }
            )


df = pd.DataFrame(
    all_results
)

seedwise_file = (
    OUT
    / "calibration_operating_points_seedwise.csv"
)

df.to_csv(
    seedwise_file,
    index=False,
)


summary_rows = []

for split in SPLITS:

    dsplit = df[
        df["split"] == split
    ]

    dcal = dsplit[
        dsplit[
            "target_validation_sensitivity"
        ] == 0.90
    ]

    for metric in [
        "outer_raw_brier",
        "outer_raw_ece10",
        "outer_platt_brier",
        "outer_platt_ece10",
    ]:

        vals = (
            dcal[metric]
            .to_numpy()
            .astype(float)
        )

        summary_rows.append(
            {
                "split":
                    split,

                "analysis":
                    "calibration",

                "target":
                    metric,

                "mean":
                    float(
                        np.mean(vals)
                    ),

                "sd":
                    float(
                        np.std(
                            vals,
                            ddof=1,
                        )
                    ),
            }
        )

    for target in [
        0.80,
        0.90,
    ]:

        dop = dsplit[
            dsplit[
                "target_validation_sensitivity"
            ] == target
        ]

        for metric in [
            "validation_selected_threshold",
            "outer_sensitivity",
            "outer_specificity",
            "outer_precision",
        ]:

            vals = (
                dop[metric]
                .to_numpy()
                .astype(float)
            )

            summary_rows.append(
                {
                    "split":
                        split,

                    "analysis":
                        "operating_point",

                    "target":
                        (
                            f"{target:.2f}_"
                            f"{metric}"
                        ),

                    "mean":
                        float(
                            np.mean(vals)
                        ),

                    "sd":
                        float(
                            np.std(
                                vals,
                                ddof=1,
                            )
                        ),
                }
            )


summary = pd.DataFrame(
    summary_rows
)

summary_file = (
    OUT
    / "calibration_operating_points_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

for _, r in summary.iterrows():

    print(
        f"{r['split']:<14} "
        f"{r['target']:<45} "
        f"{r['mean']:.6f} ± "
        f"{r['sd']:.6f}"
    )

print()
print("Saved:")
print(seedwise_file)
print(summary_file)
