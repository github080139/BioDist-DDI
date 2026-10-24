#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    "results_final/amddi_common_1087/"
    "v9D_ppi_teacher_distillation/"
    "POSTHOC_PREDICTIONS"
)

OUT = ROOT / "ANALYSIS"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [
    42,
    123,
    456,
    789,
    2026,
]

SPLITS = [
    "cold_start_S1",
    "cold_start_S2",
]


for split in SPLITS:

    dfs = []

    for seed in SEEDS:

        p = (
            ROOT
            / split
            / f"seed_{seed}"
            / "outer_predictions.csv"
        )

        if not p.is_file():
            raise FileNotFoundError(
                f"Missing prediction file: {p}"
            )

        df = pd.read_csv(p)

        needed = {
            "drug_a",
            "drug_b",
            "label",
            "probability",
        }

        missing = needed - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{p} missing columns: {missing}"
            )

        df = df[
            [
                "drug_a",
                "drug_b",
                "label",
                "probability",
            ]
        ].copy()

        df = df.rename(
            columns={
                "probability":
                    f"prob_seed_{seed}"
            }
        )

        dfs.append(df)

    merged = dfs[0]

    for df in dfs[1:]:

        merged = merged.merge(
            df,
            on=[
                "drug_a",
                "drug_b",
                "label",
            ],
            how="inner",
            validate="one_to_one",
        )

    prob_cols = [
        f"prob_seed_{seed}"
        for seed in SEEDS
    ]

    merged[
        "mean_probability_5seeds"
    ] = (
        merged[prob_cols]
        .mean(axis=1)
    )

    merged[
        "sd_probability_5seeds"
    ] = (
        merged[prob_cols]
        .std(
            axis=1,
            ddof=1,
        )
    )

    # ------------------------------------------------------
    # CASE GROUP 1:
    # Highest-scoring benchmark-positive pairs
    # ------------------------------------------------------

    top_positive = (
        merged[
            merged["label"] == 1
        ]
        .sort_values(
            "mean_probability_5seeds",
            ascending=False,
        )
        .head(2)
        .copy()
    )

    top_positive[
        "case_category"
    ] = "high_score_true_positive"

    # ------------------------------------------------------
    # CASE GROUP 2:
    # Benchmark-positive pairs with the lowest scores
    # These are the strongest missed positives.
    # ------------------------------------------------------

    missed_positive = (
        merged[
            merged["label"] == 1
        ]
        .sort_values(
            "mean_probability_5seeds",
            ascending=True,
        )
        .head(2)
        .copy()
    )

    missed_positive[
        "case_category"
    ] = "strongest_missed_positive"

    # ------------------------------------------------------
    # CASE GROUP 3:
    # Highest-scoring benchmark-negative pairs
    #
    # IMPORTANT:
    # These are NOT automatically clinically false
    # interactions. They are only negative under the
    # benchmark label source.
    # ------------------------------------------------------

    high_negative = (
        merged[
            merged["label"] == 0
        ]
        .sort_values(
            "mean_probability_5seeds",
            ascending=False,
        )
        .head(2)
        .copy()
    )

    high_negative[
        "case_category"
    ] = "high_score_benchmark_negative"

    cases = pd.concat(
        [
            top_positive,
            missed_positive,
            high_negative,
        ],
        ignore_index=True,
    )

    cases.insert(
        0,
        "split",
        split,
    )

    all_pairs_file = (
        OUT
        / f"{split}_all_pairs_5seed_mean_scores.csv"
    )

    cases_file = (
        OUT
        / f"{split}_six_case_candidates.csv"
    )

    merged.to_csv(
        all_pairs_file,
        index=False,
    )

    cases.to_csv(
        cases_file,
        index=False,
    )

    print()
    print("=" * 100)
    print(split)
    print("=" * 100)

    print(
        cases[
            [
                "case_category",
                "drug_a",
                "drug_b",
                "label",
                "mean_probability_5seeds",
                "sd_probability_5seeds",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(cases_file)
    print(all_pairs_file)


print()
print("=" * 100)
print("CASE-CANDIDATE EXTRACTION COMPLETE")
print("=" * 100)
