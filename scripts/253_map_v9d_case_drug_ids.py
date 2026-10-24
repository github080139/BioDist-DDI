#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(".").resolve()

MANIFEST = (
    ROOT
    / "data/amddi_common_1087_structure"
    / "structure_manifest.csv"
)

CASE_ROOT = (
    ROOT
    / "results_final/amddi_common_1087"
    / "v9D_ppi_teacher_distillation"
    / "POSTHOC_PREDICTIONS"
    / "ANALYSIS"
)

SPLITS = [
    "cold_start_S1",
    "cold_start_S2",
]

if not MANIFEST.is_file():
    raise FileNotFoundError(
        f"Missing manifest: {MANIFEST}"
    )

manifest = pd.read_csv(MANIFEST)

print("=" * 100)
print("STRUCTURE MANIFEST")
print("=" * 100)
print("Shape:", manifest.shape)
print("Columns:")
for c in manifest.columns:
    print("  -", c)

print()
print("First rows:")
print(manifest.head().to_string(index=False))

# ---------------------------------------------------------
# Detect canonical numeric ID column
# ---------------------------------------------------------

candidate_id_cols = [
    "node_id",
    "drug_id",
    "canonical_id",
    "id",
    "index",
]

id_col = None

for c in candidate_id_cols:
    if c in manifest.columns:
        id_col = c
        break

if id_col is None:
    raise RuntimeError(
        "Could not identify canonical node-ID column."
    )

manifest[id_col] = (
    pd.to_numeric(
        manifest[id_col],
        errors="raise"
    )
    .astype(int)
)

# ---------------------------------------------------------
# Keep useful human-readable identifier fields
# ---------------------------------------------------------

preferred_cols = [
    id_col,
    "drugbank_id",
    "drugbank",
    "DrugBank_ID",
    "drug_name",
    "name",
    "generic_name",
    "canonical_name",
    "smiles",
    "SMILES",
]

keep_cols = []

for c in preferred_cols:
    if (
        c in manifest.columns
        and c not in keep_cols
    ):
        keep_cols.append(c)

# If preferred fields are not present, keep all manifest cols.
if len(keep_cols) <= 1:
    keep_cols = list(manifest.columns)

lookup = (
    manifest[keep_cols]
    .drop_duplicates(
        subset=[id_col]
    )
    .copy()
)

for split in SPLITS:

    case_file = (
        CASE_ROOT
        / f"{split}_six_case_candidates.csv"
    )

    if not case_file.is_file():
        raise FileNotFoundError(
            f"Missing case file: {case_file}"
        )

    cases = pd.read_csv(case_file)

    # Join Drug A
    a_lookup = lookup.copy()

    a_lookup = a_lookup.rename(
        columns={
            c: (
                "drug_a"
                if c == id_col
                else f"drug_a_{c}"
            )
            for c in a_lookup.columns
        }
    )

    mapped = cases.merge(
        a_lookup,
        on="drug_a",
        how="left",
        validate="many_to_one",
    )

    # Join Drug B
    b_lookup = lookup.copy()

    b_lookup = b_lookup.rename(
        columns={
            c: (
                "drug_b"
                if c == id_col
                else f"drug_b_{c}"
            )
            for c in b_lookup.columns
        }
    )

    mapped = mapped.merge(
        b_lookup,
        on="drug_b",
        how="left",
        validate="many_to_one",
    )

    out_file = (
        CASE_ROOT
        / f"{split}_six_case_candidates_mapped.csv"
    )

    mapped.to_csv(
        out_file,
        index=False,
    )

    print()
    print("=" * 100)
    print(split)
    print("=" * 100)

    display_cols = [
        "case_category",
        "drug_a",
    ]

    for c in mapped.columns:
        if (
            c.startswith("drug_a_")
            and (
                "name" in c.lower()
                or "drugbank" in c.lower()
            )
        ):
            display_cols.append(c)

    display_cols.append("drug_b")

    for c in mapped.columns:
        if (
            c.startswith("drug_b_")
            and (
                "name" in c.lower()
                or "drugbank" in c.lower()
            )
        ):
            display_cols.append(c)

    display_cols += [
        "label",
        "mean_probability_5seeds",
        "sd_probability_5seeds",
    ]

    # remove accidental duplicates
    display_cols = list(
        dict.fromkeys(display_cols)
    )

    print(
        mapped[
            [
                c
                for c in display_cols
                if c in mapped.columns
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(out_file)

print()
print("=" * 100)
print("DRUG-ID MAPPING COMPLETE")
print("=" * 100)
