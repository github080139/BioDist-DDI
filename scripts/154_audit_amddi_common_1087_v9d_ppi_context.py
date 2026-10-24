#!/usr/bin/env python3

from pathlib import Path
import hashlib
import torch


ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROOT = (
    ROOT
    / "data/amddi_common_1087_leakage_free_features"
)

SPLIT_ROOT = (
    ROOT
    / "data/amddi_common_1087_coldstart_dev_splits_v2"
)

SPLITS = (
    "cold_start_S1",
    "cold_start_S2",
)

NUM_DRUGS = 1087
DIM = 128


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_ids(path):
    x = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    return x.long().flatten()


def main():

    print("=" * 90)
    print("V9-D PPI/ESM2 COLD-START LEAKAGE AUDIT")
    print("=" * 90)

    total_pass = 0
    total_fail = 0

    for split in SPLITS:

        print()
        print("-" * 90)
        print(split)
        print("-" * 90)

        feature_path = (
            FEATURE_ROOT
            / f"{split}_ppi_esm2_context_128.pt"
        )

        x = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=True,
        ).float()

        split_dir = (
            SPLIT_ROOT
            / split
        )

        train_ids = load_ids(
            split_dir / "inner_train_drugs.pt"
        )

        val_ids = load_ids(
            split_dir / "inner_val_drugs.pt"
        )

        heldout_ids = load_ids(
            split_dir / "outer_heldout_drugs.pt"
        )

        checks = []

        checks.append((
            "feature_shape",
            tuple(x.shape) == (NUM_DRUGS, DIM),
            tuple(x.shape),
        ))

        checks.append((
            "no_nan",
            not bool(torch.isnan(x).any()),
            bool(torch.isnan(x).any()),
        ))

        checks.append((
            "train_count_685",
            train_ids.numel() == 685,
            train_ids.numel(),
        ))

        checks.append((
            "val_count_171",
            val_ids.numel() == 171,
            val_ids.numel(),
        ))

        checks.append((
            "outer_heldout_count_215",
            heldout_ids.numel() == 215,
            heldout_ids.numel(),
        ))

        val_absmax = (
            float(
                x[val_ids]
                .abs()
                .max()
                .item()
            )
            if val_ids.numel()
            else 0.0
        )

        heldout_absmax = (
            float(
                x[heldout_ids]
                .abs()
                .max()
                .item()
            )
            if heldout_ids.numel()
            else 0.0
        )

        checks.append((
            "inner_val_context_exact_zero",
            val_absmax == 0.0,
            val_absmax,
        ))

        checks.append((
            "outer_heldout_context_exact_zero",
            heldout_absmax == 0.0,
            heldout_absmax,
        ))

        row_nonzero = (
            x.abs().sum(dim=1) > 0
        )

        nonzero_ids = torch.where(
            row_nonzero
        )[0]

        train_mask = torch.zeros(
            NUM_DRUGS,
            dtype=torch.bool,
        )

        train_mask[train_ids] = True

        nonzero_outside_train = nonzero_ids[
            ~train_mask[nonzero_ids]
        ]

        checks.append((
            "all_nonzero_context_inside_inner_train",
            nonzero_outside_train.numel() == 0,
            int(nonzero_outside_train.numel()),
        ))

        checks.append((
            "nonzero_drug_count_685",
            nonzero_ids.numel() == 685,
            int(nonzero_ids.numel()),
        ))

        train_zero_ids = train_ids[
            ~row_nonzero[train_ids]
        ]

        checks.append((
            "all_inner_train_drugs_have_context",
            train_zero_ids.numel() == 0,
            int(train_zero_ids.numel()),
        ))

        train_set = set(
            train_ids.tolist()
        )

        val_set = set(
            val_ids.tolist()
        )

        heldout_set = set(
            heldout_ids.tolist()
        )

        checks.append((
            "train_val_disjoint",
            train_set.isdisjoint(val_set),
            len(train_set & val_set),
        ))

        checks.append((
            "train_outerheldout_disjoint",
            train_set.isdisjoint(heldout_set),
            len(train_set & heldout_set),
        ))

        print("SHA256:", sha256(feature_path))
        print("Nonzero drugs:", nonzero_ids.numel())
        print("Val abs max:", val_absmax)
        print("Outer-heldout abs max:", heldout_absmax)

        for name, passed, value in checks:

            status = (
                "PASS"
                if passed
                else "FAIL"
            )

            print(
                f"{status:4s} | "
                f"{name:42s} | "
                f"{value}"
            )

            if passed:
                total_pass += 1
            else:
                total_fail += 1

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("PASS:", total_pass)
    print("FAIL:", total_fail)
    print("Outer test pairs: NOT LOADED / SEALED")

    if total_fail:
        raise SystemExit(
            "V9-D AUDIT FAILED"
        )

    print(
        "V9-D PPI/ESM2 CONTEXT CERTIFIED FOR DEVELOPMENT"
    )


if __name__ == "__main__":
    main()
