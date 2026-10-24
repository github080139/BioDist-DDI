#!/usr/bin/env python3

from pathlib import Path
import torch
import torch.nn.functional as F


ROOT = Path(
    "data/amddi_common_1087_leakage_free_features"
)

SPLIT_ROOT = Path(
    "data/amddi_common_1087_coldstart_dev_splits_v2"
)

SPLITS = (
    "cold_start_S1",
    "cold_start_S2",
)


def load(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    ).float()


def compare(name_a, a, name_b, b, train_ids):

    a = a[train_ids]
    b = b[train_ids]

    diff = a - b

    max_abs = float(
        diff.abs().max()
    )

    mean_abs = float(
        diff.abs().mean()
    )

    rmse = float(
        torch.sqrt(
            torch.mean(
                diff ** 2
            )
        )
    )

    exact = bool(
        torch.equal(a, b)
    )

    close_1e6 = bool(
        torch.allclose(
            a,
            b,
            atol=1e-6,
            rtol=1e-6,
        )
    )

    row_cos = F.cosine_similarity(
        a,
        b,
        dim=1,
        eps=1e-12,
    )

    mean_row_cos = float(
        row_cos.mean()
    )

    min_row_cos = float(
        row_cos.min()
    )

    flat_a = a.flatten()
    flat_b = b.flatten()

    flat_a_centered = (
        flat_a - flat_a.mean()
    )

    flat_b_centered = (
        flat_b - flat_b.mean()
    )

    denom = (
        torch.linalg.vector_norm(
            flat_a_centered
        )
        *
        torch.linalg.vector_norm(
            flat_b_centered
        )
    )

    pearson = float(
        torch.dot(
            flat_a_centered,
            flat_b_centered,
        )
        /
        denom.clamp_min(1e-12)
    )

    print()
    print(
        f"{name_a}  vs  {name_b}"
    )
    print("-" * 78)

    print(
        "exact_equal       :",
        exact,
    )

    print(
        "allclose_1e-6     :",
        close_1e6,
    )

    print(
        "max_abs_diff      :",
        f"{max_abs:.12g}",
    )

    print(
        "mean_abs_diff     :",
        f"{mean_abs:.12g}",
    )

    print(
        "RMSE              :",
        f"{rmse:.12g}",
    )

    print(
        "mean row cosine   :",
        f"{mean_row_cos:.12f}",
    )

    print(
        "min row cosine    :",
        f"{min_row_cos:.12f}",
    )

    print(
        "flattened Pearson :",
        f"{pearson:.12f}",
    )

    return {
        "exact": exact,
        "close": close_1e6,
        "mean_cos": mean_row_cos,
        "pearson": pearson,
    }


def main():

    print("=" * 88)
    print(
        "V9-D BIOLOGICAL REPRESENTATION REDUNDANCY AUDIT"
    )
    print("=" * 88)

    for split in SPLITS:

        print()
        print("=" * 88)
        print(split)
        print("=" * 88)

        train_ids = torch.load(
            SPLIT_ROOT
            / split
            / "inner_train_drugs.pt",
            map_location="cpu",
            weights_only=True,
        ).long().flatten()

        clean = load(
            ROOT
            / f"{split}_v9d_clean_dti_esm2_context_128.pt"
        )

        esm = load(
            ROOT
            / f"{split}_esm2_context_128.pt"
        )

        ppi = load(
            ROOT
            / f"{split}_ppi_esm2_context_128.pt"
        )

        for name, x in (
            ("CLEAN_DTI_ESM2", clean),
            ("ESM2_CONTEXT", esm),
            ("PPI_ESM2", ppi),
        ):

            if tuple(x.shape) != (
                1087,
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

            print(
                f"{name:16s} nonzero rows:",
                int(nz.sum()),
            )

            if int(nz.sum()) != 685:
                raise RuntimeError(
                    f"{name}: expected 685 nonzero rows"
                )

        r1 = compare(
            "CLEAN_DTI_ESM2",
            clean,
            "ESM2_CONTEXT",
            esm,
            train_ids,
        )

        r2 = compare(
            "CLEAN_DTI_ESM2",
            clean,
            "PPI_ESM2",
            ppi,
            train_ids,
        )

        r3 = compare(
            "ESM2_CONTEXT",
            esm,
            "PPI_ESM2",
            ppi,
            train_ids,
        )

        print()
        print("INTERPRETATION")
        print("-" * 78)

        if r1["close"]:
            print(
                "CLEAN_DTI_ESM2 and ESM2_CONTEXT are duplicate "
                "modalities at numerical tolerance."
            )
        elif (
            r1["mean_cos"] > 0.999
            and r1["pearson"] > 0.999
        ):
            print(
                "CLEAN_DTI_ESM2 and ESM2_CONTEXT are effectively "
                "redundant."
            )
        else:
            print(
                "CLEAN_DTI_ESM2 and ESM2_CONTEXT contain "
                "measurably different information."
            )

        if (
            r2["mean_cos"] > 0.999
            and r2["pearson"] > 0.999
        ):
            print(
                "PPI propagation adds very little numerical change "
                "relative to clean DTI/ESM2."
            )
        else:
            print(
                "PPI_ESM2 is measurably distinct from the "
                "non-PPI representation."
            )

    print()
    print("=" * 88)
    print(
        "PASS: V9-D BIOLOGICAL REDUNDANCY AUDIT COMPLETE"
    )
    print(
        "OUTER TEST PAIRS: NOT LOADED / SEALED"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
