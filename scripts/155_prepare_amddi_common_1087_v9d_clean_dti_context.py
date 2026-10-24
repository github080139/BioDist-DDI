#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import torch


NUM_DRUGS = 1087
DIM = 128

ROOT = Path("data/amddi_common_1087_biological_dev_v4")
OUT_ROOT = Path(
    "data/amddi_common_1087_leakage_free_features"
)
SPLIT_ROOT = Path(
    "data/amddi_common_1087_coldstart_dev_splits_v2"
)

SPLITS = (
    "cold_start_S1",
    "cold_start_S2",
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_tensor(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def main():

    print("=" * 88)
    print("V9-D CLEAN DTI/ESM2 CONTEXT PREPARATION")
    print("=" * 88)

    protein_x = load_tensor(
        ROOT / "protein_features_esm2_128.pt"
    ).float()

    if protein_x.ndim != 2:
        raise RuntimeError(
            f"Bad protein tensor shape: {protein_x.shape}"
        )

    if protein_x.shape[1] != DIM:
        raise RuntimeError(
            f"Expected protein dimension {DIM}; "
            f"got {protein_x.shape[1]}"
        )

    if torch.isnan(protein_x).any():
        raise RuntimeError(
            "Protein ESM2 tensor contains NaN"
        )

    print(
        "ESM2 protein features:",
        tuple(protein_x.shape),
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in SPLITS:

        print()
        print("-" * 88)
        print(split)
        print("-" * 88)

        split_bio = ROOT / split
        split_ids = SPLIT_ROOT / split

        dti = load_tensor(
            split_bio / "dti_edge_index.pt"
        ).long()

        inner_train = load_tensor(
            split_ids / "inner_train_drugs.pt"
        ).long().flatten()

        inner_val = load_tensor(
            split_ids / "inner_val_drugs.pt"
        ).long().flatten()

        outer = load_tensor(
            split_ids / "outer_heldout_drugs.pt"
        ).long().flatten()

        if (
            dti.ndim != 2
            or dti.shape[0] != 2
        ):
            raise RuntimeError(
                f"Bad DTI shape: {tuple(dti.shape)}"
            )

        drug_idx = dti[0]
        protein_idx = dti[1]

        if int(drug_idx.min()) < 0:
            raise RuntimeError(
                "Negative drug index"
            )

        if int(drug_idx.max()) >= NUM_DRUGS:
            raise RuntimeError(
                "Drug index outside Common-1087"
            )

        if int(protein_idx.min()) < 0:
            raise RuntimeError(
                "Negative protein index"
            )

        if int(protein_idx.max()) >= protein_x.shape[0]:
            raise RuntimeError(
                "Protein index outside ESM2 matrix"
            )

        allowed = torch.zeros(
            NUM_DRUGS,
            dtype=torch.bool,
        )

        allowed[inner_train] = True

        if not bool(
            allowed[drug_idx].all()
        ):
            raise RuntimeError(
                f"{split}: raw DTI contains "
                "non-inner-training drug endpoint"
            )

        # ---------------------------------------------------------
        # Aggregate ESM2 target embeddings over CLEAN raw v4 DTI.
        #
        # This is deliberately reconstructed from the audited raw
        # split-specific DTI instead of using the old dti_bio_128.pt.
        # ---------------------------------------------------------

        context = torch.zeros(
            NUM_DRUGS,
            DIM,
            dtype=torch.float32,
        )

        counts = torch.zeros(
            NUM_DRUGS,
            dtype=torch.float32,
        )

        context.index_add_(
            0,
            drug_idx,
            protein_x[protein_idx],
        )

        counts.index_add_(
            0,
            drug_idx,
            torch.ones(
                drug_idx.shape[0],
                dtype=torch.float32,
            ),
        )

        has_dti = counts > 0

        context[has_dti] = (
            context[has_dti]
            /
            counts[has_dti].unsqueeze(1)
        )

        row_nonzero = (
            context.abs().sum(dim=1) > 0
        )

        n_train = int(
            row_nonzero[inner_train].sum()
        )

        n_val = int(
            row_nonzero[inner_val].sum()
        )

        n_outer = int(
            row_nonzero[outer].sum()
        )

        n_total = int(
            row_nonzero.sum()
        )

        print(
            "DTI edges             :",
            dti.shape[1],
        )
        print(
            "Total nonzero drugs   :",
            n_total,
        )
        print(
            "Inner-train nonzero   :",
            n_train,
            "/",
            inner_train.numel(),
        )
        print(
            "Inner-val nonzero     :",
            n_val,
            "/",
            inner_val.numel(),
        )
        print(
            "Outer-heldout nonzero :",
            n_outer,
            "/",
            outer.numel(),
        )

        if n_total != 685:
            raise RuntimeError(
                f"{split}: expected exactly 685 "
                f"nonzero rows, got {n_total}"
            )

        if n_train != 685:
            raise RuntimeError(
                f"{split}: not all 685 training drugs "
                "have clean DTI context"
            )

        if n_val != 0:
            raise RuntimeError(
                f"{split}: validation exposure detected"
            )

        if n_outer != 0:
            raise RuntimeError(
                f"{split}: outer-heldout exposure detected"
            )

        outside = row_nonzero.clone()
        outside[inner_train] = False

        if bool(outside.any()):
            raise RuntimeError(
                f"{split}: nonzero context exists outside "
                "inner-training universe"
            )

        if torch.isnan(context).any():
            raise RuntimeError(
                f"{split}: NaN in output"
            )

        out_path = (
            OUT_ROOT
            /
            f"{split}_v9d_clean_dti_esm2_context_128.pt"
        )

        torch.save(
            context.contiguous(),
            out_path,
        )

        metadata = {
            "version":
                "V9-D",

            "split":
                split,

            "source":
                "biological_dev_v4/raw_dti_plus_esm2",

            "aggregation":
                "mean_ESM2_over_split_specific_DTI_targets",

            "num_drugs":
                NUM_DRUGS,

            "dimension":
                DIM,

            "dti_edges":
                int(dti.shape[1]),

            "nonzero_drugs":
                n_total,

            "inner_train_nonzero":
                n_train,

            "inner_val_nonzero":
                n_val,

            "outer_heldout_nonzero":
                n_outer,

            "validation_biology_used":
                False,

            "outer_test_pairs_loaded":
                False,
        }

        meta_path = out_path.with_suffix(
            ".json"
        )

        meta_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n"
        )

        print(
            "Saved:",
            out_path,
        )
        print(
            "SHA256:",
            sha256(out_path),
        )

    print()
    print("=" * 88)
    print(
        "PASS: V9-D CLEAN DTI/ESM2 CONTEXT PREPARED"
    )
    print(
        "INNER VALIDATION BIOLOGY: NOT USED"
    )
    print(
        "OUTER TEST PAIRS: NOT LOADED / SEALED"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
