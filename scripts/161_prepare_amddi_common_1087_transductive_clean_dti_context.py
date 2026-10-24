#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

import torch


NUM_DRUGS = 1087
DIM = 128

ROOT = Path.cwd()

SPLIT_ROOT = (
    ROOT
    / "data/amddi_common_splits/transductive"
)

BIO_ROOT = (
    ROOT
    / "data/amddi_common_1087_biological_dev_v4"
)

OUT_ROOT = (
    ROOT
    / "data/amddi_common_1087_leakage_free_features"
)

OUT_PATH = (
    OUT_ROOT
    / "transductive_v9d_clean_dti_esm2_context_128_train1071.pt"
)

META_PATH = (
    OUT_ROOT
    / "transductive_v9d_clean_dti_esm2_context_128_train1071.json"
)


def load(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
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


def normalize_pairs(x):
    x = x.long()

    if x.ndim != 2:
        raise RuntimeError(
            f"Bad pair tensor: {tuple(x.shape)}"
        )

    if x.shape[1] == 2:
        return x.contiguous()

    if x.shape[0] == 2:
        return x.t().contiguous()

    raise RuntimeError(
        f"Cannot interpret pair tensor: {tuple(x.shape)}"
    )


def main():

    print("=" * 76)
    print(
        "TRANSDUCTIVE CLEAN DTI/ESM2 CONTEXT "
        "— TRAIN-1071 MASK"
    )
    print("=" * 76)

    if OUT_PATH.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {OUT_PATH}"
        )

    if META_PATH.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {META_PATH}"
        )

    train_pos_path = (
        SPLIT_ROOT / "train_pos.pt"
    )

    train_neg_path = (
        SPLIT_ROOT / "train_neg.pt"
    )

    val_pos_path = (
        SPLIT_ROOT / "val_pos.pt"
    )

    val_neg_path = (
        SPLIT_ROOT / "val_neg.pt"
    )

    dti_path = (
        BIO_ROOT / "master_dti_edge_index.pt"
    )

    esm_path = (
        BIO_ROOT / "protein_features_esm2_128.pt"
    )

    train_pos = normalize_pairs(
        load(train_pos_path)
    )

    train_neg = normalize_pairs(
        load(train_neg_path)
    )

    val_pos = normalize_pairs(
        load(val_pos_path)
    )

    val_neg = normalize_pairs(
        load(val_neg_path)
    )

    train_drugs = torch.unique(
        torch.cat(
            [
                train_pos.reshape(-1),
                train_neg.reshape(-1),
            ]
        )
    ).sort().values

    val_drugs = torch.unique(
        torch.cat(
            [
                val_pos.reshape(-1),
                val_neg.reshape(-1),
            ]
        )
    ).sort().values

    if train_drugs.numel() != 1071:
        raise RuntimeError(
            f"Expected 1071 training drugs, "
            f"got {train_drugs.numel()}"
        )

    if val_drugs.numel() != 1071:
        raise RuntimeError(
            f"Expected 1071 validation drugs, "
            f"got {val_drugs.numel()}"
        )

    if not torch.equal(
        train_drugs,
        val_drugs,
    ):
        raise RuntimeError(
            "Transductive train/validation "
            "drug universes differ"
        )

    dti = load(
        dti_path
    ).long()

    protein_x = load(
        esm_path
    ).float()

    if (
        dti.ndim != 2
        or dti.shape[0] != 2
    ):
        raise RuntimeError(
            f"Bad DTI shape: {tuple(dti.shape)}"
        )

    if (
        protein_x.ndim != 2
        or protein_x.shape[1] != DIM
    ):
        raise RuntimeError(
            f"Bad ESM2 shape: {tuple(protein_x.shape)}"
        )

    if not torch.isfinite(
        protein_x
    ).all():
        raise RuntimeError(
            "Non-finite ESM2 values"
        )

    master_drug_idx = dti[0]
    master_protein_idx = dti[1]

    if int(master_drug_idx.min()) < 0:
        raise RuntimeError(
            "Negative drug index"
        )

    if int(master_drug_idx.max()) >= NUM_DRUGS:
        raise RuntimeError(
            "Drug index outside Common-1087"
        )

    if int(master_protein_idx.min()) < 0:
        raise RuntimeError(
            "Negative protein index"
        )

    if (
        int(master_protein_idx.max())
        >= protein_x.shape[0]
    ):
        raise RuntimeError(
            "Protein index outside ESM2 matrix"
        )

    # ---------------------------------------------------------
    # Frozen transductive permitted universe:
    # drugs occurring in TRAINING DDI pairs.
    #
    # Validation contains the same drug identities, but
    # validation DDI edges are not used here.
    # ---------------------------------------------------------

    allowed = torch.zeros(
        NUM_DRUGS,
        dtype=torch.bool,
    )

    allowed[train_drugs] = True

    keep = allowed[
        master_drug_idx
    ]

    drug_idx = master_drug_idx[
        keep
    ]

    protein_idx = master_protein_idx[
        keep
    ]

    if not bool(
        allowed[drug_idx].all()
    ):
        raise RuntimeError(
            "Filtered DTI contains endpoint "
            "outside transductive training universe"
        )

    # ---------------------------------------------------------
    # Exact clean V9-D aggregation semantics:
    # mean ESM2 target embedding over permitted DTI targets.
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
        row_nonzero[
            train_drugs
        ].sum()
    )

    n_total = int(
        row_nonzero.sum()
    )

    outside = (
        row_nonzero
        & ~allowed
    )

    absent_drugs = torch.where(
        ~allowed
    )[0]

    print(
        "Master DTI edges      :",
        int(dti.shape[1]),
    )

    print(
        "Filtered DTI edges    :",
        int(drug_idx.numel()),
    )

    print(
        "Training drugs        :",
        int(train_drugs.numel()),
    )

    print(
        "Validation drugs      :",
        int(val_drugs.numel()),
    )

    print(
        "Total nonzero drugs   :",
        n_total,
    )

    print(
        "Train nonzero         :",
        n_train,
        "/",
        int(train_drugs.numel()),
    )

    print(
        "Outside-train nonzero :",
        int(outside.sum()),
    )

    print(
        "Absent drug IDs       :",
        absent_drugs.tolist(),
    )

    if n_total != 1071:
        raise RuntimeError(
            f"Expected exactly 1071 "
            f"nonzero rows, got {n_total}"
        )

    if n_train != 1071:
        raise RuntimeError(
            "Not all 1071 transductive "
            "training drugs have clean DTI context"
        )

    if bool(
        outside.any()
    ):
        raise RuntimeError(
            "Biological context exists outside "
            "transductive training universe"
        )

    if not torch.all(
        context[absent_drugs] == 0
    ):
        raise RuntimeError(
            "One or more absent drugs are nonzero"
        )

    if not torch.isfinite(
        context
    ).all():
        raise RuntimeError(
            "Non-finite output context"
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        context.contiguous(),
        OUT_PATH,
    )

    metadata = {
        "version":
            "V9-D_transductive_clean_train1071_v1",

        "setting":
            "transductive",

        "source":
            "biological_dev_v4/master_dti_plus_esm2",

        "aggregation":
            "mean_ESM2_over_training_universe_DTI_targets",

        "num_drugs":
            NUM_DRUGS,

        "dimension":
            DIM,

        "master_dti_edges":
            int(dti.shape[1]),

        "filtered_dti_edges":
            int(drug_idx.numel()),

        "training_drugs":
            int(train_drugs.numel()),

        "validation_drugs":
            int(val_drugs.numel()),

        "train_validation_drug_sets_identical":
            True,

        "nonzero_drugs":
            n_total,

        "outside_training_nonzero":
            int(outside.sum()),

        "absent_drug_ids":
            absent_drugs.tolist(),

        "validation_ddi_edges_used":
            False,

        "outer_test_pairs_loaded":
            False,

        "old_dti_bio_128_used":
            False,

        "ppi_used":
            False,

        "master_dti_sha256":
            sha256(dti_path),

        "protein_features_sha256":
            sha256(esm_path),

        "train_pos_sha256":
            sha256(train_pos_path),

        "train_neg_sha256":
            sha256(train_neg_path),

        "val_pos_sha256":
            sha256(val_pos_path),

        "val_neg_sha256":
            sha256(val_neg_path),
    }

    META_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "Saved feature :",
        OUT_PATH,
    )

    print(
        "Saved metadata:",
        META_PATH,
    )

    print()
    print(
        "Feature SHA256 :",
        sha256(OUT_PATH),
    )

    print(
        "Metadata SHA256:",
        sha256(META_PATH),
    )

    print()
    print(
        "PASS: TRANSDUCTIVE CLEAN "
        "DTI/ESM2 TRAIN-1071 FEATURE"
    )

    print(
        "VALIDATION DDI EDGES USED: NO"
    )

    print(
        "OUTER TEST LOADED: NO"
    )

    print(
        "PPI USED: NO"
    )


if __name__ == "__main__":
    main()
