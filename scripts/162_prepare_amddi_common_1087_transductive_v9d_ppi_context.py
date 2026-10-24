#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

import torch


NUM_DRUGS = 1087
NUM_PROTEINS = 2321
DIM = 128

ROOT = Path(__file__).resolve().parents[1]

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
    / "transductive_ppi_esm2_context_128_train1071.pt"
)

META_PATH = (
    OUT_ROOT
    / "transductive_ppi_esm2_context_128_train1071_metadata.json"
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
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

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
        f"Cannot interpret pairs: {tuple(x.shape)}"
    )


def main():

    print("=" * 80)
    print(
        "TRANSDUCTIVE V9-D PPI/ESM2 CONTEXT "
        "— TRAIN-1071 MASK"
    )
    print("=" * 80)

    if OUT_PATH.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {OUT_PATH}"
        )

    if META_PATH.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {META_PATH}"
        )

    train_pos_path = SPLIT_ROOT / "train_pos.pt"
    train_neg_path = SPLIT_ROOT / "train_neg.pt"
    val_pos_path = SPLIT_ROOT / "val_pos.pt"
    val_neg_path = SPLIT_ROOT / "val_neg.pt"

    dti_path = BIO_ROOT / "master_dti_edge_index.pt"
    esm_path = BIO_ROOT / "protein_features_esm2_128.pt"
    ppi_path = BIO_ROOT / "ppi_edge_index.pt"

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
            "Train/validation drug universes differ"
        )

    esm = load(
        esm_path
    ).float()

    ppi = load(
        ppi_path
    ).long()

    dti = load(
        dti_path
    ).long()

    if tuple(esm.shape) != (
        NUM_PROTEINS,
        DIM,
    ):
        raise RuntimeError(
            f"Unexpected ESM2 shape: {tuple(esm.shape)}"
        )

    if (
        ppi.ndim != 2
        or ppi.shape[0] != 2
    ):
        raise RuntimeError(
            f"Unexpected PPI shape: {tuple(ppi.shape)}"
        )

    if (
        dti.ndim != 2
        or dti.shape[0] != 2
    ):
        raise RuntimeError(
            f"Unexpected DTI shape: {tuple(dti.shape)}"
        )

    if not torch.isfinite(esm).all():
        raise RuntimeError(
            "Non-finite ESM2 values"
        )

    if int(ppi.min()) < 0:
        raise RuntimeError(
            "Negative PPI protein index"
        )

    if int(ppi.max()) >= NUM_PROTEINS:
        raise RuntimeError(
            "PPI protein index out of range"
        )

    # --------------------------------------------------------
    # EXACT V9-D PPI PROPAGATION SEMANTICS
    #
    # neighbor_mean =
    #     mean incoming-neighbour ESM2
    #
    # propagated =
    #     0.5 * own ESM2
    #     + 0.5 * neighbor_mean
    #
    # Proteins without neighbours retain own ESM2.
    # --------------------------------------------------------

    src = ppi[0]
    dst = ppi[1]

    neighbor_sum = torch.zeros_like(
        esm
    )

    neighbor_sum.index_add_(
        0,
        dst,
        esm[src],
    )

    degree = torch.zeros(
        NUM_PROTEINS,
        dtype=torch.float32,
    )

    degree.index_add_(
        0,
        dst,
        torch.ones(
            dst.shape[0],
            dtype=torch.float32,
        ),
    )

    neighbor_mean = (
        neighbor_sum
        /
        degree.clamp_min(1.0).unsqueeze(1)
    )

    has_neighbor = (
        degree > 0
    ).unsqueeze(1)

    propagated = torch.where(
        has_neighbor,
        0.5 * esm + 0.5 * neighbor_mean,
        esm,
    )

    if not torch.isfinite(
        propagated
    ).all():
        raise RuntimeError(
            "Non-finite PPI-propagated features"
        )

    # --------------------------------------------------------
    # Restrict MASTER DTI to the frozen transductive
    # TRAINING drug universe.
    # --------------------------------------------------------

    master_drug_ids = dti[0]
    master_protein_ids = dti[1]

    if int(master_drug_ids.min()) < 0:
        raise RuntimeError(
            "Negative DTI drug index"
        )

    if int(master_drug_ids.max()) >= NUM_DRUGS:
        raise RuntimeError(
            "DTI drug index out of range"
        )

    if int(master_protein_ids.min()) < 0:
        raise RuntimeError(
            "Negative DTI protein index"
        )

    if (
        int(master_protein_ids.max())
        >= NUM_PROTEINS
    ):
        raise RuntimeError(
            "DTI protein index out of range"
        )

    allowed = torch.zeros(
        NUM_DRUGS,
        dtype=torch.bool,
    )

    allowed[train_drugs] = True

    keep = allowed[
        master_drug_ids
    ]

    drug_ids = master_drug_ids[
        keep
    ]

    protein_ids = master_protein_ids[
        keep
    ]

    if not bool(
        allowed[drug_ids].all()
    ):
        raise RuntimeError(
            "Filtered DTI contains drug outside "
            "training universe"
        )

    # --------------------------------------------------------
    # EXACT V9-D DRUG AGGREGATION:
    # mean PPI-aware target-protein representation.
    # --------------------------------------------------------

    drug_sum = torch.zeros(
        NUM_DRUGS,
        DIM,
        dtype=torch.float32,
    )

    drug_count = torch.zeros(
        NUM_DRUGS,
        dtype=torch.float32,
    )

    drug_sum.index_add_(
        0,
        drug_ids,
        propagated[protein_ids],
    )

    drug_count.index_add_(
        0,
        drug_ids,
        torch.ones(
            drug_ids.shape[0],
            dtype=torch.float32,
        ),
    )

    drug_context = (
        drug_sum
        /
        drug_count.clamp_min(1.0).unsqueeze(1)
    )

    zero_mask = (
        drug_count == 0
    )

    drug_context[
        zero_mask
    ] = 0.0

    if not torch.isfinite(
        drug_context
    ).all():
        raise RuntimeError(
            "Non-finite drug PPI context"
        )

    row_nonzero = (
        drug_context.abs().sum(dim=1) > 0
    )

    n_total = int(
        row_nonzero.sum()
    )

    n_train = int(
        row_nonzero[
            train_drugs
        ].sum()
    )

    outside = (
        row_nonzero
        & ~allowed
    )

    absent_drugs = torch.where(
        ~allowed
    )[0]

    print(
        "ESM2 shape            :",
        tuple(esm.shape),
    )

    print(
        "PPI edges             :",
        int(ppi.shape[1]),
    )

    print(
        "Master DTI edges      :",
        int(dti.shape[1]),
    )

    print(
        "Filtered DTI edges    :",
        int(drug_ids.numel()),
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

    if int(drug_ids.numel()) != 7330:
        raise RuntimeError(
            f"Expected 7330 filtered DTI edges, "
            f"got {drug_ids.numel()}"
        )

    if n_total != 1071:
        raise RuntimeError(
            f"Expected 1071 nonzero drugs, "
            f"got {n_total}"
        )

    if n_train != 1071:
        raise RuntimeError(
            "Not all training drugs have "
            "PPI-aware context"
        )

    if bool(outside.any()):
        raise RuntimeError(
            "PPI context exists outside "
            "training universe"
        )

    if not torch.all(
        drug_context[absent_drugs] == 0
    ):
        raise RuntimeError(
            "Absent drug has nonzero PPI context"
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        drug_context.contiguous(),
        OUT_PATH,
    )

    metadata = {
        "version":
            "V9-D_transductive_PPI_train1071_v1",

        "setting":
            "transductive",

        "num_drugs":
            NUM_DRUGS,

        "num_proteins":
            NUM_PROTEINS,

        "dimension":
            DIM,

        "ppi_edges":
            int(ppi.shape[1]),

        "master_dti_edges":
            int(dti.shape[1]),

        "filtered_dti_edges":
            int(drug_ids.numel()),

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

        "propagation":
            "0.5*self_esm2 + "
            "0.5*mean_incoming_ppi_neighbor_esm2",

        "aggregation":
            "mean_PPI_ESM2_over_training_universe_DTI_targets",

        "ddi_labels_used_for_feature":
            False,

        "validation_ddi_edges_used":
            False,

        "validation_fitting_used":
            False,

        "outer_test_loaded":
            False,

        "outer_test_status":
            "SEALED",

        "master_dti_sha256":
            sha256(dti_path),

        "esm2_sha256":
            sha256(esm_path),

        "ppi_sha256":
            sha256(ppi_path),

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
        "PASS: TRANSDUCTIVE V9-D "
        "PPI CONTEXT TRAIN-1071"
    )

    print(
        "VALIDATION DDI EDGES USED: NO"
    )

    print(
        "OUTER TEST LOADED: NO"
    )


if __name__ == "__main__":
    main()
