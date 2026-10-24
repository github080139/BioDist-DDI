#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib

import torch


ROOT = Path(__file__).resolve().parents[1]

BIO_ROOT = (
    ROOT
    / "data/amddi_common_1087_biological_dev_v4"
)

OUT_ROOT = (
    ROOT
    / "data/amddi_common_1087_leakage_free_features"
)

NUM_DRUGS = 1087
NUM_PROTEINS = 2321
DIM = 128

SPLITS = (
    "cold_start_S1",
    "cold_start_S2",
)


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def main():

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    esm = torch.load(
        BIO_ROOT / "protein_features_esm2_128.pt",
        map_location="cpu",
        weights_only=True,
    ).float()


    ppi = torch.load(
        BIO_ROOT / "ppi_edge_index.pt",
        map_location="cpu",
        weights_only=True,
    ).long()


    if tuple(esm.shape) != (
        NUM_PROTEINS,
        DIM,
    ):
        raise RuntimeError(
            f"Unexpected ESM2 shape: {tuple(esm.shape)}"
        )


    if ppi.ndim != 2 or ppi.shape[0] != 2:
        raise RuntimeError(
            f"Unexpected PPI shape: {tuple(ppi.shape)}"
        )


    if int(ppi.min()) < 0:
        raise RuntimeError(
            "Negative PPI protein index"
        )


    if int(ppi.max()) >= NUM_PROTEINS:
        raise RuntimeError(
            "PPI protein index out of range"
        )


    if torch.isnan(esm).any():
        raise RuntimeError(
            "ESM2 contains NaN"
        )


    # --------------------------------------------------------
    # Deterministic one-hop mean PPI propagation.
    #
    # For each destination protein:
    #     neighbor_mean = mean(source protein ESM2)
    #
    # Then:
    #     H_ppi = 0.5 * (H_esm + neighbor_mean)
    #
    # No DDI labels.
    # No drug IDs learned.
    # No validation/test fitting.
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


    if torch.isnan(propagated).any():
        raise RuntimeError(
            "PPI propagated protein features contain NaN"
        )


    print("=" * 90)
    print("V9-D PPI CONTEXT PREPARATION")
    print("=" * 90)

    print(
        "ESM2:",
        tuple(esm.shape),
    )

    print(
        "PPI:",
        tuple(ppi.shape),
    )

    print(
        "PPI-aware protein:",
        tuple(propagated.shape),
    )


    for split in SPLITS:

        dti_path = (
            BIO_ROOT
            / split
            / "dti_edge_index.pt"
        )


        dti = torch.load(
            dti_path,
            map_location="cpu",
            weights_only=True,
        ).long()


        if dti.ndim != 2 or dti.shape[0] != 2:
            raise RuntimeError(
                f"{split}: bad DTI shape {tuple(dti.shape)}"
            )


        drug_ids = dti[0]
        protein_ids = dti[1]


        if int(drug_ids.min()) < 0:
            raise RuntimeError(
                f"{split}: negative drug index"
            )

        if int(drug_ids.max()) >= NUM_DRUGS:
            raise RuntimeError(
                f"{split}: drug index out of range"
            )

        if int(protein_ids.min()) < 0:
            raise RuntimeError(
                f"{split}: negative protein index"
            )

        if int(protein_ids.max()) >= NUM_PROTEINS:
            raise RuntimeError(
                f"{split}: protein index out of range"
            )


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


        if torch.isnan(drug_context).any():
            raise RuntimeError(
                f"{split}: NaN in drug PPI context"
            )


        out_path = (
            OUT_ROOT
            / f"{split}_ppi_esm2_context_128.pt"
        )


        torch.save(
            drug_context.contiguous(),
            out_path,
        )


        metadata = {
            "split": split,
            "num_drugs": NUM_DRUGS,
            "num_proteins": NUM_PROTEINS,
            "dimension": DIM,
            "ppi_edges": int(ppi.shape[1]),
            "dti_edges": int(dti.shape[1]),
            "nonzero_drugs": int(
                (~zero_mask).sum().item()
            ),
            "propagation": (
                "0.5*self_esm2 + "
                "0.5*mean_incoming_ppi_neighbor_esm2"
            ),
            "ddi_labels_used": False,
            "validation_fitting_used": False,
            "outer_test_loaded": False,
            "outer_test_status": "SEALED",
        }


        meta_path = (
            OUT_ROOT
            / f"{split}_ppi_esm2_context_128_metadata.json"
        )


        meta_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n"
        )


        print("-" * 90)
        print(split)
        print(
            "DTI edges:",
            dti.shape[1],
        )
        print(
            "Drug context:",
            tuple(drug_context.shape),
        )
        print(
            "Nonzero drugs:",
            metadata["nonzero_drugs"],
        )
        print(
            "Saved:",
            out_path,
        )
        print(
            "SHA256:",
            sha256(out_path),
        )


    print("=" * 90)
    print("OUTER TEST: SEALED / NOT LOADED")
    print("V9-D PPI CONTEXT PREPARATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
