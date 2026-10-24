from pathlib import Path
import json
import os
import hashlib

import torch


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_coldstart_dev_splits_v2"
)

MASTER_DTI_PATH = (
    PROJECT_ROOT
    / "graphs"
    / "amddi_common"
    / "dti_edge_index.pt"
)

MASTER_PPI_PATH = (
    PROJECT_ROOT
    / "graphs"
    / "amddi_common"
    / "ppi_edge_index.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_biological_dev_v2"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "amddi_common"
    / "drug_mapping.csv"
)


# =============================================================================
# AUTHORITATIVE CONSTANTS
# =============================================================================

NUM_DRUGS = 1087
NUM_ACTIVE_DDI_DRUGS = 1071

SETTINGS = [
    "cold_start_S1",
    "cold_start_S2",
]

# Previously certified outer biological graph counts.
#
# These are used ONLY as integrity checks to determine which endpoint of the
# canonical DTI edge_index corresponds to the 1087-drug index space.
#
# If the current master DTI graph does not reproduce these known counts,
# Script 112 stops rather than guessing.
EXPECTED_MASTER_DTI_EDGES = 7408

EXPECTED_OUTER_DTI_EDGES = {
    "cold_start_S1": 5803,
    "cold_start_S2": 6063,
}

EXPECTED_PARTITION_SIZES = {
    "outer_train": 856,
    "outer_heldout": 215,
    "inner_train": 685,
    "inner_val": 171,
}


# =============================================================================
# IO HELPERS
# =============================================================================

def load_torch(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def atomic_torch_save(obj, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        obj,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json_save(obj, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(tmp, "w") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )

    os.replace(
        tmp,
        path,
    )


# =============================================================================
# HASHING
# =============================================================================

def tensor_sha256(x):
    x = (
        x.detach()
        .cpu()
        .contiguous()
    )

    h = hashlib.sha256()

    h.update(
        x.numpy().tobytes()
    )

    return h.hexdigest()


# =============================================================================
# TENSOR VALIDATION
# =============================================================================

def normalize_id_tensor(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{name}: expected torch.Tensor, "
            f"got {type(x)}"
        )

    x = (
        x.detach()
        .cpu()
        .long()
        .reshape(-1)
        .contiguous()
    )

    if len(x) == 0:
        raise RuntimeError(
            f"{name}: empty tensor"
        )

    if x.min().item() < 0:
        raise RuntimeError(
            f"{name}: negative drug ID found"
        )

    if x.max().item() >= NUM_DRUGS:
        raise RuntimeError(
            f"{name}: drug ID outside "
            f"[0,{NUM_DRUGS - 1}]"
        )

    if len(torch.unique(x)) != len(x):
        raise RuntimeError(
            f"{name}: duplicate drug IDs found"
        )

    return torch.sort(x).values


def normalize_edge_index(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{name}: expected torch.Tensor, "
            f"got {type(x)}"
        )

    x = (
        x.detach()
        .cpu()
        .long()
        .contiguous()
    )

    if x.ndim != 2:
        raise RuntimeError(
            f"{name}: expected rank-2 edge_index, "
            f"got {tuple(x.shape)}"
        )

    if x.shape[0] == 2:
        pass

    elif x.shape[1] == 2:
        x = x.t().contiguous()

    else:
        raise RuntimeError(
            f"{name}: expected shape [2,E] "
            f"or [E,2], got {tuple(x.shape)}"
        )

    if x.shape[1] == 0:
        raise RuntimeError(
            f"{name}: zero edges"
        )

    if x.min().item() < 0:
        raise RuntimeError(
            f"{name}: negative node index"
        )

    return x


def normalize_pair_tensor(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{name}: expected tensor"
        )

    x = (
        x.detach()
        .cpu()
        .long()
        .contiguous()
    )

    if x.ndim != 2:
        raise RuntimeError(
            f"{name}: expected rank-2 tensor"
        )

    if x.shape[1] == 2:
        pass

    elif x.shape[0] == 2:
        x = x.t().contiguous()

    else:
        raise RuntimeError(
            f"{name}: expected [N,2] "
            f"or [2,N], got {tuple(x.shape)}"
        )

    if x.min().item() < 0:
        raise RuntimeError(
            f"{name}: negative drug ID"
        )

    if x.max().item() >= NUM_DRUGS:
        raise RuntimeError(
            f"{name}: drug ID outside common-1087 space"
        )

    return x


# =============================================================================
# SET HELPERS
# =============================================================================

def as_set(x):
    return set(
        int(v)
        for v in x.tolist()
    )


def assert_disjoint(a, b, name_a, name_b):
    overlap = as_set(a) & as_set(b)

    if overlap:
        raise RuntimeError(
            f"{name_a} and {name_b} overlap: "
            f"{len(overlap)} drugs"
        )


def membership_mask(ids):
    mask = torch.zeros(
        NUM_DRUGS,
        dtype=torch.bool,
    )

    mask[ids] = True

    return mask


# =============================================================================
# LOAD DEVELOPMENT PARTITIONS
# =============================================================================

def load_partitions(setting):
    root = (
        DEV_SPLIT_ROOT
        / setting
    )

    required = {
        "inner_train":
            root / "inner_train_drugs.pt",

        "inner_val":
            root / "inner_val_drugs.pt",

        "outer_train":
            root / "outer_train_drugs.pt",

        "outer_heldout":
            root / "outer_heldout_drugs.pt",
    }

    out = {}

    for key, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required v2 split artifact:\n"
                f"{path}"
            )

        out[key] = normalize_id_tensor(
            load_torch(path),
            f"{setting}/{key}",
        )

    return out


# =============================================================================
# PARTITION CERTIFICATION
# =============================================================================

def certify_partitions(setting, p):
    for key, expected_size in (
        EXPECTED_PARTITION_SIZES.items()
    ):
        actual = len(p[key])

        if actual != expected_size:
            raise RuntimeError(
                f"{setting}/{key}: expected "
                f"{expected_size}, got {actual}"
            )

    assert_disjoint(
        p["outer_train"],
        p["outer_heldout"],
        f"{setting}/outer_train",
        f"{setting}/outer_heldout",
    )

    assert_disjoint(
        p["inner_train"],
        p["inner_val"],
        f"{setting}/inner_train",
        f"{setting}/inner_val",
    )

    outer_train_set = as_set(
        p["outer_train"]
    )

    outer_heldout_set = as_set(
        p["outer_heldout"]
    )

    inner_train_set = as_set(
        p["inner_train"]
    )

    inner_val_set = as_set(
        p["inner_val"]
    )

    if (
        inner_train_set
        | inner_val_set
    ) != outer_train_set:

        raise RuntimeError(
            f"{setting}: inner_train ∪ inner_val "
            f"does not exactly reconstruct outer_train"
        )

    if (
        inner_train_set
        & outer_heldout_set
    ):
        raise RuntimeError(
            f"{setting}: inner_train contains "
            f"outer-heldout drug"
        )

    if (
        inner_val_set
        & outer_heldout_set
    ):
        raise RuntimeError(
            f"{setting}: inner_val contains "
            f"outer-heldout drug"
        )

    reconstructed_active = (
        outer_train_set
        | outer_heldout_set
    )

    if len(
        reconstructed_active
    ) != NUM_ACTIVE_DDI_DRUGS:

        raise RuntimeError(
            f"{setting}: outer partition reconstructs "
            f"{len(reconstructed_active)} drugs; "
            f"expected {NUM_ACTIVE_DDI_DRUGS}"
        )


# =============================================================================
# DTI DRUG-ENDPOINT DETECTION
# =============================================================================

def count_edges_for_drugs(
    edge_index,
    drug_row,
    drug_ids,
):
    drug_ids = drug_ids.long()

    max_index = max(
        int(edge_index[drug_row].max().item()),
        NUM_DRUGS - 1,
    )

    lookup = torch.zeros(
        max_index + 1,
        dtype=torch.bool,
    )

    lookup[drug_ids] = True

    endpoint = edge_index[
        drug_row
    ]

    valid = (
        endpoint >= 0
    ) & (
        endpoint < len(lookup)
    )

    mask = torch.zeros_like(
        endpoint,
        dtype=torch.bool,
    )

    mask[valid] = lookup[
        endpoint[valid]
    ]

    return int(mask.sum().item())


def detect_dti_drug_row(
    master_dti,
    partitions,
):
    """
    Determine which DTI edge_index row contains common-1087 drug IDs.

    We do NOT infer this from index ranges alone because protein IDs can also
    numerically overlap [0,1086].

    Instead, each candidate row must reproduce BOTH previously certified
    setting-specific outer DTI counts:

        S1 = 5803
        S2 = 6063

    Exactly one row must satisfy both conditions.
    """

    candidates = []

    details = {}

    for row in [0, 1]:
        counts = {}

        for setting in SETTINGS:
            counts[setting] = (
                count_edges_for_drugs(
                    master_dti,
                    drug_row=row,
                    drug_ids=(
                        partitions[
                            setting
                        ][
                            "outer_train"
                        ]
                    ),
                )
            )

        details[row] = counts

        if all(
            counts[setting]
            ==
            EXPECTED_OUTER_DTI_EDGES[
                setting
            ]
            for setting in SETTINGS
        ):
            candidates.append(
                row
            )

    print()
    print(
        "DTI endpoint detection:"
    )

    for row in [0, 1]:
        print(
            f"  row {row}: "
            f"S1={details[row]['cold_start_S1']} "
            f"S2={details[row]['cold_start_S2']}"
        )

    if len(candidates) != 1:
        raise RuntimeError(
            "\nUnable to uniquely determine DTI drug endpoint.\n"
            f"Candidate rows matching certified counts: "
            f"{candidates}\n"
            f"Observed counts: {details}\n\n"
            "STOPPING rather than guessing."
        )

    return candidates[0], details


# =============================================================================
# FILTER DTI TO INNER-TRAIN DRUGS
# =============================================================================

def filter_dti_by_drugs(
    master_dti,
    drug_row,
    allowed_drugs,
):
    endpoint = master_dti[
        drug_row
    ]

    valid_range = (
        (endpoint >= 0)
        &
        (endpoint < NUM_DRUGS)
    )

    lookup = torch.zeros(
        NUM_DRUGS,
        dtype=torch.bool,
    )

    lookup[
        allowed_drugs
    ] = True

    keep = torch.zeros_like(
        endpoint,
        dtype=torch.bool,
    )

    keep[
        valid_range
    ] = lookup[
        endpoint[
            valid_range
        ]
    ]

    filtered = (
        master_dti[
            :,
            keep,
        ]
        .contiguous()
    )

    return filtered


# =============================================================================
# DDI TRAIN GRAPH
# =============================================================================

def build_inner_ddi_graph(
    train_pos,
    inner_train,
    name,
):
    pairs = normalize_pair_tensor(
        train_pos,
        name,
    )

    inner_train_set = as_set(
        inner_train
    )

    for a, b in pairs.tolist():
        if (
            int(a) not in inner_train_set
            or
            int(b) not in inner_train_set
        ):
            raise RuntimeError(
                f"{name}: training positive pair "
                f"({a},{b}) contains a non-inner-train drug"
            )

    forward = (
        pairs.t()
        .contiguous()
    )

    reverse = torch.stack(
        [
            forward[1],
            forward[0],
        ],
        dim=0,
    )

    edge_index = torch.cat(
        [
            forward,
            reverse,
        ],
        dim=1,
    )

    return edge_index


# =============================================================================
# DTI SET EXPOSURE
# =============================================================================

def dti_exposure_count(
    dti,
    drug_row,
    ids,
):
    endpoint = dti[
        drug_row
    ]

    lookup = torch.zeros(
        NUM_DRUGS,
        dtype=torch.bool,
    )

    lookup[
        ids
    ] = True

    valid = (
        (endpoint >= 0)
        &
        (endpoint < NUM_DRUGS)
    )

    exposure = torch.zeros_like(
        endpoint,
        dtype=torch.bool,
    )

    exposure[
        valid
    ] = lookup[
        endpoint[
            valid
        ]
    ]

    return int(
        exposure.sum().item()
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 110)
    print("AM-DDI COMMON-1087")
    print(
        "CREATE BIOLOGICAL DEVELOPMENT ARTIFACTS "
        "— STRICT COLD-START V2"
    )
    print("=" * 110)

    # -------------------------------------------------------------------------
    # BASIC PATH CHECKS
    # -------------------------------------------------------------------------

    if not DEV_SPLIT_ROOT.exists():
        raise FileNotFoundError(
            f"Missing certified v2 development splits:\n"
            f"{DEV_SPLIT_ROOT}"
        )

    if not MASTER_DTI_PATH.exists():
        raise FileNotFoundError(
            f"Missing canonical master DTI graph:\n"
            f"{MASTER_DTI_PATH}"
        )

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Missing common-1087 drug mapping:\n"
            f"{MAPPING_PATH}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # LOAD / CERTIFY S1 + S2 PARTITIONS
    # -------------------------------------------------------------------------

    partitions = {}

    for setting in SETTINGS:
        partitions[setting] = (
            load_partitions(
                setting
            )
        )

        certify_partitions(
            setting,
            partitions[setting],
        )

    print()
    print(
        "PASS: S1/S2 development partitions "
        "have expected sizes and semantics."
    )

    # -------------------------------------------------------------------------
    # LOAD MASTER DTI
    # -------------------------------------------------------------------------

    master_dti = normalize_edge_index(
        load_torch(
            MASTER_DTI_PATH
        ),
        "master DTI",
    )

    master_edge_count = int(
        master_dti.shape[1]
    )

    print()
    print(
        "Master DTI shape:",
        tuple(
            master_dti.shape
        ),
    )

    if (
        master_edge_count
        != EXPECTED_MASTER_DTI_EDGES
    ):
        raise RuntimeError(
            f"Master DTI has {master_edge_count} edges; "
            f"expected certified count "
            f"{EXPECTED_MASTER_DTI_EDGES}.\n"
            "STOPPING rather than using a different DTI graph."
        )

    # -------------------------------------------------------------------------
    # DETECT DRUG ENDPOINT WITHOUT GUESSING
    # -------------------------------------------------------------------------

    (
        drug_row,
        detection_details,
    ) = detect_dti_drug_row(
        master_dti,
        partitions,
    )

    protein_row = (
        1
        if drug_row == 0
        else 0
    )

    print()
    print(
        f"PASS: DTI drug endpoint uniquely detected "
        f"as row {drug_row}."
    )

    print(
        f"Protein endpoint preserved as row "
        f"{protein_row}."
    )

    # -------------------------------------------------------------------------
    # OPTIONAL PPI LOAD
    # -------------------------------------------------------------------------

    master_ppi = None
    ppi_status = "not_found"

    if MASTER_PPI_PATH.exists():
        master_ppi = normalize_edge_index(
            load_torch(
                MASTER_PPI_PATH
            ),
            "master PPI",
        )

        ppi_status = "loaded"

        print()
        print(
            "Canonical PPI found:",
            MASTER_PPI_PATH,
        )

        print(
            "Master PPI shape:",
            tuple(
                master_ppi.shape
            ),
        )

    else:
        print()
        print(
            "NOTE: canonical PPI file was not found at:"
        )

        print(
            MASTER_PPI_PATH
        )

        print(
            "Script 112 will NOT guess another PPI path."
        )

        print(
            "DTI/DDI biological development artifacts "
            "will still be created safely."
        )

    # -------------------------------------------------------------------------
    # SETTING-SPECIFIC BUILD
    # -------------------------------------------------------------------------

    manifest = {
        "version":
            "biological_dev_v2",

        "num_drugs":
            NUM_DRUGS,

        "num_active_ddi_drugs":
            NUM_ACTIVE_DDI_DRUGS,

        "master_dti_path":
            str(
                MASTER_DTI_PATH
            ),

        "master_dti_edges":
            master_edge_count,

        "master_dti_sha256":
            tensor_sha256(
                master_dti
            ),

        "dti_drug_row":
            drug_row,

        "dti_protein_row":
            protein_row,

        "dti_endpoint_detection":
            detection_details,

        "ppi_status":
            ppi_status,

        "ppi_path":
            (
                str(MASTER_PPI_PATH)
                if master_ppi is not None
                else None
            ),

        "important_policy": {
            "old_dti_bio_128_reused":
                False,

            "inner_val_dti_allowed":
                False,

            "outer_heldout_dti_allowed":
                False,

            "ddi_validation_edges_in_training_graph":
                False,

            "ddi_test_edges_in_training_graph":
                False,

            "ppi_is_setting_specific":
                False,

            "outer_test_used":
                False,
        },

        "settings": {},
    }

    # Save unchanged PPI once at root if canonical file exists.
    if master_ppi is not None:
        atomic_torch_save(
            master_ppi,
            OUTPUT_ROOT
            / "ppi_edge_index.pt",
        )

    for setting in SETTINGS:
        print()
        print("=" * 110)
        print(setting)
        print("=" * 110)

        p = partitions[
            setting
        ]

        setting_root = (
            OUTPUT_ROOT
            / setting
        )

        setting_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------------------
        # SAVE PARTITION MASKS
        # ---------------------------------------------------------------------

        masks = {
            "inner_train":
                membership_mask(
                    p["inner_train"]
                ),

            "inner_val":
                membership_mask(
                    p["inner_val"]
                ),

            "outer_train":
                membership_mask(
                    p["outer_train"]
                ),

            "outer_heldout":
                membership_mask(
                    p["outer_heldout"]
                ),
        }

        for key, mask in (
            masks.items()
        ):
            atomic_torch_save(
                mask,
                setting_root
                / f"{key}_drug_mask.pt",
            )

        # ---------------------------------------------------------------------
        # FILTER MASTER DTI TO INNER-TRAIN DRUGS ONLY
        # ---------------------------------------------------------------------

        inner_dti = filter_dti_by_drugs(
            master_dti,
            drug_row=drug_row,
            allowed_drugs=(
                p["inner_train"]
            ),
        )

        # ---------------------------------------------------------------------
        # STRICT EXPOSURE CHECKS
        # ---------------------------------------------------------------------

        inner_train_exposure = (
            dti_exposure_count(
                inner_dti,
                drug_row,
                p["inner_train"],
            )
        )

        inner_val_exposure = (
            dti_exposure_count(
                inner_dti,
                drug_row,
                p["inner_val"],
            )
        )

        outer_heldout_exposure = (
            dti_exposure_count(
                inner_dti,
                drug_row,
                p["outer_heldout"],
            )
        )

        if (
            inner_train_exposure
            != inner_dti.shape[1]
        ):
            raise RuntimeError(
                f"{setting}: filtered DTI contains "
                f"edge whose drug endpoint is not "
                f"inner-train."
            )

        if inner_val_exposure != 0:
            raise RuntimeError(
                f"{setting}: inner-validation drug "
                f"still exposed through DTI."
            )

        if outer_heldout_exposure != 0:
            raise RuntimeError(
                f"{setting}: outer-heldout drug "
                f"still exposed through DTI."
            )

        atomic_torch_save(
            inner_dti,
            setting_root
            / "dti_edge_index.pt",
        )

        # ---------------------------------------------------------------------
        # SAFE DRUG BIOLOGICAL INITIALIZATION MASK
        # ---------------------------------------------------------------------
        #
        # This is intentionally NOT the old DTI-derived 128-D vector.
        #
        # Zero feature matrix is saved as a safe placeholder so future models
        # cannot accidentally load the old leakage-prone biological vectors.
        #
        # The biological encoder may later learn representations from the
        # filtered DTI/PPI topology.
        # ---------------------------------------------------------------------

        safe_zero_bio = torch.zeros(
            (
                NUM_DRUGS,
                128,
            ),
            dtype=torch.float32,
        )

        atomic_torch_save(
            safe_zero_bio,
            setting_root
            / "drug_bio_init_zero_128.pt",
        )

        # ---------------------------------------------------------------------
        # BUILD DDI TRAIN GRAPH FROM INNER TRAIN POSITIVES ONLY
        # ---------------------------------------------------------------------

        train_pos_path = (
            DEV_SPLIT_ROOT
            / setting
            / "train_pos.pt"
        )

        if not train_pos_path.exists():
            raise FileNotFoundError(
                train_pos_path
            )

        train_pos = load_torch(
            train_pos_path
        )

        ddi_train_edge_index = (
            build_inner_ddi_graph(
                train_pos,
                p["inner_train"],
                (
                    f"{setting}/"
                    f"train_pos"
                ),
            )
        )

        atomic_torch_save(
            ddi_train_edge_index,
            setting_root
            / "ddi_train_edge_index.pt",
        )

        # ---------------------------------------------------------------------
        # COUNTS RELATIVE TO MASTER DTI
        # ---------------------------------------------------------------------

        master_inner_train_edges = (
            dti_exposure_count(
                master_dti,
                drug_row,
                p["inner_train"],
            )
        )

        master_inner_val_edges = (
            dti_exposure_count(
                master_dti,
                drug_row,
                p["inner_val"],
            )
        )

        master_outer_heldout_edges = (
            dti_exposure_count(
                master_dti,
                drug_row,
                p["outer_heldout"],
            )
        )

        master_outer_train_edges = (
            dti_exposure_count(
                master_dti,
                drug_row,
                p["outer_train"],
            )
        )

        expected_outer = (
            EXPECTED_OUTER_DTI_EDGES[
                setting
            ]
        )

        if (
            master_outer_train_edges
            != expected_outer
        ):
            raise RuntimeError(
                f"{setting}: master DTI outer-train "
                f"count changed unexpectedly. "
                f"Observed={master_outer_train_edges}, "
                f"expected={expected_outer}"
            )

        # Inner train and inner val exactly partition outer train.
        if (
            master_inner_train_edges
            +
            master_inner_val_edges
            !=
            master_outer_train_edges
        ):
            raise RuntimeError(
                f"{setting}: DTI edge partition inconsistency.\n"
                f"inner_train_edges="
                f"{master_inner_train_edges}\n"
                f"inner_val_edges="
                f"{master_inner_val_edges}\n"
                f"outer_train_edges="
                f"{master_outer_train_edges}"
            )

        # ---------------------------------------------------------------------
        # STORE IDS THEMSELVES FOR EXPLICIT DOWNSTREAM USE
        # ---------------------------------------------------------------------

        for key in [
            "inner_train",
            "inner_val",
            "outer_train",
            "outer_heldout",
        ]:
            atomic_torch_save(
                p[key],
                setting_root
                / f"{key}_drugs.pt",
            )

        # ---------------------------------------------------------------------
        # SETTING MANIFEST
        # ---------------------------------------------------------------------

        setting_manifest = {
            "setting":
                setting,

            "partition_sizes": {
                "inner_train":
                    len(
                        p["inner_train"]
                    ),

                "inner_val":
                    len(
                        p["inner_val"]
                    ),

                "outer_train":
                    len(
                        p["outer_train"]
                    ),

                "outer_heldout":
                    len(
                        p["outer_heldout"]
                    ),
            },

            "master_dti_exposure": {
                "inner_train_edges":
                    master_inner_train_edges,

                "inner_val_edges":
                    master_inner_val_edges,

                "outer_train_edges":
                    master_outer_train_edges,

                "outer_heldout_edges":
                    master_outer_heldout_edges,
            },

            "filtered_dti": {
                "num_edges":
                    int(
                        inner_dti.shape[1]
                    ),

                "inner_train_exposure":
                    inner_train_exposure,

                "inner_val_exposure":
                    inner_val_exposure,

                "outer_heldout_exposure":
                    outer_heldout_exposure,

                "sha256":
                    tensor_sha256(
                        inner_dti
                    ),
            },

            "ddi_train_graph": {
                "directed_edges":
                    int(
                        ddi_train_edge_index.shape[
                            1
                        ]
                    ),

                "positive_pairs":
                    int(
                        ddi_train_edge_index.shape[
                            1
                        ]
                        // 2
                    ),

                "sha256":
                    tensor_sha256(
                        ddi_train_edge_index
                    ),
            },

            "drug_bio_initialization": {
                "shape":
                    [
                        NUM_DRUGS,
                        128,
                    ],

                "all_zero":
                    True,

                "old_dti_derived_features_used":
                    False,
            },

            "leakage_policy": {
                "inner_val_drug_dti_edges":
                    0,

                "outer_heldout_drug_dti_edges":
                    0,

                "old_dti_bio_features_reused":
                    False,

                "ddi_graph_source":
                    "inner-development train_pos only",
            },
        }

        atomic_json_save(
            setting_manifest,
            setting_root
            / "manifest.json",
        )

        manifest[
            "settings"
        ][
            setting
        ] = setting_manifest

        # ---------------------------------------------------------------------
        # PRINT SETTING SUMMARY
        # ---------------------------------------------------------------------

        print(
            "Partition:"
        )

        print(
            f"  inner train    : "
            f"{len(p['inner_train'])}"
        )

        print(
            f"  inner val      : "
            f"{len(p['inner_val'])}"
        )

        print(
            f"  outer train    : "
            f"{len(p['outer_train'])}"
        )

        print(
            f"  outer heldout  : "
            f"{len(p['outer_heldout'])}"
        )

        print()

        print(
            "Master DTI exposure:"
        )

        print(
            f"  inner-train edges   : "
            f"{master_inner_train_edges}"
        )

        print(
            f"  inner-val edges     : "
            f"{master_inner_val_edges}"
        )

        print(
            f"  outer-train edges   : "
            f"{master_outer_train_edges}"
        )

        print(
            f"  outer-heldout edges : "
            f"{master_outer_heldout_edges}"
        )

        print()

        print(
            "Filtered development DTI:"
        )

        print(
            f"  edges                : "
            f"{inner_dti.shape[1]}"
        )

        print(
            f"  inner-val exposure   : "
            f"{inner_val_exposure}"
        )

        print(
            f"  outer-heldout exposure: "
            f"{outer_heldout_exposure}"
        )

        print()

        print(
            "DDI training graph:"
        )

        print(
            f"  positive pairs : "
            f"{ddi_train_edge_index.shape[1] // 2}"
        )

        print(
            f"  directed edges : "
            f"{ddi_train_edge_index.shape[1]}"
        )

        print()

        print(
            "PASS:",
            setting,
            "biological development artifacts created."
        )

    # -------------------------------------------------------------------------
    # CROSS-SETTING PARTITION CHECK
    # -------------------------------------------------------------------------

    s1 = partitions[
        "cold_start_S1"
    ]

    s2 = partitions[
        "cold_start_S2"
    ]

    cross = {
        "inner_train_intersection":
            len(
                as_set(
                    s1["inner_train"]
                )
                &
                as_set(
                    s2["inner_train"]
                )
            ),

        "inner_val_intersection":
            len(
                as_set(
                    s1["inner_val"]
                )
                &
                as_set(
                    s2["inner_val"]
                )
            ),

        "outer_heldout_intersection":
            len(
                as_set(
                    s1["outer_heldout"]
                )
                &
                as_set(
                    s2["outer_heldout"]
                )
            ),
    }

    manifest[
        "cross_setting"
    ] = cross

    # -------------------------------------------------------------------------
    # FINAL MANIFEST
    # -------------------------------------------------------------------------

    atomic_json_save(
        manifest,
        OUTPUT_ROOT
        / "manifest.json",
    )

    print()
    print("=" * 110)
    print("SCRIPT 112 BUILD SUMMARY")
    print("=" * 110)

    print(
        "PASS: common-1087 index preserved."
    )

    print(
        "PASS: S1/S2 setting-specific outer "
        "partitions preserved."
    )

    print(
        "PASS: inner-train/inner-val partitions preserved."
    )

    print(
        "PASS: DTI drug endpoint determined from "
        "previously certified edge counts."
    )

    print(
        "PASS: development DTI retains ONLY "
        "inner-training-drug DTI edges."
    )

    print(
        "PASS: inner-validation DTI exposure = 0."
    )

    print(
        "PASS: outer-heldout DTI exposure = 0."
    )

    print(
        "PASS: DDI training graph built only from "
        "inner-development train positives."
    )

    print(
        "PASS: old DTI-derived drug bio128 "
        "features were NOT reused."
    )

    print(
        "PASS: safe zero biological drug "
        "initialization created."
    )

    if master_ppi is not None:
        print(
            "PASS: canonical protein-only PPI graph "
            "preserved unchanged."
        )
    else:
        print(
            "NOTE: PPI was not copied because the "
            "canonical expected path was absent."
        )

    print()
    print(
        "Output root:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "IMPORTANT: these artifacts are NOT YET "
        "approved for biological training."
    )

    print(
        "Run independent Script 113 audit first."
    )


if __name__ == "__main__":
    main()
