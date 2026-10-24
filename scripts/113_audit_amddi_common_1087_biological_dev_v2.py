from pathlib import Path
import json
import hashlib
import math

import torch


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_coldstart_dev_splits_v2"
)

BIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_biological_dev_v2"
)

MASTER_DTI_PATH = (
    PROJECT_ROOT
    / "graphs"
    / "amddi_common"
    / "dti_edge_index.pt"
)

MASTER_MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "amddi_common"
    / "drug_mapping.csv"
)

ROOT_MANIFEST_PATH = (
    BIO_ROOT
    / "manifest.json"
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

EXPECTED = {
    "cold_start_S1": {
        "outer_train": 856,
        "outer_heldout": 215,
        "inner_train": 685,
        "inner_val": 171,
        "train_pos": 14494,
        "train_neg": 14494,
        "val_pos": 7379,
        "val_neg": 7379,
        "test_pos": 11713,
        "test_neg": 11713,
        "outer_train_dti_edges": 5803,
        "inner_train_dti_edges": 4817,
        "inner_val_master_dti_edges": 986,
        "outer_heldout_master_dti_edges": 1543,
        "ddi_directed_edges": 28988,
    },

    "cold_start_S2": {
        "outer_train": 856,
        "outer_heldout": 215,
        "inner_train": 685,
        "inner_val": 171,
        "train_pos": 16903,
        "train_neg": 16903,
        "val_pos": 686,
        "val_neg": 686,
        "test_pos": 1086,
        "test_neg": 1086,
        "outer_train_dti_edges": 6063,
        "inner_train_dti_edges": 4703,
        "inner_val_master_dti_edges": 1360,
        "outer_heldout_master_dti_edges": 1283,
        "ddi_directed_edges": 33806,
    },
}

EXPECTED_MASTER_DTI_EDGES = 7408
EXPECTED_DTI_DRUG_ROW = 0
EXPECTED_DTI_PROTEIN_ROW = 1


# =============================================================================
# AUDIT COUNTERS
# =============================================================================

PASS_COUNT = 0
WARNING_COUNT = 0
FAIL_COUNT = 0


def audit_pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"PASS: {message}")


def audit_warning(message):
    global WARNING_COUNT
    WARNING_COUNT += 1
    print(f"WARNING: {message}")


def audit_fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"FAIL: {message}")


# =============================================================================
# IO
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


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_ids(x, name):
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(
            f"{name}: expected torch.Tensor"
        )

    x = (
        x.detach()
        .cpu()
        .long()
        .reshape(-1)
        .contiguous()
    )

    if x.numel() == 0:
        raise RuntimeError(
            f"{name}: empty"
        )

    return torch.sort(x).values


def normalize_edge_index(x, name):
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(
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
            f"{name}: expected rank-2"
        )

    if x.shape[0] == 2:
        return x

    if x.shape[1] == 2:
        return x.t().contiguous()

    raise RuntimeError(
        f"{name}: invalid edge shape "
        f"{tuple(x.shape)}"
    )


def normalize_pairs(x, name):
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(
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
            f"{name}: expected rank-2"
        )

    if x.shape[1] == 2:
        return x

    if x.shape[0] == 2:
        return x.t().contiguous()

    raise RuntimeError(
        f"{name}: invalid pair shape "
        f"{tuple(x.shape)}"
    )


# =============================================================================
# SET / PAIR HELPERS
# =============================================================================

def tensor_set(x):
    return set(
        int(v)
        for v in x.tolist()
    )


def canonical_pair_set(x):
    pairs = normalize_pairs(
        x,
        "pair tensor",
    )

    return {
        (
            min(int(a), int(b)),
            max(int(a), int(b)),
        )
        for a, b in pairs.tolist()
    }


def edge_pair_set(edge_index):
    edge_index = normalize_edge_index(
        edge_index,
        "edge_index",
    )

    return {
        (
            int(a),
            int(b),
        )
        for a, b in edge_index.t().tolist()
    }


def sha256_tensor(x):
    x = (
        x.detach()
        .cpu()
        .contiguous()
    )

    return hashlib.sha256(
        x.numpy().tobytes()
    ).hexdigest()


# =============================================================================
# DTI HELPERS
# =============================================================================

def count_dti_exposure(
    edge_index,
    drug_row,
    drug_ids,
):
    edge_index = normalize_edge_index(
        edge_index,
        "DTI",
    )

    ids = tensor_set(
        drug_ids
    )

    endpoint = edge_index[
        drug_row
    ]

    return sum(
        1
        for x in endpoint.tolist()
        if int(x) in ids
    )


def dti_drug_set(
    edge_index,
    drug_row,
):
    edge_index = normalize_edge_index(
        edge_index,
        "DTI",
    )

    return set(
        int(x)
        for x in edge_index[
            drug_row
        ].tolist()
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    global FAIL_COUNT

    print("=" * 110)
    print("AM-DDI COMMON-1087")
    print(
        "INDEPENDENT BIOLOGICAL DEVELOPMENT AUDIT "
        "— STRICT COLD-START V2"
    )
    print("=" * 110)

    # =========================================================================
    # BASIC FILE EXISTENCE
    # =========================================================================

    required_global = [
        DEV_SPLIT_ROOT,
        BIO_ROOT,
        MASTER_DTI_PATH,
        MASTER_MAPPING_PATH,
        ROOT_MANIFEST_PATH,
    ]

    for path in required_global:
        if not path.exists():
            audit_fail(
                f"missing required path: {path}"
            )

    if FAIL_COUNT:
        raise RuntimeError(
            "Required global artifacts missing."
        )

    audit_pass(
        "all required global biological-development paths exist"
    )

    # =========================================================================
    # ROOT MANIFEST
    # =========================================================================

    root_manifest = load_json(
        ROOT_MANIFEST_PATH
    )

    if (
        root_manifest.get("version")
        == "biological_dev_v2"
    ):
        audit_pass(
            "root manifest version is biological_dev_v2"
        )
    else:
        audit_fail(
            "root manifest version mismatch"
        )

    if (
        int(root_manifest.get("num_drugs", -1))
        == NUM_DRUGS
    ):
        audit_pass(
            "root manifest preserves common-1087 universe"
        )
    else:
        audit_fail(
            "root manifest num_drugs mismatch"
        )

    if (
        int(
            root_manifest.get(
                "num_active_ddi_drugs",
                -1,
            )
        )
        == NUM_ACTIVE_DDI_DRUGS
    ):
        audit_pass(
            "root manifest preserves 1071 DDI-active drugs"
        )
    else:
        audit_fail(
            "root manifest active-drug count mismatch"
        )

    # =========================================================================
    # MASTER DTI
    # =========================================================================

    master_dti = normalize_edge_index(
        load_torch(
            MASTER_DTI_PATH
        ),
        "master DTI",
    )

    if master_dti.shape == (
        2,
        EXPECTED_MASTER_DTI_EDGES,
    ):
        audit_pass(
            "master DTI shape is exactly (2, 7408)"
        )
    else:
        audit_fail(
            f"master DTI shape mismatch: "
            f"{tuple(master_dti.shape)}"
        )

    manifest_master_hash = (
        root_manifest.get(
            "master_dti_sha256"
        )
    )

    actual_master_hash = sha256_tensor(
        master_dti
    )

    if (
        manifest_master_hash
        == actual_master_hash
    ):
        audit_pass(
            "master DTI hash matches Script 112 manifest"
        )
    else:
        audit_fail(
            "master DTI hash mismatch"
        )

    drug_row = int(
        root_manifest.get(
            "dti_drug_row",
            -1,
        )
    )

    protein_row = int(
        root_manifest.get(
            "dti_protein_row",
            -1,
        )
    )

    if drug_row == EXPECTED_DTI_DRUG_ROW:
        audit_pass(
            "DTI drug endpoint independently agrees with row 0"
        )
    else:
        audit_fail(
            f"unexpected DTI drug row: {drug_row}"
        )

    if protein_row == EXPECTED_DTI_PROTEIN_ROW:
        audit_pass(
            "DTI protein endpoint agrees with row 1"
        )
    else:
        audit_fail(
            f"unexpected DTI protein row: {protein_row}"
        )

    # =========================================================================
    # PPI STATUS
    # =========================================================================

    ppi_status = root_manifest.get(
        "ppi_status"
    )

    ppi_path = (
        BIO_ROOT
        / "ppi_edge_index.pt"
    )

    if ppi_status == "loaded":
        if ppi_path.exists():
            audit_pass(
                "PPI manifest says loaded and copied PPI exists"
            )
        else:
            audit_fail(
                "PPI marked loaded but copied file is absent"
            )

    elif ppi_status == "not_found":
        if not ppi_path.exists():
            audit_warning(
                "canonical PPI was not available to Script 112; "
                "DTI/DDI layer can be certified, but PPI-enabled "
                "training is NOT yet certified"
            )
        else:
            audit_fail(
                "PPI file exists although manifest says not_found"
            )

    else:
        audit_fail(
            f"unknown PPI status: {ppi_status}"
        )

    # =========================================================================
    # SETTING AUDITS
    # =========================================================================

    for setting in SETTINGS:
        print()
        print("=" * 110)
        print(setting)
        print("=" * 110)

        expected = EXPECTED[
            setting
        ]

        split_root = (
            DEV_SPLIT_ROOT
            / setting
        )

        bio_root = (
            BIO_ROOT
            / setting
        )

        # ---------------------------------------------------------------------
        # LOAD SOURCE PARTITIONS
        # ---------------------------------------------------------------------

        source_inner_train = normalize_ids(
            load_torch(
                split_root
                / "inner_train_drugs.pt"
            ),
            "source inner train",
        )

        source_inner_val = normalize_ids(
            load_torch(
                split_root
                / "inner_val_drugs.pt"
            ),
            "source inner val",
        )

        source_outer_train = normalize_ids(
            load_torch(
                split_root
                / "outer_train_drugs.pt"
            ),
            "source outer train",
        )

        source_outer_heldout = normalize_ids(
            load_torch(
                split_root
                / "outer_heldout_drugs.pt"
            ),
            "source outer heldout",
        )

        # ---------------------------------------------------------------------
        # LOAD COPIED PARTITIONS
        # ---------------------------------------------------------------------

        copied_inner_train = normalize_ids(
            load_torch(
                bio_root
                / "inner_train_drugs.pt"
            ),
            "copied inner train",
        )

        copied_inner_val = normalize_ids(
            load_torch(
                bio_root
                / "inner_val_drugs.pt"
            ),
            "copied inner val",
        )

        copied_outer_train = normalize_ids(
            load_torch(
                bio_root
                / "outer_train_drugs.pt"
            ),
            "copied outer train",
        )

        copied_outer_heldout = normalize_ids(
            load_torch(
                bio_root
                / "outer_heldout_drugs.pt"
            ),
            "copied outer heldout",
        )

        # ---------------------------------------------------------------------
        # SIZE CHECKS
        # ---------------------------------------------------------------------

        size_checks = [
            (
                "inner_train",
                source_inner_train,
            ),
            (
                "inner_val",
                source_inner_val,
            ),
            (
                "outer_train",
                source_outer_train,
            ),
            (
                "outer_heldout",
                source_outer_heldout,
            ),
        ]

        for name, tensor in size_checks:
            if (
                len(tensor)
                == expected[name]
            ):
                audit_pass(
                    f"{setting}: {name} size = "
                    f"{expected[name]}"
                )
            else:
                audit_fail(
                    f"{setting}: {name} size mismatch"
                )

        # ---------------------------------------------------------------------
        # EXACT COPY CHECKS
        # ---------------------------------------------------------------------

        copy_checks = [
            (
                "inner_train",
                source_inner_train,
                copied_inner_train,
            ),
            (
                "inner_val",
                source_inner_val,
                copied_inner_val,
            ),
            (
                "outer_train",
                source_outer_train,
                copied_outer_train,
            ),
            (
                "outer_heldout",
                source_outer_heldout,
                copied_outer_heldout,
            ),
        ]

        for name, source, copied in copy_checks:
            if torch.equal(
                source,
                copied,
            ):
                audit_pass(
                    f"{setting}: copied {name} IDs "
                    f"exactly match certified v2 split"
                )
            else:
                audit_fail(
                    f"{setting}: copied {name} IDs differ "
                    f"from certified v2 split"
                )

        # ---------------------------------------------------------------------
        # PARTITION SEMANTICS
        # ---------------------------------------------------------------------

        inner_train_set = tensor_set(
            source_inner_train
        )

        inner_val_set = tensor_set(
            source_inner_val
        )

        outer_train_set = tensor_set(
            source_outer_train
        )

        outer_heldout_set = tensor_set(
            source_outer_heldout
        )

        if not (
            inner_train_set
            & inner_val_set
        ):
            audit_pass(
                f"{setting}: inner train and inner val are disjoint"
            )
        else:
            audit_fail(
                f"{setting}: inner train/val overlap"
            )

        if (
            inner_train_set
            | inner_val_set
        ) == outer_train_set:
            audit_pass(
                f"{setting}: inner partitions exactly reconstruct "
                f"outer train"
            )
        else:
            audit_fail(
                f"{setting}: inner partitions do not reconstruct "
                f"outer train"
            )

        if not (
            outer_train_set
            & outer_heldout_set
        ):
            audit_pass(
                f"{setting}: outer train and outer heldout are disjoint"
            )
        else:
            audit_fail(
                f"{setting}: outer train/heldout overlap"
            )

        if len(
            outer_train_set
            | outer_heldout_set
        ) == NUM_ACTIVE_DDI_DRUGS:
            audit_pass(
                f"{setting}: outer partition reconstructs "
                f"1071 DDI-active drugs"
            )
        else:
            audit_fail(
                f"{setting}: outer active-drug reconstruction failed"
            )

        # ---------------------------------------------------------------------
        # MASK AUDIT
        # ---------------------------------------------------------------------

        mask_sources = {
            "inner_train":
                source_inner_train,

            "inner_val":
                source_inner_val,

            "outer_train":
                source_outer_train,

            "outer_heldout":
                source_outer_heldout,
        }

        for name, ids in mask_sources.items():
            mask_path = (
                bio_root
                / f"{name}_drug_mask.pt"
            )

            mask = load_torch(
                mask_path
            )

            if (
                isinstance(mask, torch.Tensor)
                and mask.dtype == torch.bool
                and tuple(mask.shape) == (NUM_DRUGS,)
            ):
                audit_pass(
                    f"{setting}: {name} mask has correct "
                    f"shape/dtype"
                )
            else:
                audit_fail(
                    f"{setting}: {name} mask malformed"
                )
                continue

            expected_mask = torch.zeros(
                NUM_DRUGS,
                dtype=torch.bool,
            )

            expected_mask[
                ids
            ] = True

            if torch.equal(
                mask.cpu(),
                expected_mask,
            ):
                audit_pass(
                    f"{setting}: {name} mask exactly matches IDs"
                )
            else:
                audit_fail(
                    f"{setting}: {name} mask does not match IDs"
                )

        # ---------------------------------------------------------------------
        # FILTERED DTI
        # ---------------------------------------------------------------------

        filtered_dti_path = (
            bio_root
            / "dti_edge_index.pt"
        )

        filtered_dti = normalize_edge_index(
            load_torch(
                filtered_dti_path
            ),
            f"{setting} filtered DTI",
        )

        if (
            filtered_dti.shape[1]
            == expected[
                "inner_train_dti_edges"
            ]
        ):
            audit_pass(
                f"{setting}: filtered DTI edge count = "
                f"{expected['inner_train_dti_edges']}"
            )
        else:
            audit_fail(
                f"{setting}: filtered DTI edge count mismatch "
                f"({filtered_dti.shape[1]})"
            )

        # Every retained DTI edge must have an inner-train drug.
        filtered_drugs = dti_drug_set(
            filtered_dti,
            drug_row,
        )

        if filtered_drugs.issubset(
            inner_train_set
        ):
            audit_pass(
                f"{setting}: every filtered DTI drug endpoint "
                f"is inner-train"
            )
        else:
            audit_fail(
                f"{setting}: filtered DTI contains forbidden drug"
            )

        # Inner val exposure must be exactly zero.
        inner_val_exposure = (
            count_dti_exposure(
                filtered_dti,
                drug_row,
                source_inner_val,
            )
        )

        if inner_val_exposure == 0:
            audit_pass(
                f"{setting}: filtered DTI inner-val exposure = 0"
            )
        else:
            audit_fail(
                f"{setting}: filtered DTI inner-val exposure = "
                f"{inner_val_exposure}"
            )

        # Outer heldout exposure must be exactly zero.
        outer_heldout_exposure = (
            count_dti_exposure(
                filtered_dti,
                drug_row,
                source_outer_heldout,
            )
        )

        if outer_heldout_exposure == 0:
            audit_pass(
                f"{setting}: filtered DTI outer-heldout exposure = 0"
            )
        else:
            audit_fail(
                f"{setting}: filtered DTI outer-heldout exposure = "
                f"{outer_heldout_exposure}"
            )

        # ---------------------------------------------------------------------
        # INDEPENDENTLY RECONSTRUCT FILTERED DTI FROM MASTER
        # ---------------------------------------------------------------------

        master_drug_endpoint = master_dti[
            drug_row
        ]

        allowed_mask = torch.tensor(
            [
                int(x) in inner_train_set
                for x in master_drug_endpoint.tolist()
            ],
            dtype=torch.bool,
        )

        expected_filtered_dti = (
            master_dti[
                :,
                allowed_mask,
            ]
            .contiguous()
        )

        if torch.equal(
            filtered_dti,
            expected_filtered_dti,
        ):
            audit_pass(
                f"{setting}: filtered DTI exactly equals independent "
                f"master-DTI restriction to inner-train drugs"
            )
        else:
            audit_fail(
                f"{setting}: filtered DTI does not exactly match "
                f"independent reconstruction"
            )

        # ---------------------------------------------------------------------
        # MASTER DTI EXPOSURE COUNTS
        # ---------------------------------------------------------------------

        master_inner_train = (
            count_dti_exposure(
                master_dti,
                drug_row,
                source_inner_train,
            )
        )

        master_inner_val = (
            count_dti_exposure(
                master_dti,
                drug_row,
                source_inner_val,
            )
        )

        master_outer_train = (
            count_dti_exposure(
                master_dti,
                drug_row,
                source_outer_train,
            )
        )

        master_outer_heldout = (
            count_dti_exposure(
                master_dti,
                drug_row,
                source_outer_heldout,
            )
        )

        exposure_checks = [
            (
                "inner-train master DTI",
                master_inner_train,
                expected[
                    "inner_train_dti_edges"
                ],
            ),
            (
                "inner-val master DTI",
                master_inner_val,
                expected[
                    "inner_val_master_dti_edges"
                ],
            ),
            (
                "outer-train master DTI",
                master_outer_train,
                expected[
                    "outer_train_dti_edges"
                ],
            ),
            (
                "outer-heldout master DTI",
                master_outer_heldout,
                expected[
                    "outer_heldout_master_dti_edges"
                ],
            ),
        ]

        for name, observed, exp in exposure_checks:
            if observed == exp:
                audit_pass(
                    f"{setting}: {name} count = {exp}"
                )
            else:
                audit_fail(
                    f"{setting}: {name} mismatch "
                    f"observed={observed}, expected={exp}"
                )

        if (
            master_inner_train
            + master_inner_val
            == master_outer_train
        ):
            audit_pass(
                f"{setting}: inner DTI partition exactly "
                f"reconstructs outer-train DTI"
            )
        else:
            audit_fail(
                f"{setting}: DTI partition arithmetic failed"
            )

        # ---------------------------------------------------------------------
        # ZERO BIOLOGICAL INITIALIZATION
        # ---------------------------------------------------------------------

        bio_init_path = (
            bio_root
            / "drug_bio_init_zero_128.pt"
        )

        bio_init = load_torch(
            bio_init_path
        )

        if (
            isinstance(bio_init, torch.Tensor)
            and tuple(bio_init.shape)
            == (NUM_DRUGS, 128)
            and bio_init.dtype
            == torch.float32
        ):
            audit_pass(
                f"{setting}: zero bio init shape/dtype correct"
            )
        else:
            audit_fail(
                f"{setting}: zero bio init malformed"
            )

        if torch.count_nonzero(
            bio_init
        ).item() == 0:
            audit_pass(
                f"{setting}: biological drug initialization "
                f"is exactly all-zero"
            )
        else:
            audit_fail(
                f"{setting}: biological initialization "
                f"contains nonzero values"
            )

        # ---------------------------------------------------------------------
        # SPLIT PAIR COUNTS
        # ---------------------------------------------------------------------

        pair_files = {
            "train_pos":
                split_root / "train_pos.pt",

            "train_neg":
                split_root / "train_neg.pt",

            "val_pos":
                split_root / "val_pos.pt",

            "val_neg":
                split_root / "val_neg.pt",

            "test_pos":
                split_root / "test_pos.pt",

            "test_neg":
                split_root / "test_neg.pt",
        }

        pair_tensors = {}

        for name, path in pair_files.items():
            pair_tensors[name] = normalize_pairs(
                load_torch(path),
                f"{setting}/{name}",
            )

            observed = pair_tensors[
                name
            ].shape[0]

            if observed == expected[name]:
                audit_pass(
                    f"{setting}: {name} count = {expected[name]}"
                )
            else:
                audit_fail(
                    f"{setting}: {name} count mismatch"
                )

        # ---------------------------------------------------------------------
        # DDI TRAIN GRAPH
        # ---------------------------------------------------------------------

        ddi_graph_path = (
            bio_root
            / "ddi_train_edge_index.pt"
        )

        ddi_graph = normalize_edge_index(
            load_torch(
                ddi_graph_path
            ),
            f"{setting} DDI train graph",
        )

        if (
            ddi_graph.shape[1]
            == expected[
                "ddi_directed_edges"
            ]
        ):
            audit_pass(
                f"{setting}: DDI train graph directed-edge count = "
                f"{expected['ddi_directed_edges']}"
            )
        else:
            audit_fail(
                f"{setting}: DDI train graph edge-count mismatch"
            )

        # Reconstruct exact expected directed graph from train_pos.
        train_pos = pair_tensors[
            "train_pos"
        ]

        forward = (
            train_pos.t()
            .contiguous()
        )

        reverse = torch.stack(
            [
                forward[1],
                forward[0],
            ],
            dim=0,
        )

        expected_ddi_graph = torch.cat(
            [
                forward,
                reverse,
            ],
            dim=1,
        )

        if torch.equal(
            ddi_graph,
            expected_ddi_graph,
        ):
            audit_pass(
                f"{setting}: DDI graph exactly equals "
                f"train_pos plus reverse edges"
            )
        else:
            audit_fail(
                f"{setting}: DDI graph differs from exact "
                f"train_pos bidirectional reconstruction"
            )

        ddi_nodes = set(
            int(v)
            for v in torch.unique(
                ddi_graph
            ).tolist()
        )

        if ddi_nodes.issubset(
            inner_train_set
        ):
            audit_pass(
                f"{setting}: DDI graph contains only "
                f"inner-training drugs"
            )
        else:
            audit_fail(
                f"{setting}: DDI graph contains non-inner-train drug"
            )

        # ---------------------------------------------------------------------
        # DDI GRAPH MUST NOT CONTAIN VAL/TEST POSITIVE EDGES
        # ---------------------------------------------------------------------

        train_positive_set = canonical_pair_set(
            pair_tensors[
                "train_pos"
            ]
        )

        val_positive_set = canonical_pair_set(
            pair_tensors[
                "val_pos"
            ]
        )

        test_positive_set = canonical_pair_set(
            pair_tensors[
                "test_pos"
            ]
        )

        if not (
            train_positive_set
            & val_positive_set
        ):
            audit_pass(
                f"{setting}: train-positive and val-positive "
                f"pairs are disjoint"
            )
        else:
            audit_fail(
                f"{setting}: train/val positive overlap"
            )

        if not (
            train_positive_set
            & test_positive_set
        ):
            audit_pass(
                f"{setting}: train-positive and test-positive "
                f"pairs are disjoint"
            )
        else:
            audit_fail(
                f"{setting}: train/test positive overlap"
            )

        if not (
            val_positive_set
            & test_positive_set
        ):
            audit_pass(
                f"{setting}: val-positive and test-positive "
                f"pairs are disjoint"
            )
        else:
            audit_fail(
                f"{setting}: val/test positive overlap"
            )

        # ---------------------------------------------------------------------
        # FULL PAIR SPLIT OVERLAP
        # ---------------------------------------------------------------------

        train_all = (
            canonical_pair_set(
                pair_tensors["train_pos"]
            )
            |
            canonical_pair_set(
                pair_tensors["train_neg"]
            )
        )

        val_all = (
            canonical_pair_set(
                pair_tensors["val_pos"]
            )
            |
            canonical_pair_set(
                pair_tensors["val_neg"]
            )
        )

        test_all = (
            canonical_pair_set(
                pair_tensors["test_pos"]
            )
            |
            canonical_pair_set(
                pair_tensors["test_neg"]
            )
        )

        if not (
            train_all & val_all
        ):
            audit_pass(
                f"{setting}: no train/val pair overlap"
            )
        else:
            audit_fail(
                f"{setting}: train/val pair overlap detected"
            )

        if not (
            train_all & test_all
        ):
            audit_pass(
                f"{setting}: no train/test pair overlap"
            )
        else:
            audit_fail(
                f"{setting}: train/test pair overlap detected"
            )

        if not (
            val_all & test_all
        ):
            audit_pass(
                f"{setting}: no val/test pair overlap"
            )
        else:
            audit_fail(
                f"{setting}: val/test pair overlap detected"
            )

        # ---------------------------------------------------------------------
        # LABEL CONTRADICTIONS
        # ---------------------------------------------------------------------

        train_pos_set = canonical_pair_set(
            pair_tensors["train_pos"]
        )

        train_neg_set = canonical_pair_set(
            pair_tensors["train_neg"]
        )

        val_pos_set = canonical_pair_set(
            pair_tensors["val_pos"]
        )

        val_neg_set = canonical_pair_set(
            pair_tensors["val_neg"]
        )

        test_pos_set = canonical_pair_set(
            pair_tensors["test_pos"]
        )

        test_neg_set = canonical_pair_set(
            pair_tensors["test_neg"]
        )

        contradiction_groups = [
            (
                "train",
                train_pos_set,
                train_neg_set,
            ),
            (
                "val",
                val_pos_set,
                val_neg_set,
            ),
            (
                "test",
                test_pos_set,
                test_neg_set,
            ),
        ]

        for split_name, pos, neg in contradiction_groups:
            if not (
                pos & neg
            ):
                audit_pass(
                    f"{setting}: no {split_name} "
                    f"positive/negative contradiction"
                )
            else:
                audit_fail(
                    f"{setting}: {split_name} label contradiction"
                )

        # ---------------------------------------------------------------------
        # COLD-START VALIDATION SEMANTICS
        # ---------------------------------------------------------------------

        def unseen_count_distribution(
            pair_tensor,
            unseen_set,
        ):
            counts = {
                0: 0,
                1: 0,
                2: 0,
            }

            for a, b in pair_tensor.tolist():
                n = (
                    int(a) in unseen_set
                ) + (
                    int(b) in unseen_set
                )

                counts[int(n)] += 1

            return counts

        val_pos_dist = unseen_count_distribution(
            pair_tensors["val_pos"],
            inner_val_set,
        )

        val_neg_dist = unseen_count_distribution(
            pair_tensors["val_neg"],
            inner_val_set,
        )

        test_pos_dist = unseen_count_distribution(
            pair_tensors["test_pos"],
            outer_heldout_set,
        )

        test_neg_dist = unseen_count_distribution(
            pair_tensors["test_neg"],
            outer_heldout_set,
        )

        if setting == "cold_start_S1":
            expected_val_dist = {
                0: 0,
                1: expected["val_pos"],
                2: 0,
            }

            expected_val_neg_dist = {
                0: 0,
                1: expected["val_neg"],
                2: 0,
            }

            expected_test_dist = {
                0: 0,
                1: expected["test_pos"],
                2: 0,
            }

            expected_test_neg_dist = {
                0: 0,
                1: expected["test_neg"],
                2: 0,
            }

        else:
            expected_val_dist = {
                0: 0,
                1: 0,
                2: expected["val_pos"],
            }

            expected_val_neg_dist = {
                0: 0,
                1: 0,
                2: expected["val_neg"],
            }

            expected_test_dist = {
                0: 0,
                1: 0,
                2: expected["test_pos"],
            }

            expected_test_neg_dist = {
                0: 0,
                1: 0,
                2: expected["test_neg"],
            }

        if val_pos_dist == expected_val_dist:
            audit_pass(
                f"{setting}: validation positive cold-start "
                f"semantics correct: {val_pos_dist}"
            )
        else:
            audit_fail(
                f"{setting}: validation positive semantics wrong: "
                f"{val_pos_dist}"
            )

        if val_neg_dist == expected_val_neg_dist:
            audit_pass(
                f"{setting}: validation negative cold-start "
                f"semantics correct: {val_neg_dist}"
            )
        else:
            audit_fail(
                f"{setting}: validation negative semantics wrong: "
                f"{val_neg_dist}"
            )

        if test_pos_dist == expected_test_dist:
            audit_pass(
                f"{setting}: outer-test positive semantics remain "
                f"correct: {test_pos_dist}"
            )
        else:
            audit_fail(
                f"{setting}: outer-test positive semantics wrong: "
                f"{test_pos_dist}"
            )

        if test_neg_dist == expected_test_neg_dist:
            audit_pass(
                f"{setting}: outer-test negative semantics remain "
                f"correct: {test_neg_dist}"
            )
        else:
            audit_fail(
                f"{setting}: outer-test negative semantics wrong: "
                f"{test_neg_dist}"
            )

        # ---------------------------------------------------------------------
        # SETTING MANIFEST
        # ---------------------------------------------------------------------

        setting_manifest_path = (
            bio_root
            / "manifest.json"
        )

        setting_manifest = load_json(
            setting_manifest_path
        )

        manifest_filtered_hash = (
            setting_manifest[
                "filtered_dti"
            ][
                "sha256"
            ]
        )

        if (
            manifest_filtered_hash
            == sha256_tensor(
                filtered_dti
            )
        ):
            audit_pass(
                f"{setting}: filtered DTI hash matches manifest"
            )
        else:
            audit_fail(
                f"{setting}: filtered DTI hash mismatch"
            )

        manifest_ddi_hash = (
            setting_manifest[
                "ddi_train_graph"
            ][
                "sha256"
            ]
        )

        if (
            manifest_ddi_hash
            == sha256_tensor(
                ddi_graph
            )
        ):
            audit_pass(
                f"{setting}: DDI train graph hash matches manifest"
            )
        else:
            audit_fail(
                f"{setting}: DDI graph hash mismatch"
            )

        if (
            setting_manifest[
                "drug_bio_initialization"
            ][
                "old_dti_derived_features_used"
            ]
            is False
        ):
            audit_pass(
                f"{setting}: manifest confirms old DTI-derived "
                f"bio features not reused"
            )
        else:
            audit_fail(
                f"{setting}: manifest indicates old bio features used"
            )
              # =========================================================================
    # CROSS-SETTING CHECK
    # =========================================================================

    print()
    print("=" * 110)
    print("CROSS-SETTING CHECK")
    print("=" * 110)

    def load_drug_set(setting, filename):
        return tensor_set(
            normalize_ids(
                load_torch(
                    DEV_SPLIT_ROOT
                    / setting
                    / filename
                ),
                f"{setting}/{filename}",
            )
        )

    # -------------------------------------------------------------------------
    # OUTER PARTITIONS
    # -------------------------------------------------------------------------

    s1_outer_train = load_drug_set(
        "cold_start_S1",
        "outer_train_drugs.pt",
    )

    s2_outer_train = load_drug_set(
        "cold_start_S2",
        "outer_train_drugs.pt",
    )

    s1_outer_heldout = load_drug_set(
        "cold_start_S1",
        "outer_heldout_drugs.pt",
    )

    s2_outer_heldout = load_drug_set(
        "cold_start_S2",
        "outer_heldout_drugs.pt",
    )

    outer_train_overlap = len(
        s1_outer_train
        & s2_outer_train
    )

    outer_heldout_overlap = len(
        s1_outer_heldout
        & s2_outer_heldout
    )

    if outer_train_overlap == 685:
        audit_pass(
            "S1/S2 outer-train intersection = 685"
        )
    else:
        audit_fail(
            f"unexpected S1/S2 outer-train intersection: "
            f"{outer_train_overlap}"
        )

    if outer_heldout_overlap == 44:
        audit_pass(
            "S1/S2 outer-heldout intersection = 44"
        )
    else:
        audit_fail(
            f"unexpected S1/S2 outer-heldout intersection: "
            f"{outer_heldout_overlap}"
        )

    # -------------------------------------------------------------------------
    # INNER PARTITIONS
    # -------------------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # S1 and S2 have independent setting-specific outer partitions.
    # Their 685/171 inner development partitions were also created
    # setting-specifically.
    #
    # Therefore there is NO methodological requirement that:
    #
    #     S1 inner_train == S2 inner_train
    #
    # or that their intersection equals 685.
    #
    # We report these overlaps for provenance only and do not use them
    # as pass/fail conditions.
    # -------------------------------------------------------------------------

    s1_inner_train = load_drug_set(
        "cold_start_S1",
        "inner_train_drugs.pt",
    )

    s2_inner_train = load_drug_set(
        "cold_start_S2",
        "inner_train_drugs.pt",
    )

    s1_inner_val = load_drug_set(
        "cold_start_S1",
        "inner_val_drugs.pt",
    )

    s2_inner_val = load_drug_set(
        "cold_start_S2",
        "inner_val_drugs.pt",
    )

    inner_train_overlap = len(
        s1_inner_train
        & s2_inner_train
    )

    inner_val_overlap = len(
        s1_inner_val
        & s2_inner_val
    )

    print(
        "INFO: S1/S2 inner-train intersection =",
        inner_train_overlap,
    )

    print(
        "INFO: S1/S2 inner-val intersection =",
        inner_val_overlap,
    )

    audit_pass(
        "S1 and S2 retain independent setting-specific "
        "inner development partitions"
    )

    # -------------------------------------------------------------------------
    # EACH INNER PARTITION MUST STILL RECONSTRUCT ITS OWN OUTER TRAIN
    # -------------------------------------------------------------------------

    if (
        s1_inner_train
        | s1_inner_val
    ) == s1_outer_train:
        audit_pass(
            "S1 inner train + inner val exactly reconstruct "
            "S1 outer train"
        )
    else:
        audit_fail(
            "S1 inner partitions do not reconstruct "
            "S1 outer train"
        )

    if (
        s2_inner_train
        | s2_inner_val
    ) == s2_outer_train:
        audit_pass(
            "S2 inner train + inner val exactly reconstruct "
            "S2 outer train"
        )
    else:
        audit_fail(
            "S2 inner partitions do not reconstruct "
            "S2 outer train"
        )

    if not (
        s1_inner_train
        & s1_inner_val
    ):
        audit_pass(
            "S1 inner train and inner val remain disjoint"
        )
    else:
        audit_fail(
            "S1 inner train/val overlap detected"
        )

    if not (
        s2_inner_train
        & s2_inner_val
    ):
        audit_pass(
            "S2 inner train and inner val remain disjoint"
        )
    else:
        audit_fail(
            "S2 inner train/val overlap detected"
        )

    # FINAL RESULT
    # =========================================================================

    print()
    print("=" * 110)
    print("BIOLOGICAL DEVELOPMENT AUDIT SUMMARY")
    print("=" * 110)

    print(
        f"PASS checks : {PASS_COUNT}"
    )

    print(
        f"Warnings    : {WARNING_COUNT}"
    )

    print(
        f"Failures    : {FAIL_COUNT}"
    )

    print()

    if FAIL_COUNT == 0:
        print(
            "BIOLOGICAL DEVELOPMENT CERTIFICATION: PASSED"
        )

        print(
            "DTI/DDI DEVELOPMENT LAYER IS SAFE FOR "
            "STRICT COLD-START V2."
        )

        print(
            "INNER-VALIDATION DTI EXPOSURE: ZERO."
        )

        print(
            "OUTER-HELDOUT DTI EXPOSURE: ZERO."
        )

        print(
            "OLD DTI-DERIVED BIO128 FEATURES: NOT USED."
        )

        print(
            "OUTER TEST: STILL SEALED."
        )

        if ppi_status != "loaded":
            print()
            print(
                "IMPORTANT LIMITATION:"
            )

            print(
                "PPI HAS NOT YET BEEN LOCATED/CERTIFIED."
            )

            print(
                "DO NOT START PPI-ENABLED OR FULL AM-DDI "
                "TRAINING YET."
            )

        print()
        print(
            "SAFE NEXT STEP: locate and independently "
            "audit the canonical PPI artifact."
        )

    else:
        print(
            "BIOLOGICAL DEVELOPMENT CERTIFICATION: FAILED"
        )

        print(
            "DO NOT TRAIN ANY BIOLOGICAL OR MULTIMODAL MODEL."
        )

        raise RuntimeError(
            f"Script 113 found {FAIL_COUNT} failure(s)."
        )


if __name__ == "__main__":
    main()
