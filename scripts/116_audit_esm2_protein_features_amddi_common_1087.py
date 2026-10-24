from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

ESM_TENSOR = (
    ROOT
    / "features"
    / "protein_sequence"
    / "esm2_protein_embeddings.pt"
)

ESM_MAPPING = (
    ROOT
    / "features"
    / "protein_sequence"
    / "esm2_protein_mapping.csv"
)

DTI_MAPPING = (
    ROOT
    / "graphs"
    / "dti_graph"
    / "protein_node_mapping.csv"
)

PPI_MAPPING = (
    ROOT
    / "graphs"
    / "dpddi"
    / "protein_mapping.csv"
)

MASTER_DTI = (
    ROOT
    / "graphs"
    / "amddi_common"
    / "dti_edge_index.pt"
)

HIST_ROOT = (
    ROOT
    / "graphs"
    / "amddi_common_1087_leakage_free_training"
)

BIO_ROOT = (
    ROOT
    / "data"
    / "amddi_common_1087_biological_dev_v2"
)

OUT_JSON = (
    BIO_ROOT
    / "esm2_protein_feature_certification.json"
)

SETTINGS = [
    "transductive",
    "cold_start_S1",
    "cold_start_S2",
]

EXPECTED_SHAPE = (2321, 128)
EXPECTED_NUM_PROTEINS = 2321

PASS = 0
FAIL = 0
WARN = 0


def passed(msg):
    global PASS
    PASS += 1
    print(f"PASS: {msg}")


def failed(msg):
    global FAIL
    FAIL += 1
    print(f"FAIL: {msg}")


def warned(msg):
    global WARN
    WARN += 1
    print(f"WARNING: {msg}")


def check(condition, msg):
    if condition:
        passed(msg)
        return True

    failed(msg)
    return False


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


def sha256_tensor(x):
    x = (
        x.detach()
        .cpu()
        .contiguous()
    )

    return hashlib.sha256(
        x.numpy().tobytes()
    ).hexdigest()


def protein_id_column(df):
    candidates = [
        "protein_id",
        "protein",
        "uniprot_id",
        "uniprot",
        "Protein",
        "Protein_ID",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    # Conservative fallback:
    # find string-like column containing accession-looking values.
    for c in df.columns:
        s = df[c].astype(str)

        if s.str.match(
            r"^[A-Z0-9]{5,10}$"
        ).mean() > 0.8:
            return c

    raise RuntimeError(
        f"Could not identify protein-ID column: "
        f"{list(df.columns)}"
    )


def index_column(df):
    preferred = [
        "protein_index",
        "protein_node_id",
        "node_id",
        "local_id",
        "index",
    ]

    for c in preferred:
        if c in df.columns:
            return c

    return None


print("=" * 110)
print("AM-DDI COMMON-1087")
print("SCRIPT 116 — ESM2 PROTEIN FEATURE PROVENANCE / ALIGNMENT AUDIT")
print("=" * 110)


# =============================================================================
# REQUIRED PATHS
# =============================================================================

required = [
    ESM_TENSOR,
    ESM_MAPPING,
    DTI_MAPPING,
    PPI_MAPPING,
    MASTER_DTI,
    HIST_ROOT,
    BIO_ROOT,
]

all_ok = True

for p in required:
    if not p.exists():
        failed(
            f"required path missing: {p}"
        )
        all_ok = False

if not all_ok:
    print("DO NOT TRAIN.")
    sys.exit(1)

passed(
    "all required ESM2/mapping/graph artifacts exist"
)


# =============================================================================
# ESM2 TENSOR
# =============================================================================

esm = load_torch(
    ESM_TENSOR
)

check(
    isinstance(esm, torch.Tensor),
    "ESM2 embedding artifact is a tensor",
)

if not isinstance(esm, torch.Tensor):
    sys.exit(1)

esm = (
    esm.detach()
    .cpu()
    .float()
    .contiguous()
)

check(
    tuple(esm.shape)
    == EXPECTED_SHAPE,
    "ESM2 tensor shape exactly (2321,128)",
)

check(
    bool(
        torch.isfinite(esm).all()
    ),
    "ESM2 tensor contains only finite values",
)

check(
    torch.count_nonzero(
        esm
    ).item() > 0,
    "ESM2 tensor is non-zero",
)

esm_hash = sha256_tensor(
    esm
)

print()
print("ESM2 tensor")
print("-" * 110)
print("shape       :", tuple(esm.shape))
print("dtype       :", esm.dtype)
print("nonzero     :", int(torch.count_nonzero(esm)))
print("mean        :", float(esm.mean()))
print("std         :", float(esm.std()))
print("SHA256      :", esm_hash)


# =============================================================================
# MAPPINGS
# =============================================================================

esm_map = pd.read_csv(
    ESM_MAPPING
)

dti_map = pd.read_csv(
    DTI_MAPPING
)

ppi_map = pd.read_csv(
    PPI_MAPPING
)

esm_id_col = protein_id_column(
    esm_map
)

dti_id_col = protein_id_column(
    dti_map
)

ppi_id_col = protein_id_column(
    ppi_map
)

esm_ids = (
    esm_map[esm_id_col]
    .astype(str)
    .tolist()
)

dti_ids = (
    dti_map[dti_id_col]
    .astype(str)
    .tolist()
)

ppi_ids = (
    ppi_map[ppi_id_col]
    .astype(str)
    .tolist()
)

esm_set = set(
    esm_ids
)

dti_set = set(
    dti_ids
)

ppi_set = set(
    ppi_ids
)

check(
    len(esm_ids)
    == EXPECTED_NUM_PROTEINS,
    "ESM2 mapping contains exactly 2321 rows",
)

check(
    len(esm_set)
    == EXPECTED_NUM_PROTEINS,
    "ESM2 protein IDs are unique",
)

check(
    esm_set.issubset(
        dti_set
    ),
    "all ESM2 proteins occur in DTI mapping",
)

check(
    ppi_set.issubset(
        esm_set
    ),
    "all PPI-mapped proteins occur in ESM2 mapping",
)

check(
    len(
        dti_set
        - esm_set
    )
    == 3,
    "DTI mapping has exactly three proteins outside ESM2/common namespace",
)

expected_dti_only = {
    "P04745",
    "P62158",
    "Q13748",
}

check(
    dti_set
    - esm_set
    == expected_dti_only,
    "three DTI-only proteins are exactly P04745/P62158/Q13748",
)

print()
print("Mapping diagnostics")
print("-" * 110)
print("ESM2 proteins       :", len(esm_set))
print("DTI proteins        :", len(dti_set))
print("PPI proteins        :", len(ppi_set))
print("ESM2 ∩ DTI          :", len(esm_set & dti_set))
print("ESM2 ∩ PPI          :", len(esm_set & ppi_set))
print("DTI-only            :", sorted(dti_set - esm_set))


# =============================================================================
# ESM LOCAL INDEX
# =============================================================================

esm_index_col = index_column(
    esm_map
)

if esm_index_col is None:
    failed(
        "could not identify ESM2 local protein-index column"
    )

else:
    esm_indices = (
        esm_map[esm_index_col]
        .astype(int)
        .tolist()
    )

    check(
        esm_indices
        == list(
            range(
                EXPECTED_NUM_PROTEINS
            )
        ),
        "ESM2 local protein indices are exactly 0..2320 in row order",
    )


# =============================================================================
# DTI COMMON ORDER
# =============================================================================

dti_common_order = [
    protein_id
    for protein_id in dti_ids
    if protein_id in esm_set
]

check(
    dti_common_order
    == esm_ids,
    "DTI common-protein order exactly equals ESM2 row order",
)


# =============================================================================
# MASTER DTI INDEX BOUNDS
# =============================================================================

master_dti = load_torch(
    MASTER_DTI
)

if (
    master_dti.ndim == 2
    and master_dti.shape[0] != 2
    and master_dti.shape[1] == 2
):
    master_dti = (
        master_dti
        .t()
        .contiguous()
    )

master_dti = (
    master_dti
    .detach()
    .cpu()
    .long()
)

check(
    tuple(master_dti.shape)
    == (2, 7408),
    "master common-1087 DTI shape exactly (2,7408)",
)

protein_row = master_dti[1]

check(
    int(
        protein_row.min()
    ) >= 0,
    "master DTI protein IDs are non-negative",
)

check(
    int(
        protein_row.max()
    ) < EXPECTED_NUM_PROTEINS,
    "master DTI protein IDs fit ESM2 0..2320 namespace",
)


# =============================================================================
# STRONGEST ALIGNMENT TEST:
# HISTORICAL GRAPH protein.x MUST EXACTLY EQUAL ESM2 TENSOR
# =============================================================================

print()
print("=" * 110)
print("HISTORICAL COMMON-1087 PROTEIN FEATURE EXACT-MATCH")
print("=" * 110)

historical_results = {}

for setting in SETTINGS:

    graph_path = (
        HIST_ROOT
        / f"{setting}_train_graph.pt"
    )

    if not graph_path.exists():
        failed(
            f"{setting}: historical graph missing"
        )
        continue

    graph = load_torch(
        graph_path
    )

    if not hasattr(
        graph,
        "node_types",
    ):
        failed(
            f"{setting}: object is not heterogeneous graph"
        )
        continue

    if "protein" not in graph.node_types:
        failed(
            f"{setting}: protein node type missing"
        )
        continue

    protein_x = graph[
        "protein"
    ].x

    if protein_x is None:
        failed(
            f"{setting}: protein.x missing"
        )
        continue

    protein_x = (
        protein_x
        .detach()
        .cpu()
        .float()
        .contiguous()
    )

    shape_ok = (
        tuple(protein_x.shape)
        == EXPECTED_SHAPE
    )

    exact = (
        shape_ok
        and torch.equal(
            protein_x,
            esm,
        )
    )

    max_diff = None

    if shape_ok:
        max_diff = float(
            (
                protein_x
                - esm
            )
            .abs()
            .max()
        )

    check(
        shape_ok,
        f"{setting}: historical protein.x shape = (2321,128)",
    )

    check(
        exact,
        f"{setting}: historical protein.x exactly equals ESM2 tensor",
    )

    if shape_ok:
        check(
            max_diff == 0.0,
            f"{setting}: historical protein.x max absolute difference = 0",
        )

    historical_results[
        setting
    ] = {
        "graph":
            str(
                graph_path
            ),

        "shape":
            list(
                protein_x.shape
            ),

        "hash":
            sha256_tensor(
                protein_x
            ),

        "exact_match":
            bool(
                exact
            ),

        "max_abs_difference":
            max_diff,
    }


# =============================================================================
# PROVENANCE CHECK
# =============================================================================

print()
print("=" * 110)
print("FEATURE PROVENANCE")
print("=" * 110)

encoder_script = (
    ROOT
    / "scripts"
    / "63_esm2_protein_encoder.py"
)

if not encoder_script.exists():
    failed(
        "ESM2 construction script missing"
    )

else:
    text = encoder_script.read_text(
        errors="ignore"
    ).lower()

    check(
        "esm" in text,
        "construction script explicitly uses ESM model",
    )

    # These are conservative provenance indicators.
    suspicious_terms = [
        "train_pos",
        "test_pos",
        "val_pos",
        "ddi_edge",
        "ddi_graph",
    ]

    found_suspicious = [
        term
        for term in suspicious_terms
        if term in text
    ]

    check(
        len(
            found_suspicious
        )
        == 0,
        "ESM2 construction script has no DDI pair/label dependencies",
    )

    if found_suspicious:
        print(
            "Suspicious terms:",
            found_suspicious,
        )

    # DTI use should also not be part of sequence embedding generation.
    dti_dependency_terms = [
        "dti_edge_index",
        "drug_protein_embeddings",
    ]

    found_dti_dependency = [
        term
        for term in dti_dependency_terms
        if term in text
    ]

    check(
        len(
            found_dti_dependency
        )
        == 0,
        "ESM2 construction script has no DTI-derived embedding dependency",
    )

    if found_dti_dependency:
        print(
            "DTI-related terms:",
            found_dti_dependency,
        )


# =============================================================================
# WRITE CERTIFICATION RECORD ONLY IF CLEAN
# =============================================================================

certification = {
    "certification":
        "esm2_protein_features_common_1087",

    "esm2_path":
        str(
            ESM_TENSOR
        ),

    "esm2_mapping":
        str(
            ESM_MAPPING
        ),

    "shape":
        list(
            esm.shape
        ),

    "sha256":
        esm_hash,

    "num_proteins":
        EXPECTED_NUM_PROTEINS,

    "feature_dim":
        128,

    "dti_only_proteins_excluded_from_current_namespace":
        sorted(
            dti_set
            - esm_set
        ),

    "mapping_checks": {
        "esm_unique":
            len(
                esm_set
            )
            == EXPECTED_NUM_PROTEINS,

        "ppi_subset_of_esm":
            ppi_set.issubset(
                esm_set
            ),

        "esm_subset_of_dti_mapping":
            esm_set.issubset(
                dti_set
            ),

        "dti_common_order_matches_esm":
            dti_common_order
            == esm_ids,
    },

    "historical_exact_matches":
        historical_results,

    "policy": {
        "intrinsic_protein_sequence_features":
            True,

        "drug_split_specific":
            False,

        "ddi_label_derived":
            False,

        "dti_topology_derived":
            False,

        "safe_for_s1_s2_common_protein_namespace":
            FAIL == 0,
    },

    "status":
        (
            "PASSED"
            if FAIL == 0
            and WARN == 0
            else "FAILED"
        ),
}


print()
print("=" * 110)
print("SCRIPT 116 SUMMARY")
print("=" * 110)

print(
    f"PASS checks : {PASS}"
)

print(
    f"Warnings    : {WARN}"
)

print(
    f"Failures    : {FAIL}"
)

if FAIL == 0 and WARN == 0:

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = OUT_JSON.with_suffix(
        ".json.tmp"
    )

    with open(
        tmp,
        "w",
    ) as f:
        json.dump(
            certification,
            f,
            indent=2,
            sort_keys=True,
        )

    tmp.replace(
        OUT_JSON
    )

    print()
    print(
        "ESM2 PROTEIN FEATURE CERTIFICATION: PASSED"
    )

    print(
        "PROTEIN FEATURE SOURCE: ESM2 SEQUENCE EMBEDDINGS."
    )

    print(
        "PROTEIN FEATURE SHAPE: (2321,128)."
    )

    print(
        "DTI/PPI LOCAL PROTEIN NAMESPACE ALIGNMENT: PASSED."
    )

    print(
        "HISTORICAL COMMON-1087 ROW ALIGNMENT: PASSED."
    )

    print(
        "DDI LABEL DEPENDENCY: NONE DETECTED."
    )

    print(
        "DTI-DERIVED DRUG BIO FEATURES: NOT USED."
    )

    print(
        "OUTER TEST: NOT USED."
    )

    print()
    print(
        f"Certification written to:\n{OUT_JSON}"
    )

    print()
    print(
        "SAFE NEXT STEP: biological HGT architecture audit."
    )

    sys.exit(0)

else:

    print()
    print(
        "ESM2 PROTEIN FEATURE CERTIFICATION: FAILED"
    )

    print(
        "DO NOT USE THESE FEATURES FOR TRAINING."
    )

    sys.exit(1)
