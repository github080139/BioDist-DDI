from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import sys

import pandas as pd
import torch


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_ROOT = ROOT / "data/amddi_common_1087_coldstart_dev_splits_v2"
BIO_V2_ROOT = ROOT / "data/amddi_common_1087_biological_dev_v2"

COMMON_DRUG_MAP = ROOT / "data/amddi_common/drug_mapping.csv"

RAW_DTI_EDGE = ROOT / "graphs/dti_graph/edge_index.pt"
DTI_DRUG_MAP = ROOT / "graphs/dti_graph/drug_node_mapping.csv"
DTI_PROTEIN_MAP = ROOT / "graphs/dti_graph/protein_node_mapping.csv"

ESM_MAP = ROOT / "features/protein_sequence/esm2_protein_mapping.csv"
ESM_X = ROOT / "features/protein_sequence/esm2_protein_embeddings.pt"

PPI_SEMANTIC = ROOT / "graphs/dpddi/ppi_uniprot_edges.csv"

OUT_ROOT = ROOT / "data/amddi_common_1087_biological_dev_v4"

SETTINGS = [
    "cold_start_S1",
    "cold_start_S2",
]


# =============================================================================
# EXPECTED CERTIFIED VALUES
# =============================================================================

NUM_DRUGS = 1087
NUM_ACTIVE_DRUGS = 1071

NUM_PROTEINS = 2321
PROTEIN_DIM = 128

RAW_DTI_RELATIONS = 15138

EXPECTED_COMMON_DTI_BEFORE_ESM_FILTER = 7408
EXPECTED_COMMON_DTI_AFTER_ESM_FILTER = 7392

EXPECTED_DTI_MISSING_ESM_COMMON = 16

EXPECTED_MISSING_PROTEINS = {
    "P04745",
    "P62158",
    "Q13748",
}

EXPECTED_GLOBAL_MISSING_DTI = {
    "P04745": 5,
    "P62158": 25,
    "Q13748": 3,
}

EXPECTED_PPI_ROWS = 53972
EXPECTED_PPI_DROPPED_ROWS = 392
EXPECTED_PPI_KEPT_ROWS = 53580
EXPECTED_PPI_DIRECTED = 107160
EXPECTED_PPI_PROTEINS = 2263

EXPECTED_SETTING_DTI = {
    "cold_start_S1": 4805,
    "cold_start_S2": 4695,
}

EXPECTED_SETTING_DDI = {
    "cold_start_S1": 28988,
    "cold_start_S2": 33806,
}


# =============================================================================
# AUDIT HELPERS
# =============================================================================

PASS = 0
FAIL = 0


def passed(msg):
    global PASS
    PASS += 1
    print(f"PASS: {msg}")


def failed(msg):
    global FAIL
    FAIL += 1
    print(f"FAIL: {msg}")


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


def normalize_edge_index(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{name}: expected tensor"
        )

    x = x.detach().cpu().long().contiguous()

    if x.ndim != 2:
        raise RuntimeError(
            f"{name}: malformed shape {tuple(x.shape)}"
        )

    if x.shape[0] == 2:
        return x

    if x.shape[1] == 2:
        return x.t().contiguous()

    raise RuntimeError(
        f"{name}: expected [2,E] or [E,2]"
    )


def tensor_hash(x):
    x = x.detach().cpu().contiguous()

    return hashlib.sha256(
        x.numpy().tobytes()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def atomic_torch_save(x, path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        x,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json_save(obj, path):
    path = Path(path)

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


def tensor_set(x):
    return set(
        int(v)
        for v in x.detach().cpu().view(-1).tolist()
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 110)
print("SCRIPT 116b")
print("AM-DDI COMMON-1087 — SEMANTIC BIOLOGICAL DEVELOPMENT LAYER V4")
print("=" * 110)

print()
print("Architecture preserved:")
print("Drug -- DDI -- Drug")
print(" |             |-- Morgan fingerprint")
print("DTI            |-- Molecular structure")
print(" |")
print("Protein -- PPI -- Protein")
print("                  |-- Sequence / ESM2")
print()

print(
    "Protein node identity is defined ONLY by "
    "ESM2 protein_index / UniProt mapping."
)

print(
    "No raw numeric protein-offset assumption is permitted."
)

print()


# =============================================================================
# FAIL IF V4 ALREADY EXISTS
# =============================================================================

if OUT_ROOT.exists():
    print(
        f"ERROR: {OUT_ROOT} already exists."
    )

    print(
        "Refusing to overwrite versioned v4 artifacts."
    )

    sys.exit(1)


# =============================================================================
# REQUIRED SOURCES
# =============================================================================

required = [
    SPLIT_ROOT,
    BIO_V2_ROOT,
    COMMON_DRUG_MAP,
    RAW_DTI_EDGE,
    DTI_DRUG_MAP,
    DTI_PROTEIN_MAP,
    ESM_MAP,
    ESM_X,
    PPI_SEMANTIC,
]

for path in required:
    if not path.exists():
        print(
            "MISSING:",
            path,
        )
        sys.exit(1)

passed(
    "all required source artifacts exist"
)


# =============================================================================
# 1. ESM2 CANONICAL PROTEIN NAMESPACE
# =============================================================================

print()
print("=" * 110)
print("1. CANONICAL ESM2 PROTEIN NAMESPACE")
print("=" * 110)

esm_map = pd.read_csv(
    ESM_MAP
)

esm_x = load_torch(
    ESM_X
).detach().cpu().float()

check(
    len(esm_map) == NUM_PROTEINS,
    "ESM mapping contains exactly 2321 proteins",
)

check(
    esm_map["protein_index"].astype(int).tolist()
    == list(range(NUM_PROTEINS)),
    "ESM protein_index exactly 0..2320",
)

check(
    esm_map["protein_index"].is_unique,
    "ESM protein indices unique",
)

check(
    esm_map["protein_id"].astype(str).is_unique,
    "ESM protein IDs unique",
)

check(
    tuple(esm_x.shape)
    == (
        NUM_PROTEINS,
        PROTEIN_DIM,
    ),
    "ESM feature tensor shape exactly (2321,128)",
)

check(
    bool(
        torch.isfinite(
            esm_x
        ).all()
    ),
    "ESM feature tensor finite",
)

protein_to_esm = dict(
    zip(
        esm_map["protein_id"].astype(str),
        esm_map["protein_index"].astype(int),
    )
)

esm_ids = set(
    protein_to_esm
)


# =============================================================================
# 2. COMMON DRUG NAMESPACE
# =============================================================================

print()
print("=" * 110)
print("2. COMMON-1087 DRUG NAMESPACE")
print("=" * 110)

common = pd.read_csv(
    COMMON_DRUG_MAP
)

check(
    len(common) == NUM_DRUGS,
    "common drug mapping contains 1087 drugs",
)

check(
    common["node_id"].astype(int).tolist()
    == list(range(NUM_DRUGS)),
    "common drug IDs exactly 0..1086",
)

check(
    common["drug_id"].astype(str).is_unique,
    "common drug identifiers unique",
)

drug_to_common = dict(
    zip(
        common["drug_id"].astype(str),
        common["node_id"].astype(int),
    )
)


# =============================================================================
# 3. DTI ORIENTATION + SEMANTIC NODE MAP
# =============================================================================

print()
print("=" * 110)
print("3. NORMALIZE ORIGINAL DTI RELATIONS")
print("=" * 110)

raw_dti = normalize_edge_index(
    load_torch(
        RAW_DTI_EDGE
    ),
    "raw DTI",
)

dti_drug_map = pd.read_csv(
    DTI_DRUG_MAP
)

dti_protein_map = pd.read_csv(
    DTI_PROTEIN_MAP
)

old_drug_to_id = dict(
    zip(
        dti_drug_map["node_id"].astype(int),
        dti_drug_map["drug_id"].astype(str),
    )
)

old_protein_to_id = dict(
    zip(
        dti_protein_map["node_id"].astype(int),
        dti_protein_map["protein_id"].astype(str),
    )
)

drug_nodes = set(
    old_drug_to_id
)

protein_nodes = set(
    old_protein_to_id
)

forward = set()
reverse_normalized = set()
invalid = []

for a, b in raw_dti.t().tolist():

    a = int(a)
    b = int(b)

    if a in drug_nodes and b in protein_nodes:
        forward.add(
            (
                a,
                b,
            )
        )

    elif a in protein_nodes and b in drug_nodes:
        reverse_normalized.add(
            (
                b,
                a,
            )
        )

    else:
        invalid.append(
            (
                a,
                b,
            )
        )

check(
    len(invalid) == 0,
    "raw DTI contains no invalid endpoint orientation",
)

check(
    len(forward)
    == RAW_DTI_RELATIONS,
    "raw DTI has exactly 15138 unique drug-to-protein relations",
)

check(
    len(reverse_normalized)
    == RAW_DTI_RELATIONS,
    "raw DTI has exactly 15138 normalized reverse relations",
)

check(
    forward
    == reverse_normalized,
    "forward and reverse DTI encode exactly the same relations",
)

normalized_dti = sorted(
    forward
)


# =============================================================================
# 4. EXPLAIN DTI PROTEINS WITHOUT ESM
# =============================================================================

print()
print("=" * 110)
print("4. DTI / ESM SEMANTIC DIFFERENCE")
print("=" * 110)

dti_protein_ids = set(
    dti_protein_map["protein_id"].astype(str)
)

missing_esm = (
    dti_protein_ids
    - esm_ids
)

check(
    missing_esm
    == EXPECTED_MISSING_PROTEINS,
    "exactly P04745, P62158, Q13748 are absent from ESM",
)

global_missing_counter = Counter()

for _, old_p in normalized_dti:
    protein_id = old_protein_to_id[
        old_p
    ]

    if protein_id not in esm_ids:
        global_missing_counter[
            protein_id
        ] += 1

print(
    "Global missing-ESM DTI relations:",
    dict(
        sorted(
            global_missing_counter.items()
        )
    ),
)

check(
    dict(
        sorted(
            global_missing_counter.items()
        )
    )
    == EXPECTED_GLOBAL_MISSING_DTI,
    "global missing-ESM DTI counts exactly match independent audit",
)


# =============================================================================
# 5. REBUILD COMMON DTI SEMANTICALLY
# =============================================================================

print()
print("=" * 110)
print("5. REBUILD COMMON-1087 DTI BY IDENTIFIERS")
print("=" * 110)

common_before_esm = []
common_after_esm = []

dropped_noncommon = 0
dropped_missing_esm = Counter()

for old_d, old_p in normalized_dti:

    drug_id = old_drug_to_id[
        old_d
    ]

    protein_id = old_protein_to_id[
        old_p
    ]

    if drug_id not in drug_to_common:
        dropped_noncommon += 1
        continue

    common_drug = int(
        drug_to_common[
            drug_id
        ]
    )

    common_before_esm.append(
        (
            common_drug,
            protein_id,
        )
    )

    if protein_id not in protein_to_esm:
        dropped_missing_esm[
            protein_id
        ] += 1
        continue

    common_after_esm.append(
        (
            common_drug,
            int(
                protein_to_esm[
                    protein_id
                ]
            ),
        )
    )

check(
    len(common_before_esm)
    == EXPECTED_COMMON_DTI_BEFORE_ESM_FILTER,
    "common DTI before ESM filtering = 7408",
)

check(
    sum(
        dropped_missing_esm.values()
    )
    == EXPECTED_DTI_MISSING_ESM_COMMON,
    "exactly 16 common DTI relations removed for missing ESM",
)

print(
    "Common missing-ESM relations:",
    dict(
        sorted(
            dropped_missing_esm.items()
        )
    ),
)

# No duplicate semantic relations should be created.
check(
    len(
        set(
            common_after_esm
        )
    )
    == len(
        common_after_esm
    ),
    "semantic common DTI contains no duplicate relations",
)

check(
    len(common_after_esm)
    == EXPECTED_COMMON_DTI_AFTER_ESM_FILTER,
    "corrected semantic common DTI = 7392 edges",
)

master_dti = torch.tensor(
    sorted(
        common_after_esm
    ),
    dtype=torch.long,
).t().contiguous()

check(
    int(
        master_dti[0].min()
    )
    >= 0
    and int(
        master_dti[0].max()
    )
    < NUM_DRUGS,
    "master DTI drug indices fit common-1087 namespace",
)

check(
    int(
        master_dti[1].min()
    )
    >= 0
    and int(
        master_dti[1].max()
    )
    < NUM_PROTEINS,
    "master DTI protein indices fit ESM namespace",
)


# =============================================================================
# 6. REBUILD PPI SEMANTICALLY
# =============================================================================

print()
print("=" * 110)
print("6. REBUILD PPI BY UNIPROT -> ESM INDEX")
print("=" * 110)

ppi_df = pd.read_csv(
    PPI_SEMANTIC
)

check(
    list(
        ppi_df.columns
    )
    == [
        "protein1",
        "protein2",
    ],
    "PPI semantic source columns correct",
)

check(
    len(
        ppi_df
    )
    == EXPECTED_PPI_ROWS,
    "PPI semantic source has 53972 rows",
)

source_endpoint_ids = (
    set(
        ppi_df["protein1"].astype(str)
    )
    |
    set(
        ppi_df["protein2"].astype(str)
    )
)

ppi_missing = (
    source_endpoint_ids
    - esm_ids
)

check(
    ppi_missing
    == EXPECTED_MISSING_PROTEINS,
    "PPI proteins absent from ESM are exactly the same three proteins",
)

missing_ppi_counter = Counter()
kept_ppi_rows = []

for a, b in zip(
    ppi_df["protein1"].astype(str),
    ppi_df["protein2"].astype(str),
):

    missing = False

    if a not in protein_to_esm:
        missing_ppi_counter[
            a
        ] += 1
        missing = True

    if b not in protein_to_esm:
        missing_ppi_counter[
            b
        ] += 1
        missing = True

    if missing:
        continue

    kept_ppi_rows.append(
        (
            int(
                protein_to_esm[a]
            ),
            int(
                protein_to_esm[b]
            ),
        )
    )

dropped_ppi_rows = (
    len(ppi_df)
    - len(kept_ppi_rows)
)

print(
    "PPI missing protein occurrence counts:",
    dict(
        sorted(
            missing_ppi_counter.items()
        )
    ),
)

check(
    dropped_ppi_rows
    == EXPECTED_PPI_DROPPED_ROWS,
    "exactly 392 PPI rows removed for missing ESM proteins",
)

check(
    len(
        kept_ppi_rows
    )
    == EXPECTED_PPI_KEPT_ROWS,
    "53580 PPI source rows remain",
)

# Preserve source-row semantics exactly.
# Each retained source row contributes forward + reverse.
directed_ppi = []

for a, b in kept_ppi_rows:

    directed_ppi.append(
        (
            a,
            b,
        )
    )

    directed_ppi.append(
        (
            b,
            a,
        )
    )

canonical_ppi = torch.tensor(
    directed_ppi,
    dtype=torch.long,
).t().contiguous()

check(
    canonical_ppi.shape[1]
    == EXPECTED_PPI_DIRECTED,
    "canonical PPI contains exactly 107160 directed edges",
)

ppi_nodes = tensor_set(
    canonical_ppi
)

check(
    len(
        ppi_nodes
    )
    == EXPECTED_PPI_PROTEINS,
    "canonical PPI contains exactly 2263 ESM-aligned proteins",
)

check(
    int(
        canonical_ppi.min()
    )
    >= 0
    and int(
        canonical_ppi.max()
    )
    < NUM_PROTEINS,
    "all canonical PPI endpoints fit ESM namespace",
)

# Verify edge reversal multiplicity exactly.
directed_counter = Counter(
    tuple(x)
    for x in canonical_ppi.t().tolist()
)

reverse_ok = all(
    directed_counter[
        (
            b,
            a,
        )
    ]
    == count
    for (
        a,
        b,
    ), count
    in directed_counter.items()
)

check(
    reverse_ok,
    "canonical PPI has symmetric edge multiplicity",
)


# =============================================================================
# 7. CREATE NEW V4 ROOT
# =============================================================================

print()
print("=" * 110)
print("7. CREATE BIOLOGICAL DEVELOPMENT V4")
print("=" * 110)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

passed(
    "created biological_dev_v4 without modifying v2/v3"
)

atomic_torch_save(
    esm_x,
    OUT_ROOT
    / "protein_features_esm2_128.pt",
)

esm_map.to_csv(
    OUT_ROOT
    / "protein_mapping_esm2.csv",
    index=False,
)

atomic_torch_save(
    master_dti,
    OUT_ROOT
    / "master_dti_edge_index.pt",
)

atomic_torch_save(
    canonical_ppi,
    OUT_ROOT
    / "ppi_edge_index.pt",
)

passed(
    "canonical global biological artifacts saved"
)


# =============================================================================
# 8. SETTING-SPECIFIC DEVELOPMENT DTI / DDI
# =============================================================================

print()
print("=" * 110)
print("8. SETTING-SPECIFIC STRICT COLD-START DEVELOPMENT")
print("=" * 110)

setting_records = {}

for setting in SETTINGS:

    print()
    print("-" * 110)
    print(setting)
    print("-" * 110)

    split_dir = (
        SPLIT_ROOT
        / setting
    )

    old_bio_dir = (
        BIO_V2_ROOT
        / setting
    )

    out_dir = (
        OUT_ROOT
        / setting
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -------------------------------------------------------------------------
    # PARTITIONS
    # -------------------------------------------------------------------------

    partitions = {}

    for stem in [
        "inner_train_drugs",
        "inner_val_drugs",
        "outer_train_drugs",
        "outer_heldout_drugs",
    ]:

        candidates = [
            old_bio_dir
            / f"{stem}.pt",

            split_dir
            / f"{stem}.pt",
        ]

        src = next(
            (
                x
                for x in candidates
                if x.exists()
            ),
            None,
        )

        if src is None:
            raise FileNotFoundError(
                f"{setting}: {stem}"
            )

        x = load_torch(
            src
        ).long().view(-1)

        partitions[
            stem
        ] = x

        atomic_torch_save(
            x,
            out_dir
            / f"{stem}.pt",
        )

    inner_train = tensor_set(
        partitions[
            "inner_train_drugs"
        ]
    )

    inner_val = tensor_set(
        partitions[
            "inner_val_drugs"
        ]
    )

    outer_train = tensor_set(
        partitions[
            "outer_train_drugs"
        ]
    )

    outer_heldout = tensor_set(
        partitions[
            "outer_heldout_drugs"
        ]
    )

    check(
        len(
            inner_train
        )
        == 685,
        f"{setting}: inner train = 685",
    )

    check(
        len(
            inner_val
        )
        == 171,
        f"{setting}: inner val = 171",
    )

    check(
        len(
            outer_train
        )
        == 856,
        f"{setting}: outer train = 856",
    )

    check(
        len(
            outer_heldout
        )
        == 215,
        f"{setting}: outer heldout = 215",
    )

    check(
        inner_train.isdisjoint(
            inner_val
        ),
        f"{setting}: inner train/val disjoint",
    )

    check(
        inner_train
        | inner_val
        == outer_train,
        f"{setting}: inner partitions reconstruct outer train",
    )

    check(
        outer_train.isdisjoint(
            outer_heldout
        ),
        f"{setting}: outer train/heldout disjoint",
    )

    check(
        len(
            outer_train
            | outer_heldout
        )
        == NUM_ACTIVE_DRUGS,
        f"{setting}: outer partitions reconstruct 1071 active drugs",
    )

    # -------------------------------------------------------------------------
    # FILTER CANONICAL DTI TO INNER-TRAIN DRUGS
    # -------------------------------------------------------------------------

    mask = torch.tensor(
        [
            int(d)
            in inner_train
            for d
            in master_dti[
                0
            ].tolist()
        ],
        dtype=torch.bool,
    )

    dev_dti = (
        master_dti[
            :,
            mask,
        ]
        .contiguous()
    )

    check(
        int(
            dev_dti.shape[1]
        )
        == EXPECTED_SETTING_DTI[
            setting
        ],
        (
            f"{setting}: semantic development DTI edges = "
            f"{EXPECTED_SETTING_DTI[setting]}"
        ),
    )

    dev_dti_drugs = tensor_set(
        dev_dti[
            0
        ]
    )

    check(
        dev_dti_drugs.issubset(
            inner_train
        ),
        f"{setting}: DTI contains only inner-train drug endpoints",
    )

    check(
        len(
            dev_dti_drugs
            & inner_val
        )
        == 0,
        f"{setting}: inner-validation DTI exposure = 0",
    )

    check(
        len(
            dev_dti_drugs
            & outer_heldout
        )
        == 0,
        f"{setting}: outer-heldout DTI exposure = 0",
    )

    check(
        int(
            dev_dti[1].max()
        )
        < NUM_PROTEINS,
        f"{setting}: DTI protein endpoints are ESM indices",
    )

    atomic_torch_save(
        dev_dti,
        out_dir
        / "dti_edge_index.pt",
    )

    # -------------------------------------------------------------------------
    # COPY CERTIFIED TRAIN-ONLY DDI
    # -------------------------------------------------------------------------

    ddi_candidates = [
        old_bio_dir
        / "ddi_train_edge_index.pt",

        old_bio_dir
        / "ddi_edge_index.pt",

        old_bio_dir
        / "train_ddi_edge_index.pt",
    ]

    ddi_src = next(
        (
            x
            for x in ddi_candidates
            if x.exists()
        ),
        None,
    )

    if ddi_src is None:
        raise FileNotFoundError(
            f"{setting}: DDI graph missing"
        )

    ddi = normalize_edge_index(
        load_torch(
            ddi_src
        ),
        f"{setting} DDI",
    )

    check(
        int(
            ddi.shape[1]
        )
        == EXPECTED_SETTING_DDI[
            setting
        ],
        (
            f"{setting}: DDI directed edge count = "
            f"{EXPECTED_SETTING_DDI[setting]}"
        ),
    )

    ddi_nodes = tensor_set(
        ddi
    )

    check(
        ddi_nodes.issubset(
            inner_train
        ),
        f"{setting}: DDI contains only inner-train drugs",
    )

    check(
        len(
            ddi_nodes
            & inner_val
        )
        == 0,
        f"{setting}: DDI inner-validation exposure = 0",
    )

    check(
        len(
            ddi_nodes
            & outer_heldout
        )
        == 0,
        f"{setting}: DDI outer-heldout exposure = 0",
    )

    atomic_torch_save(
        ddi,
        out_dir
        / "ddi_train_edge_index.pt",
    )

    # -------------------------------------------------------------------------
    # ZERO DRUG BIO INITIALIZATION
    # -------------------------------------------------------------------------

    zero_candidates = [
        old_bio_dir
        / "drug_bio_init_zero_128.pt",

        BIO_V2_ROOT
        / "drug_bio_init_zero_128.pt",
    ]

    zero_src = next(
        (
            x
            for x in zero_candidates
            if x.exists()
        ),
        None,
    )

    if zero_src is None:
        raise FileNotFoundError(
            f"{setting}: zero bio init missing"
        )

    zero_x = load_torch(
        zero_src
    ).detach().cpu()

    check(
        tuple(
            zero_x.shape
        )
        == (
            NUM_DRUGS,
            128,
        ),
        f"{setting}: drug bio init shape = (1087,128)",
    )

    check(
        int(
            torch.count_nonzero(
                zero_x
            )
        )
        == 0,
        f"{setting}: drug bio initialization exactly zero",
    )

    atomic_torch_save(
        zero_x,
        out_dir
        / "drug_bio_init_zero_128.pt",
    )

    # -------------------------------------------------------------------------
    # COPY SPLIT PAIRS EXACTLY
    # -------------------------------------------------------------------------

    pair_hashes = {}

    for name in [
        "train_pos",
        "train_neg",
        "val_pos",
        "val_neg",
        "test_pos",
        "test_neg",
    ]:

        src = (
            split_dir
            / f"{name}.pt"
        )

        x = load_torch(
            src
        )

        pair_hashes[
            name
        ] = tensor_hash(
            x
        )

        atomic_torch_save(
            x,
            out_dir
            / f"{name}.pt",
        )

    setting_record = {
        "setting":
            setting,

        "inner_train_drugs":
            685,

        "inner_val_drugs":
            171,

        "outer_train_drugs":
            856,

        "outer_heldout_drugs":
            215,

        "development_dti_edges":
            int(
                dev_dti.shape[1]
            ),

        "ddi_directed_edges":
            int(
                ddi.shape[1]
            ),

        "protein_namespace":
            "esm2_protein_index_0_2320",

        "inner_validation_dti_exposure":
            0,

        "outer_heldout_dti_exposure":
            0,

        "old_dti_drug_bio_features_used":
            False,

        "outer_test_used":
            False,

        "pair_tensor_hashes":
            pair_hashes,
    }

    atomic_json_save(
        setting_record,
        out_dir
        / "manifest.json",
    )

    setting_records[
        setting
    ] = setting_record


# =============================================================================
# 9. ROOT MANIFEST
# =============================================================================

print()
print("=" * 110)
print("9. WRITE V4 PROVENANCE MANIFEST")
print("=" * 110)

manifest = {
    "version":
        "biological_dev_v4",

    "created_by":
        "scripts/116b_build_amddi_common_1087_semantic_bio_v4.py",

    "architecture_contract": {
        "drug_drug_relation":
            "DDI",

        "drug_intrinsic_modalities": [
            "Morgan fingerprint",
            "molecular structure",
        ],

        "drug_protein_relation":
            "DTI",

        "protein_protein_relation":
            "PPI",

        "protein_intrinsic_modality":
            "sequence-derived ESM2",
    },

    "num_drugs":
        NUM_DRUGS,

    "num_active_drugs":
        NUM_ACTIVE_DRUGS,

    "num_proteins":
        NUM_PROTEINS,

    "protein_feature_dim":
        PROTEIN_DIM,

    "protein_namespace":
        "ESM2 protein_index 0..2320",

    "protein_mapping_method":
        "protein_id / UniProt -> ESM2 protein_index",

    "numeric_offset_mapping_used":
        False,

    "missing_esm_proteins":
        sorted(
            EXPECTED_MISSING_PROTEINS
        ),

    "global_missing_esm_dti_relations":
        dict(
            sorted(
                global_missing_counter.items()
            )
        ),

    "common_dti_before_esm_filter":
        len(
            common_before_esm
        ),

    "common_dti_dropped_missing_esm":
        int(
            sum(
                dropped_missing_esm.values()
            )
        ),

    "common_dti_after_esm_filter":
        int(
            master_dti.shape[1]
        ),

    "ppi_source_rows":
        int(
            len(
                ppi_df
            )
        ),

    "ppi_rows_dropped_missing_esm":
        int(
            dropped_ppi_rows
        ),

    "ppi_rows_retained":
        int(
            len(
                kept_ppi_rows
            )
        ),

    "ppi_directed_edges":
        int(
            canonical_ppi.shape[1]
        ),

    "ppi_unique_proteins":
        int(
            len(
                ppi_nodes
            )
        ),

    "protein_features_sha256":
        tensor_hash(
            esm_x
        ),

    "master_dti_sha256":
        tensor_hash(
            master_dti
        ),

    "ppi_sha256":
        tensor_hash(
            canonical_ppi
        ),

    "esm_mapping_file_sha256":
        file_hash(
            ESM_MAP
        ),

    "ppi_semantic_file_sha256":
        file_hash(
            PPI_SEMANTIC
        ),

    "source_biological_v2_preserved":
        True,

    "failed_biological_v3_preserved":
        True,

    "old_dti_derived_drug_bio128_used":
        False,

    "outer_test_used":
        False,

    "settings":
        setting_records,
}

atomic_json_save(
    manifest,
    OUT_ROOT
    / "manifest.json",
)

passed(
    "v4 provenance manifest written"
)


# =============================================================================
# 10. READ-BACK
# =============================================================================

print()
print("=" * 110)
print("10. READ-BACK VERIFICATION")
print("=" * 110)

check(
    torch.equal(
        load_torch(
            OUT_ROOT
            / "protein_features_esm2_128.pt"
        ),
        esm_x,
    ),
    "saved protein features read back exactly",
)

check(
    torch.equal(
        normalize_edge_index(
            load_torch(
                OUT_ROOT
                / "master_dti_edge_index.pt"
            ),
            "readback DTI",
        ),
        master_dti,
    ),
    "saved master DTI reads back exactly",
)

check(
    torch.equal(
        normalize_edge_index(
            load_torch(
                OUT_ROOT
                / "ppi_edge_index.pt"
            ),
            "readback PPI",
        ),
        canonical_ppi,
    ),
    "saved PPI reads back exactly",
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 110)
print("SCRIPT 116b SUMMARY")
print("=" * 110)

print(
    "PASS checks :",
    PASS,
)

print(
    "Failures    :",
    FAIL,
)

print()
print(
    "Canonical ESM proteins     :",
    NUM_PROTEINS,
)

print(
    "Master semantic DTI edges  :",
    master_dti.shape[1],
)

print(
    "Canonical PPI edges        :",
    canonical_ppi.shape[1],
)

print(
    "Canonical PPI proteins     :",
    len(
        ppi_nodes
    ),
)

print()

for setting in SETTINGS:
    print(
        setting,
        "DTI:",
        setting_records[
            setting
        ][
            "development_dti_edges"
        ],
        "DDI:",
        setting_records[
            setting
        ][
            "ddi_directed_edges"
        ],
    )

print()

if FAIL == 0:

    print(
        "SEMANTIC BIOLOGICAL DEVELOPMENT V4 BUILD: PASSED"
    )

    print(
        "DTI: identifier-mapped into ESM2 namespace."
    )

    print(
        "PPI: UniProt-mapped into ESM2 namespace."
    )

    print(
        "PROTEIN FEATURES: intrinsic sequence-derived ESM2."
    )

    print(
        "INNER-VALIDATION DTI EXPOSURE: ZERO."
    )

    print(
        "OUTER-HELDOUT DTI EXPOSURE: ZERO."
    )

    print(
        "OLD DTI-DERIVED DRUG BIO128: NOT USED."
    )

    print(
        "OUTER TEST: STILL SEALED."
    )

    print()
    print(
        "V4 IS BUILT BUT NOT YET FROZEN."
    )

    print(
        "SAFE NEXT STEP: Script 117 independent read-only audit."
    )

    sys.exit(0)

else:

    print(
        "SEMANTIC BIOLOGICAL DEVELOPMENT V4 BUILD: FAILED"
    )

    print(
        "DO NOT TRAIN."
    )

    sys.exit(1)
