from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

import pandas as pd
import torch


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

# Certified inputs - NEVER MODIFY
SPLIT_ROOT = ROOT / "data/amddi_common_1087_coldstart_dev_splits_v2"
BIO_V2_ROOT = ROOT / "data/amddi_common_1087_biological_dev_v2"

COMMON_DRUG_MAP = ROOT / "data/amddi_common/drug_mapping.csv"

ORIGINAL_DTI_EDGE = ROOT / "graphs/dti_graph/edge_index.pt"
ORIGINAL_DTI_DRUG_MAP = ROOT / "graphs/dti_graph/drug_node_mapping.csv"
ORIGINAL_DTI_PROTEIN_MAP = ROOT / "graphs/dti_graph/protein_node_mapping.csv"

ESM_EMBED = ROOT / "features/protein_sequence/esm2_protein_embeddings.pt"
ESM_MAP = ROOT / "features/protein_sequence/esm2_protein_mapping.csv"

PPI_UNIPROT = ROOT / "graphs/dpddi/ppi_uniprot_edges.csv"

# NEW corrected semantic biological layer
OUT_ROOT = ROOT / "data/amddi_common_1087_biological_dev_v3"

SETTINGS = [
    "cold_start_S1",
    "cold_start_S2",
]

EXPECTED_NUM_DRUGS = 1087
EXPECTED_ACTIVE_DRUGS = 1071
EXPECTED_PROTEINS = 2321
EXPECTED_PROTEIN_DIM = 128
EXPECTED_PPI_UNDIRECTED = 53972


# =============================================================================
# HELPERS
# =============================================================================

PASS = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"PASS: {msg}")


def bad(msg):
    global FAIL
    FAIL += 1
    print(f"FAIL: {msg}")


def check(cond, msg):
    if cond:
        ok(msg)
        return True
    bad(msg)
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
        raise TypeError(f"{name}: expected tensor")

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
        f"{name}: expected [2,E] or [E,2], got {tuple(x.shape)}"
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


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(obj, tmp)
    os.replace(tmp, path)


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

    os.replace(tmp, path)


def tensor_set(x):
    return set(
        int(v)
        for v in x.detach().cpu().view(-1).tolist()
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 110)
print("SCRIPT 116")
print("AM-DDI COMMON-1087 — CANONICAL SEMANTIC PROTEIN NAMESPACE REBUILD")
print("=" * 110)

print()
print("IMPORTANT:")
print("  Source certified v2 artifacts are READ ONLY.")
print("  New output is written only to:")
print(f"  {OUT_ROOT}")
print()


# =============================================================================
# REQUIRED INPUTS
# =============================================================================

required = [
    SPLIT_ROOT,
    BIO_V2_ROOT,
    COMMON_DRUG_MAP,
    ORIGINAL_DTI_EDGE,
    ORIGINAL_DTI_DRUG_MAP,
    ORIGINAL_DTI_PROTEIN_MAP,
    ESM_EMBED,
    ESM_MAP,
    PPI_UNIPROT,
]

missing = [
    str(p)
    for p in required
    if not p.exists()
]

if missing:
    print("Missing required inputs:")

    for p in missing:
        print("  ", p)

    sys.exit(1)

ok("all required source artifacts exist")


# =============================================================================
# LOAD ESM2 — AUTHORITATIVE PROTEIN NAMESPACE
# =============================================================================

print()
print("=" * 110)
print("1. AUTHORITATIVE ESM2 PROTEIN NAMESPACE")
print("=" * 110)

esm_map = pd.read_csv(
    ESM_MAP
)

esm_x = load_torch(
    ESM_EMBED
)

check(
    list(esm_map.columns)
    == ["protein_index", "protein_id"],
    "ESM mapping columns correct",
)

check(
    len(esm_map)
    == EXPECTED_PROTEINS,
    "ESM mapping contains 2321 proteins",
)

check(
    esm_map["protein_index"].tolist()
    == list(range(EXPECTED_PROTEINS)),
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
    not esm_map.isna().any().any(),
    "ESM mapping has no missing values",
)

check(
    tuple(esm_x.shape)
    == (
        EXPECTED_PROTEINS,
        EXPECTED_PROTEIN_DIM,
    ),
    "ESM tensor shape exactly (2321,128)",
)

check(
    torch.isfinite(esm_x).all().item(),
    "ESM tensor contains only finite values",
)

check(
    torch.count_nonzero(esm_x).item() > 0,
    "ESM protein features are non-zero",
)

protein_to_esm = dict(
    zip(
        esm_map["protein_id"].astype(str),
        esm_map["protein_index"].astype(int),
    )
)

esm_ids = set(
    protein_to_esm.keys()
)


# =============================================================================
# COMMON DRUG NAMESPACE
# =============================================================================

print()
print("=" * 110)
print("2. COMMON-1087 DRUG NAMESPACE")
print("=" * 110)

common = pd.read_csv(
    COMMON_DRUG_MAP
)

check(
    len(common)
    == EXPECTED_NUM_DRUGS,
    "common drug mapping contains exactly 1087 drugs",
)

check(
    common["node_id"].astype(int).tolist()
    == list(range(EXPECTED_NUM_DRUGS)),
    "common drug node IDs exactly 0..1086",
)

check(
    common["drug_id"].astype(str).is_unique,
    "common drug IDs unique",
)

drug_to_common = dict(
    zip(
        common["drug_id"].astype(str),
        common["node_id"].astype(int),
    )
)


# =============================================================================
# ORIGINAL DTI MAPPINGS
# =============================================================================

print()
print("=" * 110)
print("3. ORIGINAL DTI SEMANTIC MAPPINGS")
print("=" * 110)

dti_drug_map = pd.read_csv(
    ORIGINAL_DTI_DRUG_MAP
)

dti_protein_map = pd.read_csv(
    ORIGINAL_DTI_PROTEIN_MAP
)

check(
    len(dti_protein_map) == 2324,
    "original DTI mapping contains 2324 proteins",
)

check(
    dti_protein_map["protein_id"].astype(str).is_unique,
    "original DTI protein IDs unique",
)

dti_protein_ids = set(
    dti_protein_map["protein_id"].astype(str)
)

missing_from_esm = sorted(
    dti_protein_ids - esm_ids
)

extra_in_esm = sorted(
    esm_ids - dti_protein_ids
)

print()
print("DTI proteins absent from ESM:")
print(missing_from_esm)

print()
print("ESM proteins absent from DTI:")
print(extra_in_esm)

check(
    missing_from_esm
    == [
        "P04745",
        "P62158",
        "Q13748",
    ],
    "exactly three known DTI proteins are absent from ESM",
)

check(
    len(extra_in_esm) == 0,
    "all 2321 ESM proteins exist in original DTI mapping",
)

old_drug_node_to_id = dict(
    zip(
        dti_drug_map["node_id"].astype(int),
        dti_drug_map["drug_id"].astype(str),
    )
)

old_protein_node_to_id = dict(
    zip(
        dti_protein_map["node_id"].astype(int),
        dti_protein_map["protein_id"].astype(str),
    )
)


# =============================================================================
# REBUILD MASTER COMMON DTI SEMANTICALLY
# =============================================================================

print()
print("=" * 110)
print("4. REBUILD MASTER COMMON DTI BY DRUG_ID + PROTEIN_ID")
print("=" * 110)

raw_dti = normalize_edge_index(
    load_torch(
        ORIGINAL_DTI_EDGE
    ),
    "original DTI",
)

print(
    "Original DTI tensor:",
    tuple(raw_dti.shape),
)

semantic_dti_pairs = set()

dropped_noncommon_drug = 0
dropped_missing_esm = 0
dropped_unknown = 0

missing_esm_edge_counts = {}

for old_d, old_p in raw_dti.t().tolist():

    old_d = int(old_d)
    old_p = int(old_p)

    if old_d not in old_drug_node_to_id:
        dropped_unknown += 1
        continue

    if old_p not in old_protein_node_to_id:
        dropped_unknown += 1
        continue

    drug_id = old_drug_node_to_id[
        old_d
    ]

    protein_id = old_protein_node_to_id[
        old_p
    ]

    if drug_id not in drug_to_common:
        dropped_noncommon_drug += 1
        continue

    if protein_id not in protein_to_esm:
        dropped_missing_esm += 1

        missing_esm_edge_counts[
            protein_id
        ] = (
            missing_esm_edge_counts.get(
                protein_id,
                0,
            )
            + 1
        )

        continue

    semantic_dti_pairs.add(
        (
            int(
                drug_to_common[
                    drug_id
                ]
            ),
            int(
                protein_to_esm[
                    protein_id
                ]
            ),
        )
    )

master_dti_pairs = sorted(
    semantic_dti_pairs
)

master_dti = torch.tensor(
    master_dti_pairs,
    dtype=torch.long,
).t().contiguous()

print()
print(
    "Canonical common DTI edges :",
    master_dti.shape[1],
)

print(
    "Dropped non-common drug    :",
    dropped_noncommon_drug,
)

print(
    "Dropped missing-ESM protein:",
    dropped_missing_esm,
)

print(
    "Missing-ESM edge counts    :",
    missing_esm_edge_counts,
)

print(
    "Dropped unknown mappings   :",
    dropped_unknown,
)

check(
    master_dti.shape[0] == 2,
    "canonical DTI has shape [2,E]",
)

check(
    master_dti.shape[1] > 0,
    "canonical DTI is non-empty",
)

check(
    int(master_dti[0].min()) >= 0
    and int(master_dti[0].max()) < EXPECTED_NUM_DRUGS,
    "canonical DTI drug IDs fit 0..1086",
)

check(
    int(master_dti[1].min()) >= 0
    and int(master_dti[1].max()) < EXPECTED_PROTEINS,
    "canonical DTI protein IDs fit ESM namespace 0..2320",
)

check(
    dropped_unknown == 0,
    "no DTI edges had unknown node mappings",
)


# =============================================================================
# REBUILD PPI SEMANTICALLY
# =============================================================================

print()
print("=" * 110)
print("5. REBUILD PPI BY UNIPROT ID -> ESM INDEX")
print("=" * 110)

ppi_df = pd.read_csv(
    PPI_UNIPROT
)

check(
    list(ppi_df.columns)
    == ["protein1", "protein2"],
    "PPI semantic CSV columns correct",
)

check(
    len(ppi_df)
    == EXPECTED_PPI_UNDIRECTED,
    "PPI semantic CSV contains 53,972 source pairs",
)

ppi_pairs_semantic = set()

missing_ppi_ids = set()

for a, b in zip(
    ppi_df["protein1"].astype(str),
    ppi_df["protein2"].astype(str),
):

    if a not in protein_to_esm:
        missing_ppi_ids.add(a)
        continue

    if b not in protein_to_esm:
        missing_ppi_ids.add(b)
        continue

    ia = int(
        protein_to_esm[a]
    )

    ib = int(
        protein_to_esm[b]
    )

    ppi_pairs_semantic.add(
        (
            ia,
            ib,
        )
    )

check(
    len(missing_ppi_ids) == 0,
    "every PPI UniProt ID exists in canonical ESM namespace",
)

print(
    "Unique semantic source pairs:",
    len(ppi_pairs_semantic),
)

# Build both directions exactly.
ppi_directed = []

for a, b in sorted(
    ppi_pairs_semantic
):

    ppi_directed.append(
        (a, b)
    )

    ppi_directed.append(
        (b, a)
    )

canonical_ppi = torch.tensor(
    ppi_directed,
    dtype=torch.long,
).t().contiguous()

check(
    canonical_ppi.shape[0] == 2,
    "canonical PPI has shape [2,E]",
)

check(
    canonical_ppi.shape[1]
    == 2 * len(ppi_pairs_semantic),
    "canonical PPI contains forward+reverse edge for each source pair",
)

check(
    int(canonical_ppi.min()) >= 0
    and int(canonical_ppi.max()) < EXPECTED_PROTEINS,
    "canonical PPI IDs fit ESM namespace 0..2320",
)

ppi_nodes = tensor_set(
    canonical_ppi
)

print(
    "Canonical PPI directed edges:",
    canonical_ppi.shape[1],
)

print(
    "Canonical PPI unique proteins:",
    len(ppi_nodes),
)


# =============================================================================
# PREPARE NEW VERSIONED OUTPUT
# =============================================================================

print()
print("=" * 110)
print("6. CREATE VERSIONED BIOLOGICAL V3 LAYER")
print("=" * 110)

if OUT_ROOT.exists():

    # Never silently overwrite an existing corrected layer.
    existing_manifest = (
        OUT_ROOT
        / "manifest.json"
    )

    if existing_manifest.exists():
        print(
            "ERROR: biological_dev_v3 already exists."
        )
        print(
            "Refusing to overwrite an existing versioned artifact."
        )
        print(
            "Inspect it first or explicitly remove it only if this run "
            "was never completed."
        )
        sys.exit(1)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

ok(
    "created new biological_dev_v3 directory without modifying v2"
)


# =============================================================================
# INSTALL CANONICAL PROTEIN ARTIFACTS
# =============================================================================

canonical_esm_path = (
    OUT_ROOT
    / "protein_features_esm2_128.pt"
)

canonical_esm_map_path = (
    OUT_ROOT
    / "protein_mapping_esm2.csv"
)

canonical_dti_path = (
    OUT_ROOT
    / "master_dti_edge_index.pt"
)

canonical_ppi_path = (
    OUT_ROOT
    / "ppi_edge_index.pt"
)

atomic_torch_save(
    esm_x.detach().cpu().float(),
    canonical_esm_path,
)

esm_map.to_csv(
    canonical_esm_map_path,
    index=False,
)

atomic_torch_save(
    master_dti,
    canonical_dti_path,
)

atomic_torch_save(
    canonical_ppi,
    canonical_ppi_path,
)

ok(
    "installed canonical ESM2 protein features"
)

ok(
    "installed canonical semantic master DTI"
)

ok(
    "installed canonical semantic PPI"
)


# =============================================================================
# SETTING-SPECIFIC DEVELOPMENT GRAPHS
# =============================================================================

print()
print("=" * 110)
print("7. REBUILD STRICT COLD-START SETTING-SPECIFIC DTI")
print("=" * 110)

setting_manifest = {}

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

    new_dir = (
        OUT_ROOT
        / setting
    )

    new_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -------------------------------------------------------------------------
    # COPY CERTIFIED PARTITIONS
    # -------------------------------------------------------------------------

    partition_names = [
        "inner_train_drugs.pt",
        "inner_val_drugs.pt",
        "outer_train_drugs.pt",
        "outer_heldout_drugs.pt",
    ]

    partitions = {}

    for filename in partition_names:

        src_candidates = [
            old_bio_dir / filename,
            split_dir / filename,
        ]

        src = None

        for candidate in src_candidates:
            if candidate.exists():
                src = candidate
                break

        if src is None:
            raise FileNotFoundError(
                f"{setting}: missing {filename}"
            )

        x = load_torch(
            src
        ).long().view(-1)

        atomic_torch_save(
            x,
            new_dir / filename,
        )

        partitions[
            filename
        ] = x

    inner_train = tensor_set(
        partitions[
            "inner_train_drugs.pt"
        ]
    )

    inner_val = tensor_set(
        partitions[
            "inner_val_drugs.pt"
        ]
    )

    outer_train = tensor_set(
        partitions[
            "outer_train_drugs.pt"
        ]
    )

    outer_heldout = tensor_set(
        partitions[
            "outer_heldout_drugs.pt"
        ]
    )

    check(
        len(inner_train) == 685,
        f"{setting}: inner train = 685",
    )

    check(
        len(inner_val) == 171,
        f"{setting}: inner val = 171",
    )

    check(
        len(outer_train) == 856,
        f"{setting}: outer train = 856",
    )

    check(
        len(outer_heldout) == 215,
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
        == EXPECTED_ACTIVE_DRUGS,
        f"{setting}: outer partition reconstructs 1071 active drugs",
    )

    # -------------------------------------------------------------------------
    # FILTER DTI TO INNER TRAIN DRUGS ONLY
    # -------------------------------------------------------------------------

    mask = torch.tensor(
        [
            int(d)
            in inner_train
            for d
            in master_dti[0].tolist()
        ],
        dtype=torch.bool,
    )

    setting_dti = (
        master_dti[
            :,
            mask,
        ]
        .contiguous()
    )

    setting_dti_drugs = tensor_set(
        setting_dti[0]
    )

    check(
        setting_dti_drugs.issubset(
            inner_train
        ),
        f"{setting}: all DTI drug endpoints are inner-train drugs",
    )

    check(
        len(
            setting_dti_drugs
            & inner_val
        )
        == 0,
        f"{setting}: inner-validation DTI exposure = 0",
    )

    check(
        len(
            setting_dti_drugs
            & outer_heldout
        )
        == 0,
        f"{setting}: outer-heldout DTI exposure = 0",
    )

    check(
        int(setting_dti[1].max())
        < EXPECTED_PROTEINS,
        f"{setting}: DTI protein IDs remain canonical ESM indices",
    )

    atomic_torch_save(
        setting_dti,
        new_dir
        / "dti_edge_index.pt",
    )

    # -------------------------------------------------------------------------
    # COPY CERTIFIED TRAIN-ONLY DDI GRAPH
    # -------------------------------------------------------------------------

    ddi_candidates = [
        old_bio_dir
        / "ddi_train_edge_index.pt",

        old_bio_dir
        / "ddi_edge_index.pt",

        old_bio_dir
        / "train_ddi_edge_index.pt",
    ]

    ddi_src = None

    for candidate in ddi_candidates:
        if candidate.exists():
            ddi_src = candidate
            break

    if ddi_src is None:
        raise FileNotFoundError(
            f"{setting}: certified DDI training graph not found"
        )

    ddi = normalize_edge_index(
        load_torch(
            ddi_src
        ),
        f"{setting} certified DDI",
    )

    ddi_nodes = tensor_set(
        ddi
    )

    check(
        ddi_nodes.issubset(
            inner_train
        ),
        f"{setting}: copied DDI graph contains only inner-train drugs",
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
        new_dir
        / "ddi_train_edge_index.pt",
    )

    # -------------------------------------------------------------------------
    # COPY ZERO DRUG BIO INITIALIZATION
    # -------------------------------------------------------------------------

    zero_candidates = [
        old_bio_dir
        / "drug_bio_init_zero_128.pt",

        BIO_V2_ROOT
        / "drug_bio_init_zero_128.pt",
    ]

    zero_src = None

    for candidate in zero_candidates:
        if candidate.exists():
            zero_src = candidate
            break

    if zero_src is None:
        raise FileNotFoundError(
            f"{setting}: zero drug bio initialization missing"
        )

    zero_x = load_torch(
        zero_src
    )

    check(
        tuple(zero_x.shape)
        == (
            EXPECTED_NUM_DRUGS,
            128,
        ),
        f"{setting}: zero drug bio feature shape = (1087,128)",
    )

    check(
        torch.count_nonzero(
            zero_x
        ).item()
        == 0,
        f"{setting}: zero drug bio features remain exactly zero",
    )

    atomic_torch_save(
        zero_x,
        new_dir
        / "drug_bio_init_zero_128.pt",
    )

    # -------------------------------------------------------------------------
    # COPY SPLIT PAIR TENSORS UNCHANGED
    # -------------------------------------------------------------------------

    pair_names = [
        "train_pos.pt",
        "train_neg.pt",
        "val_pos.pt",
        "val_neg.pt",
        "test_pos.pt",
        "test_neg.pt",
    ]

    pair_hashes = {}

    for filename in pair_names:

        src = (
            split_dir
            / filename
        )

        if not src.exists():
            raise FileNotFoundError(
                src
            )

        x = load_torch(
            src
        )

        atomic_torch_save(
            x,
            new_dir
            / filename,
        )

        pair_hashes[
            filename
        ] = tensor_hash(
            x
        )

    # -------------------------------------------------------------------------
    # SETTING MANIFEST
    # -------------------------------------------------------------------------

    setting_info = {
        "setting":
            setting,

        "protein_namespace":
            "esm2_protein_index_0_2320",

        "protein_num_nodes":
            EXPECTED_PROTEINS,

        "protein_feature_dim":
            EXPECTED_PROTEIN_DIM,

        "inner_train_drugs":
            len(inner_train),

        "inner_val_drugs":
            len(inner_val),

        "outer_train_drugs":
            len(outer_train),

        "outer_heldout_drugs":
            len(outer_heldout),

        "semantic_master_dti_edges":
            int(
                master_dti.shape[1]
            ),

        "filtered_development_dti_edges":
            int(
                setting_dti.shape[1]
            ),

        "ddi_directed_edges":
            int(
                ddi.shape[1]
            ),

        "inner_val_dti_exposure":
            0,

        "outer_heldout_dti_exposure":
            0,

        "old_dti_derived_drug_features_used":
            False,

        "protein_features":
            "ESM2 sequence-derived 128-D",

        "outer_test_used":
            False,

        "pair_tensor_hashes":
            pair_hashes,
    }

    atomic_json_save(
        setting_info,
        new_dir
        / "manifest.json",
    )

    setting_manifest[
        setting
    ] = setting_info

    print(
        f"{setting} canonical DTI edges:",
        setting_dti.shape[1],
    )


# =============================================================================
# ROOT MANIFEST
# =============================================================================

print()
print("=" * 110)
print("8. WRITE PROVENANCE MANIFEST")
print("=" * 110)

root_manifest = {
    "version":
        "biological_dev_v3",

    "created_by":
        "scripts/116_build_amddi_common_1087_semantic_protein_namespace.py",

    "purpose":
        (
            "Canonical semantic protein namespace correction. "
            "All DTI and PPI protein endpoints are mapped through "
            "UniProt/protein_id into ESM2 protein_index 0..2320."
        ),

    "source_split_protocol":
        "strict_coldstart_dev_v2",

    "num_drugs":
        EXPECTED_NUM_DRUGS,

    "num_active_drugs":
        EXPECTED_ACTIVE_DRUGS,

    "protein_namespace":
        "ESM2 protein_index",

    "protein_num_nodes":
        EXPECTED_PROTEINS,

    "protein_feature_dim":
        EXPECTED_PROTEIN_DIM,

    "protein_features_source":
        str(
            ESM_EMBED.relative_to(
                ROOT
            )
        ),

    "protein_mapping_source":
        str(
            ESM_MAP.relative_to(
                ROOT
            )
        ),

    "protein_features_tensor_sha256":
        tensor_hash(
            esm_x
        ),

    "protein_mapping_file_sha256":
        file_hash(
            ESM_MAP
        ),

    "ppi_semantic_source":
        str(
            PPI_UNIPROT.relative_to(
                ROOT
            )
        ),

    "ppi_source_pairs":
        int(
            len(ppi_df)
        ),

    "ppi_unique_semantic_pairs":
        int(
            len(ppi_pairs_semantic)
        ),

    "ppi_directed_edges":
        int(
            canonical_ppi.shape[1]
        ),

    "ppi_unique_proteins":
        int(
            len(ppi_nodes)
        ),

    "ppi_tensor_sha256":
        tensor_hash(
            canonical_ppi
        ),

    "original_dti_protein_count":
        int(
            len(dti_protein_map)
        ),

    "canonical_esm_protein_count":
        EXPECTED_PROTEINS,

    "dti_proteins_excluded_for_missing_esm":
        missing_from_esm,

    "dti_edges_dropped_missing_esm":
        int(
            dropped_missing_esm
        ),

    "dti_missing_esm_edge_counts":
        missing_esm_edge_counts,

    "master_semantic_dti_edges":
        int(
            master_dti.shape[1]
        ),

    "master_semantic_dti_sha256":
        tensor_hash(
            master_dti
        ),

    "semantic_alignment_policy": {
        "numeric_offset_mapping_used":
            False,

        "dti_protein_mapping":
            (
                "old DTI node_id -> protein_id "
                "-> ESM2 protein_index"
            ),

        "ppi_protein_mapping":
            (
                "PPI UniProt ID -> "
                "ESM2 protein_index"
            ),

        "protein_features":
            (
                "sequence-derived ESM2 only"
            ),

        "old_dti_derived_drug_bio128_used":
            False,

        "outer_test_used":
            False,
    },

    "settings":
        setting_manifest,

    "source_certified_biological_v2_preserved":
        True,
}

atomic_json_save(
    root_manifest,
    OUT_ROOT
    / "manifest.json",
)

ok(
    "root semantic provenance manifest written"
)


# =============================================================================
# FINAL READ-BACK
# =============================================================================

print()
print("=" * 110)
print("9. READ-BACK VERIFICATION")
print("=" * 110)

esm_check = load_torch(
    canonical_esm_path
)

dti_check = normalize_edge_index(
    load_torch(
        canonical_dti_path
    ),
    "read-back master DTI",
)

ppi_check = normalize_edge_index(
    load_torch(
        canonical_ppi_path
    ),
    "read-back PPI",
)

check(
    torch.equal(
        esm_check,
        esm_x.float(),
    ),
    "saved ESM tensor reads back exactly",
)

check(
    torch.equal(
        dti_check,
        master_dti,
    ),
    "saved semantic master DTI reads back exactly",
)

check(
    torch.equal(
        ppi_check,
        canonical_ppi,
    ),
    "saved semantic PPI reads back exactly",
)


# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 110)
print("SCRIPT 116 SUMMARY")
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
    "Canonical proteins          :",
    EXPECTED_PROTEINS,
)

print(
    "Protein features            :",
    tuple(esm_x.shape),
)

print(
    "Canonical master DTI edges  :",
    master_dti.shape[1],
)

print(
    "Canonical PPI directed edges:",
    canonical_ppi.shape[1],
)

print(
    "Canonical PPI proteins      :",
    len(ppi_nodes),
)

print(
    "Dropped DTI edges because protein lacks ESM:",
    dropped_missing_esm,
)

print(
    "Excluded DTI proteins:",
    missing_from_esm,
)

print()

for setting in SETTINGS:
    print(
        setting,
        "development DTI edges:",
        setting_manifest[
            setting
        ][
            "filtered_development_dti_edges"
        ],
    )

print()

if FAIL == 0:

    print(
        "SEMANTIC PROTEIN NAMESPACE REBUILD: PASSED"
    )

    print(
        "DTI PROTEIN ENDPOINTS: UniProt -> ESM2 index."
    )

    print(
        "PPI PROTEIN ENDPOINTS: UniProt -> ESM2 index."
    )

    print(
        "PROTEIN FEATURES: sequence-derived ESM2."
    )

    print(
        "OLD NUMERIC PROTEIN OFFSET ASSUMPTIONS: NOT USED."
    )

    print(
        "OUTER TEST: NOT USED."
    )

    print()
    print(
        "IMPORTANT: V3 IS NOT YET FROZEN."
    )

    print(
        "SAFE NEXT STEP: independent Script 117 audit."
    )

    sys.exit(0)

else:

    print(
        "SEMANTIC PROTEIN NAMESPACE REBUILD: FAILED"
    )

    print(
        "DO NOT TRAIN."
    )

    sys.exit(1)
