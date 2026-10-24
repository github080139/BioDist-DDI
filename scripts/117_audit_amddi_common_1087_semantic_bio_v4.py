from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

import pandas as pd
import torch


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

V4_ROOT = ROOT / "data/amddi_common_1087_biological_dev_v4"
SPLIT_ROOT = ROOT / "data/amddi_common_1087_coldstart_dev_splits_v2"

COMMON_DRUG_MAP = ROOT / "data/amddi_common/drug_mapping.csv"

RAW_DTI_EDGE = ROOT / "graphs/dti_graph/edge_index.pt"
DTI_DRUG_MAP = ROOT / "graphs/dti_graph/drug_node_mapping.csv"
DTI_PROTEIN_MAP = ROOT / "graphs/dti_graph/protein_node_mapping.csv"

ESM_MAP = ROOT / "features/protein_sequence/esm2_protein_mapping.csv"
ESM_X = ROOT / "features/protein_sequence/esm2_protein_embeddings.pt"

PPI_SEMANTIC = ROOT / "graphs/dpddi/ppi_uniprot_edges.csv"

SETTINGS = [
    "cold_start_S1",
    "cold_start_S2",
]


# =============================================================================
# EXPECTED VALUES
# =============================================================================

NUM_DRUGS = 1087
NUM_ACTIVE = 1071
NUM_PROTEINS = 2321
PROTEIN_DIM = 128

EXPECTED_MASTER_DTI = 7392
EXPECTED_PPI_DIRECTED = 107160
EXPECTED_PPI_PROTEINS = 2263

EXPECTED_MISSING = {
    "P04745",
    "P62158",
    "Q13748",
}

EXPECTED_SETTING = {
    "cold_start_S1": {
        "dti": 4805,
        "ddi": 28988,
        "train_pos": 14494,
        "train_neg": 14494,
        "val_pos": 7379,
        "val_neg": 7379,
        "test_pos": 11713,
        "test_neg": 11713,
        "val_semantics": {0: 0, 1: 7379, 2: 0},
        "test_semantics": {0: 0, 1: 11713, 2: 0},
    },
    "cold_start_S2": {
        "dti": 4695,
        "ddi": 33806,
        "train_pos": 16903,
        "train_neg": 16903,
        "val_pos": 686,
        "val_neg": 686,
        "test_pos": 1086,
        "test_neg": 1086,
        "val_semantics": {0: 0, 1: 0, 2: 686},
        "test_semantics": {0: 0, 1: 0, 2: 1086},
    },
}


# =============================================================================
# AUDIT HELPERS
# =============================================================================

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


def check(cond, msg):
    if cond:
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
        f"{name}: expected [2,E] or [E,2]"
    )


def tensor_hash(x):
    x = x.detach().cpu().contiguous()

    return hashlib.sha256(
        x.numpy().tobytes()
    ).hexdigest()


def tensor_set(x):
    return set(
        int(v)
        for v in x.detach().cpu().view(-1).tolist()
    )


def pair_set(x):
    x = x.detach().cpu().long()

    if x.shape[0] == 2 and x.shape[1] != 2:
        x = x.t().contiguous()

    out = set()

    for a, b in x.tolist():
        a = int(a)
        b = int(b)

        if a <= b:
            out.add((a, b))
        else:
            out.add((b, a))

    return out


def semantics(pairs, unseen):
    pairs = pairs.detach().cpu().long()

    if pairs.shape[0] == 2 and pairs.shape[1] != 2:
        pairs = pairs.t().contiguous()

    unseen = set(
        int(x)
        for x in unseen
    )

    result = {
        0: 0,
        1: 0,
        2: 0,
    }

    for a, b in pairs.tolist():
        n = (
            int(int(a) in unseen)
            + int(int(b) in unseen)
        )

        result[n] += 1

    return result


# =============================================================================
# HEADER
# =============================================================================

print("=" * 110)
print("SCRIPT 117")
print("FINAL INDEPENDENT READ-ONLY AUDIT")
print("AM-DDI COMMON-1087 SEMANTIC BIOLOGICAL DEVELOPMENT V4")
print("=" * 110)

print()
print("READ ONLY.")
print("No artifact will be modified.")
print()


# =============================================================================
# REQUIRED PATHS
# =============================================================================

required = [
    V4_ROOT,
    V4_ROOT / "manifest.json",
    V4_ROOT / "protein_features_esm2_128.pt",
    V4_ROOT / "protein_mapping_esm2.csv",
    V4_ROOT / "master_dti_edge_index.pt",
    V4_ROOT / "ppi_edge_index.pt",
    COMMON_DRUG_MAP,
    RAW_DTI_EDGE,
    DTI_DRUG_MAP,
    DTI_PROTEIN_MAP,
    ESM_MAP,
    ESM_X,
    PPI_SEMANTIC,
]

missing = [
    p
    for p in required
    if not p.exists()
]

check(
    len(missing) == 0,
    "all required v4 and source artifacts exist",
)

if missing:
    for p in missing:
        print("MISSING:", p)
    sys.exit(1)


# =============================================================================
# 1. ESM2 PROTEIN NAMESPACE
# =============================================================================

print()
print("=" * 110)
print("1. ESM2 PROTEIN FEATURE CERTIFICATION")
print("=" * 110)

esm_map = pd.read_csv(
    ESM_MAP
)

v4_map = pd.read_csv(
    V4_ROOT / "protein_mapping_esm2.csv"
)

esm_x = load_torch(
    ESM_X
).detach().cpu().float()

v4_x = load_torch(
    V4_ROOT / "protein_features_esm2_128.pt"
).detach().cpu().float()

check(
    len(esm_map) == NUM_PROTEINS,
    "source ESM mapping contains 2321 proteins",
)

check(
    esm_map["protein_index"].astype(int).tolist()
    == list(range(NUM_PROTEINS)),
    "source ESM index exactly 0..2320",
)

check(
    esm_map["protein_id"].astype(str).is_unique,
    "source ESM protein IDs unique",
)

check(
    tuple(esm_x.shape)
    == (NUM_PROTEINS, PROTEIN_DIM),
    "source ESM tensor shape = (2321,128)",
)

check(
    bool(torch.isfinite(esm_x).all()),
    "source ESM tensor finite",
)

check(
    v4_map.equals(esm_map),
    "v4 protein mapping exactly equals source ESM mapping",
)

check(
    torch.equal(v4_x, esm_x),
    "v4 ESM features exactly equal source sequence-derived features",
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
# 2. INDEPENDENT DTI RECONSTRUCTION
# =============================================================================

print()
print("=" * 110)
print("2. INDEPENDENT SEMANTIC DTI RECONSTRUCTION")
print("=" * 110)

common = pd.read_csv(
    COMMON_DRUG_MAP
)

drug_map = pd.read_csv(
    DTI_DRUG_MAP
)

protein_map = pd.read_csv(
    DTI_PROTEIN_MAP
)

raw = normalize_edge_index(
    load_torch(RAW_DTI_EDGE),
    "raw DTI",
)

drug_to_common = dict(
    zip(
        common["drug_id"].astype(str),
        common["node_id"].astype(int),
    )
)

old_drug_to_id = dict(
    zip(
        drug_map["node_id"].astype(int),
        drug_map["drug_id"].astype(str),
    )
)

old_protein_to_id = dict(
    zip(
        protein_map["node_id"].astype(int),
        protein_map["protein_id"].astype(str),
    )
)

drug_nodes = set(old_drug_to_id)
protein_nodes = set(old_protein_to_id)

forward = set()
reverse = set()
invalid = []

for a, b in raw.t().tolist():

    a = int(a)
    b = int(b)

    if a in drug_nodes and b in protein_nodes:
        forward.add((a, b))

    elif a in protein_nodes and b in drug_nodes:
        reverse.add((b, a))

    else:
        invalid.append((a, b))

check(
    len(invalid) == 0,
    "raw DTI orientation has zero invalid edges",
)

check(
    len(forward) == 15138,
    "raw DTI has 15138 forward relations",
)

check(
    len(reverse) == 15138,
    "raw DTI has 15138 reverse relations",
)

check(
    forward == reverse,
    "forward and reverse raw DTI relations are identical",
)

protein_ids = set(
    protein_map["protein_id"].astype(str)
)

check(
    protein_ids - esm_ids
    == EXPECTED_MISSING,
    "three DTI proteins absent from ESM exactly match expected IDs",
)

independent_dti = []

missing_common = Counter()

for old_d, old_p in sorted(forward):

    drug_id = old_drug_to_id[old_d]
    protein_id = old_protein_to_id[old_p]

    if drug_id not in drug_to_common:
        continue

    if protein_id not in protein_to_esm:
        missing_common[protein_id] += 1
        continue

    independent_dti.append(
        (
            int(drug_to_common[drug_id]),
            int(protein_to_esm[protein_id]),
        )
    )

check(
    sum(missing_common.values()) == 16,
    "independent reconstruction removes exactly 16 common DTI relations",
)

check(
    dict(missing_common)
    == {"P62158": 16},
    "all 16 removed common DTI relations involve P62158",
)

check(
    len(independent_dti)
    == EXPECTED_MASTER_DTI,
    "independent semantic master DTI contains 7392 edges",
)

check(
    len(set(independent_dti))
    == EXPECTED_MASTER_DTI,
    "independent semantic master DTI has no duplicates",
)

independent_dti_tensor = torch.tensor(
    sorted(independent_dti),
    dtype=torch.long,
).t().contiguous()

saved_master_dti = normalize_edge_index(
    load_torch(
        V4_ROOT / "master_dti_edge_index.pt"
    ),
    "saved v4 master DTI",
)

check(
    torch.equal(
        saved_master_dti,
        independent_dti_tensor,
    ),
    "saved v4 master DTI exactly equals independent reconstruction",
)


# =============================================================================
# 3. INDEPENDENT PPI RECONSTRUCTION
# =============================================================================

print()
print("=" * 110)
print("3. INDEPENDENT SEMANTIC PPI RECONSTRUCTION")
print("=" * 110)

ppi = pd.read_csv(
    PPI_SEMANTIC
)

source_ids = (
    set(ppi["protein1"].astype(str))
    |
    set(ppi["protein2"].astype(str))
)

check(
    source_ids - esm_ids
    == EXPECTED_MISSING,
    "PPI proteins absent from ESM are exactly the same three IDs",
)

kept = []
dropped = 0
missing_counter = Counter()

for a, b in zip(
    ppi["protein1"].astype(str),
    ppi["protein2"].astype(str),
):

    miss = False

    if a not in protein_to_esm:
        missing_counter[a] += 1
        miss = True

    if b not in protein_to_esm:
        missing_counter[b] += 1
        miss = True

    if miss:
        dropped += 1
        continue

    kept.append(
        (
            int(protein_to_esm[a]),
            int(protein_to_esm[b]),
        )
    )

check(
    dropped == 392,
    "independent PPI reconstruction drops exactly 392 rows",
)

check(
    len(kept) == 53580,
    "independent PPI reconstruction retains exactly 53580 source rows",
)

check(
    dict(sorted(missing_counter.items()))
    == {
        "P04745": 39,
        "P62158": 305,
        "Q13748": 50,
    },
    "PPI missing-protein occurrence counts match independent diagnostic",
)

directed = []

for a, b in kept:
    directed.append((a, b))
    directed.append((b, a))

independent_ppi = torch.tensor(
    directed,
    dtype=torch.long,
).t().contiguous()

saved_ppi = normalize_edge_index(
    load_torch(
        V4_ROOT / "ppi_edge_index.pt"
    ),
    "saved v4 PPI",
)

check(
    independent_ppi.shape[1]
    == EXPECTED_PPI_DIRECTED,
    "independent PPI contains exactly 107160 directed edges",
)

check(
    torch.equal(
        independent_ppi,
        saved_ppi,
    ),
    "saved v4 PPI exactly equals independent semantic reconstruction",
)

check(
    len(tensor_set(saved_ppi))
    == EXPECTED_PPI_PROTEINS,
    "saved v4 PPI contains exactly 2263 protein nodes",
)

check(
    int(saved_ppi.min()) >= 0
    and int(saved_ppi.max()) < NUM_PROTEINS,
    "all v4 PPI node IDs fit ESM namespace 0..2320",
)


# =============================================================================
# 4. MANIFEST CROSS-CHECK
# =============================================================================

print()
print("=" * 110)
print("4. V4 MANIFEST CROSS-CHECK")
print("=" * 110)

with open(
    V4_ROOT / "manifest.json",
    "r",
) as f:
    manifest = json.load(f)

check(
    manifest.get("version")
    == "biological_dev_v4",
    "manifest version = biological_dev_v4",
)

check(
    manifest.get("protein_namespace")
    == "ESM2 protein_index 0..2320",
    "manifest records canonical ESM namespace",
)

check(
    manifest.get("numeric_offset_mapping_used")
    is False,
    "manifest confirms numeric offset mapping not used",
)

check(
    int(manifest.get("common_dti_after_esm_filter", -1))
    == EXPECTED_MASTER_DTI,
    "manifest records master DTI = 7392",
)

check(
    int(manifest.get("ppi_directed_edges", -1))
    == EXPECTED_PPI_DIRECTED,
    "manifest records PPI = 107160 directed edges",
)

check(
    int(manifest.get("ppi_unique_proteins", -1))
    == EXPECTED_PPI_PROTEINS,
    "manifest records PPI proteins = 2263",
)

check(
    manifest.get("old_dti_derived_drug_bio128_used")
    is False,
    "manifest confirms old DTI-derived drug bio128 not used",
)

check(
    manifest.get("outer_test_used")
    is False,
    "manifest confirms outer test not used",
)

check(
    manifest.get("master_dti_sha256")
    == tensor_hash(saved_master_dti),
    "manifest DTI hash matches actual tensor",
)

check(
    manifest.get("ppi_sha256")
    == tensor_hash(saved_ppi),
    "manifest PPI hash matches actual tensor",
)

check(
    manifest.get("protein_features_sha256")
    == tensor_hash(v4_x),
    "manifest protein feature hash matches actual tensor",
)


# =============================================================================
# 5. SETTING-SPECIFIC AUDIT
# =============================================================================

print()
print("=" * 110)
print("5. STRICT COLD-START SETTING-SPECIFIC AUDIT")
print("=" * 110)

partitions = {}

for setting in SETTINGS:

    print()
    print("-" * 110)
    print(setting)
    print("-" * 110)

    exp = EXPECTED_SETTING[
        setting
    ]

    setting_dir = (
        V4_ROOT
        / setting
    )

    split_dir = (
        SPLIT_ROOT
        / setting
    )

    inner_train = load_torch(
        setting_dir
        / "inner_train_drugs.pt"
    ).long().view(-1)

    inner_val = load_torch(
        setting_dir
        / "inner_val_drugs.pt"
    ).long().view(-1)

    outer_train = load_torch(
        setting_dir
        / "outer_train_drugs.pt"
    ).long().view(-1)

    outer_heldout = load_torch(
        setting_dir
        / "outer_heldout_drugs.pt"
    ).long().view(-1)

    s_train = tensor_set(inner_train)
    s_val = tensor_set(inner_val)
    s_outer = tensor_set(outer_train)
    s_hold = tensor_set(outer_heldout)

    partitions[setting] = {
        "inner_train": s_train,
        "inner_val": s_val,
        "outer_train": s_outer,
        "outer_heldout": s_hold,
    }

    check(
        len(s_train) == 685,
        f"{setting}: inner train = 685",
    )

    check(
        len(s_val) == 171,
        f"{setting}: inner val = 171",
    )

    check(
        len(s_outer) == 856,
        f"{setting}: outer train = 856",
    )

    check(
        len(s_hold) == 215,
        f"{setting}: outer heldout = 215",
    )

    check(
        s_train.isdisjoint(s_val),
        f"{setting}: inner train/val disjoint",
    )

    check(
        s_train | s_val == s_outer,
        f"{setting}: inner partition reconstructs outer train",
    )

    check(
        s_outer.isdisjoint(s_hold),
        f"{setting}: outer train/heldout disjoint",
    )

    check(
        len(s_outer | s_hold)
        == NUM_ACTIVE,
        f"{setting}: outer partition reconstructs 1071 active drugs",
    )

    dev_dti = normalize_edge_index(
        load_torch(
            setting_dir
            / "dti_edge_index.pt"
        ),
        f"{setting} v4 DTI",
    )

    independent_mask = torch.tensor(
        [
            int(d) in s_train
            for d in saved_master_dti[0].tolist()
        ],
        dtype=torch.bool,
    )

    independently_filtered = (
        saved_master_dti[
            :,
            independent_mask,
        ]
        .contiguous()
    )

    check(
        dev_dti.shape[1]
        == exp["dti"],
        f"{setting}: development DTI edge count = {exp['dti']}",
    )

    check(
        torch.equal(
            dev_dti,
            independently_filtered,
        ),
        f"{setting}: development DTI exactly equals independent inner-train restriction",
    )

    dti_drugs = tensor_set(
        dev_dti[0]
    )

    check(
        dti_drugs.issubset(
            s_train
        ),
        f"{setting}: DTI drug endpoints only inner-train",
    )

    check(
        len(dti_drugs & s_val) == 0,
        f"{setting}: inner-val DTI exposure = 0",
    )

    check(
        len(dti_drugs & s_hold) == 0,
        f"{setting}: outer-heldout DTI exposure = 0",
    )

    check(
        int(dev_dti[1].min()) >= 0
        and int(dev_dti[1].max()) < NUM_PROTEINS,
        f"{setting}: DTI protein IDs fit ESM namespace",
    )

    ddi = normalize_edge_index(
        load_torch(
            setting_dir
            / "ddi_train_edge_index.pt"
        ),
        f"{setting} DDI",
    )

    check(
        ddi.shape[1] == exp["ddi"],
        f"{setting}: DDI directed edges = {exp['ddi']}",
    )

    ddi_nodes = tensor_set(
        ddi
    )

    check(
        ddi_nodes.issubset(
            s_train
        ),
        f"{setting}: DDI nodes only inner-train",
    )

    check(
        len(ddi_nodes & s_val) == 0,
        f"{setting}: DDI inner-val exposure = 0",
    )

    check(
        len(ddi_nodes & s_hold) == 0,
        f"{setting}: DDI outer-heldout exposure = 0",
    )

    zero_x = load_torch(
        setting_dir
        / "drug_bio_init_zero_128.pt"
    )

    check(
        tuple(zero_x.shape)
        == (NUM_DRUGS, 128),
        f"{setting}: zero drug bio feature shape = (1087,128)",
    )

    check(
        int(torch.count_nonzero(zero_x))
        == 0,
        f"{setting}: zero drug bio feature exactly zero",
    )

    pairs = {}

    for name in [
        "train_pos",
        "train_neg",
        "val_pos",
        "val_neg",
        "test_pos",
        "test_neg",
    ]:

        v4_pair = load_torch(
            setting_dir
            / f"{name}.pt"
        )

        source_pair = load_torch(
            split_dir
            / f"{name}.pt"
        )

        check(
            torch.equal(
                v4_pair,
                source_pair,
            ),
            f"{setting}: {name} exactly preserved from certified split",
        )

        pairs[name] = v4_pair

        n = (
            v4_pair.shape[0]
            if v4_pair.shape[1] == 2
            else v4_pair.shape[1]
        )

        check(
            int(n)
            == exp[name],
            f"{setting}: {name} count = {exp[name]}",
        )

    train_all = (
        pair_set(pairs["train_pos"])
        |
        pair_set(pairs["train_neg"])
    )

    val_all = (
        pair_set(pairs["val_pos"])
        |
        pair_set(pairs["val_neg"])
    )

    test_all = (
        pair_set(pairs["test_pos"])
        |
        pair_set(pairs["test_neg"])
    )

    check(
        len(train_all & val_all) == 0,
        f"{setting}: train/val pair overlap = 0",
    )

    check(
        len(train_all & test_all) == 0,
        f"{setting}: train/test pair overlap = 0",
    )

    check(
        len(val_all & test_all) == 0,
        f"{setting}: val/test pair overlap = 0",
    )

    check(
        len(
            pair_set(pairs["train_pos"])
            &
            pair_set(pairs["train_neg"])
        ) == 0,
        f"{setting}: train positive/negative contradictions = 0",
    )

    check(
        len(
            pair_set(pairs["val_pos"])
            &
            pair_set(pairs["val_neg"])
        ) == 0,
        f"{setting}: val positive/negative contradictions = 0",
    )

    check(
        len(
            pair_set(pairs["test_pos"])
            &
            pair_set(pairs["test_neg"])
        ) == 0,
        f"{setting}: test positive/negative contradictions = 0",
    )

    check(
        semantics(
            pairs["val_pos"],
            s_val,
        )
        == exp["val_semantics"],
        f"{setting}: validation positive cold-start semantics correct",
    )

    check(
        semantics(
            pairs["val_neg"],
            s_val,
        )
        == exp["val_semantics"],
        f"{setting}: validation negative cold-start semantics correct",
    )

    check(
        semantics(
            pairs["test_pos"],
            s_hold,
        )
        == exp["test_semantics"],
        f"{setting}: test positive cold-start semantics correct",
    )

    check(
        semantics(
            pairs["test_neg"],
            s_hold,
        )
        == exp["test_semantics"],
        f"{setting}: test negative cold-start semantics correct",
    )


# =============================================================================
# 6. CROSS-SETTING AUDIT
# =============================================================================

print()
print("=" * 110)
print("6. CROSS-SETTING AUDIT")
print("=" * 110)

s1 = partitions[
    "cold_start_S1"
]

s2 = partitions[
    "cold_start_S2"
]

check(
    len(
        s1["outer_train"]
        &
        s2["outer_train"]
    )
    == 685,
    "S1/S2 outer-train intersection = 685",
)

check(
    len(
        s1["outer_heldout"]
        &
        s2["outer_heldout"]
    )
    == 44,
    "S1/S2 outer-heldout intersection = 44",
)

print(
    "INFO: inner-train intersection =",
    len(
        s1["inner_train"]
        &
        s2["inner_train"]
    ),
)

print(
    "INFO: inner-val intersection   =",
    len(
        s1["inner_val"]
        &
        s2["inner_val"]
    ),
)


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 110)
print("SCRIPT 117 FINAL SUMMARY")
print("=" * 110)

print("PASS checks :", PASS)
print("Warnings    :", WARN)
print("Failures    :", FAIL)

print()

if FAIL == 0 and WARN == 0:

    print(
        "FINAL SEMANTIC BIOLOGICAL V4 CERTIFICATION: PASSED"
    )

    print(
        "PROTEIN FEATURES: CERTIFIED SEQUENCE-DERIVED ESM2."
    )

    print(
        "DTI SEMANTIC ALIGNMENT: CERTIFIED."
    )

    print(
        "PPI SEMANTIC ALIGNMENT: CERTIFIED."
    )

    print(
        "DDI TRAIN-ONLY GRAPH: CERTIFIED."
    )

    print(
        "INNER-VALIDATION DTI EXPOSURE: ZERO."
    )

    print(
        "OUTER-HELDOUT DTI EXPOSURE: ZERO."
    )

    print(
        "OUTER TEST: STILL SEALED."
    )

    print()
    print(
        "BIOLOGICAL V4 DATA LAYER MAY NOW BE FROZEN."
    )

    print(
        "SAFE NEXT STEP: architecture/code audit only."
    )

    sys.exit(0)

else:

    print(
        "FINAL SEMANTIC BIOLOGICAL V4 CERTIFICATION: FAILED"
    )

    print(
        "DO NOT TRAIN."
    )

    sys.exit(1)
