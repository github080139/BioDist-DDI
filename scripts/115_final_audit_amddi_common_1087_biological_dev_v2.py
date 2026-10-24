from pathlib import Path
import hashlib
import json
import sys

import torch


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_biological_dev_v2"
)

ROOT_MANIFEST = BIO_ROOT / "manifest.json"

PPI_CERT = BIO_ROOT / "ppi_certification.json"

INSTALLED_PPI = BIO_ROOT / "ppi_edge_index.pt"

SOURCE_PPI = (
    PROJECT_ROOT
    / "graphs"
    / "dpddi"
    / "ppi_edge_index.pt"
)

MASTER_DTI = (
    PROJECT_ROOT
    / "graphs"
    / "amddi_common"
    / "dti_edge_index.pt"
)

LEAKAGE_FREE_ROOT = (
    PROJECT_ROOT
    / "graphs"
    / "amddi_common_1087_leakage_free_training"
)

DEV_SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_coldstart_dev_splits_v2"
)

SETTINGS = [
    "cold_start_S1",
    "cold_start_S2",
]

HISTORICAL_SETTINGS = [
    "transductive",
    "cold_start_S1",
    "cold_start_S2",
]


# =============================================================================
# AUTHORITATIVE CONSTANTS
# =============================================================================

EXPECTED_NUM_DRUGS = 1087
EXPECTED_ACTIVE_DRUGS = 1071

EXPECTED_PPI_SHAPE = (2, 107944)
EXPECTED_PPI_EDGES = 107944

EXPECTED_DTI_SHAPE = (2, 7408)
EXPECTED_DTI_EDGES = 7408

EXPECTED_PROTEIN_NUM_NODES = 2321

EXPECTED_PPI_HASH = (
    "74b7ebb904b9f6abd572e823d4177f05"
    "e67cc53c04baa1b1eff6cdb2b559edb1"
)

EXPECTED_PARTITIONS = {
    "cold_start_S1": {
        "inner_train": 685,
        "inner_val": 171,
        "outer_train": 856,
        "outer_heldout": 215,
        "dti_edges": 4817,
        "train_pos": 14494,
        "train_neg": 14494,
        "val_pos": 7379,
        "val_neg": 7379,
        "test_pos": 11713,
        "test_neg": 11713,
        "ddi_directed": 28988,
        "val_semantics": {0: 0, 1: 7379, 2: 0},
        "test_semantics": {0: 0, 1: 11713, 2: 0},
    },
    "cold_start_S2": {
        "inner_train": 685,
        "inner_val": 171,
        "outer_train": 856,
        "outer_heldout": 215,
        "dti_edges": 4703,
        "train_pos": 16903,
        "train_neg": 16903,
        "val_pos": 686,
        "val_neg": 686,
        "test_pos": 1086,
        "test_neg": 1086,
        "ddi_directed": 33806,
        "val_semantics": {0: 0, 1: 0, 2: 686},
        "test_semantics": {0: 0, 1: 0, 2: 1086},
    },
}


# =============================================================================
# AUDIT STATE
# =============================================================================

PASS_COUNT = 0
WARN_COUNT = 0
FAIL_COUNT = 0


def passed(msg):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"PASS: {msg}")


def warned(msg):
    global WARN_COUNT
    WARN_COUNT += 1
    print(f"WARNING: {msg}")


def failed(msg):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"FAIL: {msg}")


def check(condition, msg):
    if condition:
        passed(msg)
        return True
    else:
        failed(msg)
        return False


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


def normalize_edge_index(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{name}: expected tensor, got {type(x)}"
        )

    x = x.detach().cpu().long().contiguous()

    if x.ndim != 2:
        raise RuntimeError(
            f"{name}: expected rank-2 tensor, got {tuple(x.shape)}"
        )

    if x.shape[0] == 2:
        return x

    if x.shape[1] == 2:
        return x.t().contiguous()

    raise RuntimeError(
        f"{name}: expected [2,E] or [E,2], got {tuple(x.shape)}"
    )


def tensor_sha256(x):
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

    if x.ndim != 2:
        raise RuntimeError(
            f"pair tensor malformed: {tuple(x.shape)}"
        )

    if x.shape[1] == 2:
        pass
    elif x.shape[0] == 2:
        x = x.t().contiguous()
    else:
        raise RuntimeError(
            f"pair tensor malformed: {tuple(x.shape)}"
        )

    result = set()

    for a, b in x.tolist():
        a = int(a)
        b = int(b)

        if a <= b:
            result.add((a, b))
        else:
            result.add((b, a))

    return result


def unseen_semantics(pairs, unseen_ids):
    pairs = pairs.detach().cpu().long()

    if pairs.shape[1] != 2:
        pairs = pairs.t().contiguous()

    unseen_ids = set(
        int(x)
        for x in unseen_ids
    )

    counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    for a, b in pairs.tolist():
        n = (
            int(int(a) in unseen_ids)
            + int(int(b) in unseen_ids)
        )

        counts[n] += 1

    return counts


def load_partition(setting, name):
    path = (
        BIO_ROOT
        / setting
        / f"{name}_drugs.pt"
    )

    return load_torch(path).long().view(-1)


def load_pair(setting, name):
    candidates = [
        BIO_ROOT / setting / f"{name}.pt",
        DEV_SPLIT_ROOT / setting / f"{name}.pt",
    ]

    for path in candidates:
        if path.exists():
            return load_torch(path)

    raise FileNotFoundError(
        f"Could not locate {setting}/{name}.pt"
    )


def load_setting_tensor(setting, filename):
    path = BIO_ROOT / setting / filename

    if not path.exists():
        raise FileNotFoundError(path)

    return load_torch(path)


# =============================================================================
# HISTORICAL GRAPH HELPERS
# =============================================================================

PPI_EDGE_TYPE = (
    "protein",
    "ppi",
    "protein",
)


def get_ppi_relation(graph):
    if not hasattr(graph, "edge_types"):
        raise RuntimeError(
            "Historical graph lacks edge_types"
        )

    if PPI_EDGE_TYPE not in graph.edge_types:
        raise RuntimeError(
            f"Historical graph missing {PPI_EDGE_TYPE}"
        )

    return normalize_edge_index(
        graph[PPI_EDGE_TYPE].edge_index,
        "historical PPI",
    )


def get_protein_num_nodes(graph):
    if not hasattr(graph, "node_types"):
        raise RuntimeError(
            "Historical graph lacks node_types"
        )

    if "protein" not in graph.node_types:
        raise RuntimeError(
            "Historical graph lacks protein node type"
        )

    store = graph["protein"]

    try:
        num_nodes = store.num_nodes
    except Exception:
        num_nodes = None

    if num_nodes is not None:
        return int(num_nodes)

    if hasattr(store, "x") and store.x is not None:
        return int(store.x.shape[0])

    raise RuntimeError(
        "Cannot determine historical protein node count"
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 110)
print("AM-DDI COMMON-1087")
print(
    "SCRIPT 115 — FINAL INDEPENDENT BIOLOGICAL DEVELOPMENT AUDIT"
)
print("=" * 110)


# =============================================================================
# GLOBAL FILE EXISTENCE
# =============================================================================

required_paths = [
    BIO_ROOT,
    ROOT_MANIFEST,
    PPI_CERT,
    INSTALLED_PPI,
    SOURCE_PPI,
    MASTER_DTI,
    LEAKAGE_FREE_ROOT,
    DEV_SPLIT_ROOT,
]

all_exist = True

for path in required_paths:
    if not path.exists():
        failed(
            f"required path missing: {path}"
        )
        all_exist = False

if not all_exist:
    print()
    print("FATAL: required paths missing.")
    sys.exit(1)

passed(
    "all required global artifacts exist"
)


# =============================================================================
# ROOT MANIFEST
# =============================================================================

manifest = load_json(
    ROOT_MANIFEST
)

check(
    manifest.get("version")
    == "biological_dev_v2",
    "root manifest version = biological_dev_v2",
)

check(
    int(
        manifest.get(
            "num_drugs",
            -1,
        )
    )
    == EXPECTED_NUM_DRUGS,
    "root manifest preserves 1087 common drugs",
)

check(
    int(
        manifest.get(
            "num_active_drugs",
            EXPECTED_ACTIVE_DRUGS,
        )
    )
    == EXPECTED_ACTIVE_DRUGS,
    "root manifest preserves 1071 DDI-active drugs",
)

check(
    manifest.get("ppi_status")
    == "loaded",
    "root manifest says PPI loaded",
)

check(
    manifest.get("ppi_certified")
    is True,
    "root manifest says PPI certified",
)

check(
    int(
        manifest.get(
            "ppi_edges",
            -1,
        )
    )
    == EXPECTED_PPI_EDGES,
    "root manifest records 107944 PPI edges",
)

check(
    int(
        manifest.get(
            "protein_num_nodes",
            -1,
        )
    )
    == EXPECTED_PROTEIN_NUM_NODES,
    "root manifest records 2321 protein nodes",
)

check(
    manifest.get("ppi_sha256")
    == EXPECTED_PPI_HASH,
    "root manifest PPI hash matches certified hash",
)


# =============================================================================
# PPI CERTIFICATION JSON
# =============================================================================

print()
print("=" * 110)
print("PPI CERTIFICATION AUDIT")
print("=" * 110)

ppi_cert = load_json(
    PPI_CERT
)

check(
    ppi_cert.get("status")
    == "PASSED",
    "Script 114 certification status = PASSED",
)

check(
    ppi_cert.get("certification")
    == "ppi_biological_dev_v2",
    "PPI certification type is correct",
)

check(
    int(
        ppi_cert.get(
            "ppi_edges",
            -1,
        )
    )
    == EXPECTED_PPI_EDGES,
    "PPI certification records 107944 edges",
)

protein_ns = ppi_cert.get(
    "protein_namespace",
    {},
)

check(
    int(
        protein_ns.get(
            "num_nodes",
            -1,
        )
    )
    == EXPECTED_PROTEIN_NUM_NODES,
    "PPI certification records 2321-node protein namespace",
)

policy = ppi_cert.get(
    "policy",
    {},
)

check(
    policy.get(
        "ppi_is_protein_only"
    )
    is True,
    "PPI policy confirms protein-only relation",
)

check(
    policy.get(
        "inner_validation_drug_exposure_through_ppi"
    )
    is False,
    "PPI policy confirms no inner-validation drug exposure",
)

check(
    policy.get(
        "outer_heldout_drug_exposure_through_ppi"
    )
    is False,
    "PPI policy confirms no outer-heldout drug exposure",
)

check(
    policy.get(
        "old_dti_derived_bio128_used"
    )
    is False,
    "PPI certification confirms old DTI-derived bio128 not used",
)

check(
    policy.get(
        "outer_test_used"
    )
    is False,
    "PPI certification confirms outer test was not used",
)


# =============================================================================
# LOAD PPI + DTI
# =============================================================================

source_ppi = normalize_edge_index(
    load_torch(
        SOURCE_PPI
    ),
    "source PPI",
)

installed_ppi = normalize_edge_index(
    load_torch(
        INSTALLED_PPI
    ),
    "installed PPI",
)

master_dti = normalize_edge_index(
    load_torch(
        MASTER_DTI
    ),
    "master DTI",
)

check(
    tuple(
        source_ppi.shape
    )
    == EXPECTED_PPI_SHAPE,
    "source PPI shape exactly (2,107944)",
)

check(
    tuple(
        installed_ppi.shape
    )
    == EXPECTED_PPI_SHAPE,
    "installed PPI shape exactly (2,107944)",
)

check(
    tuple(
        master_dti.shape
    )
    == EXPECTED_DTI_SHAPE,
    "master DTI shape exactly (2,7408)",
)

check(
    torch.equal(
        source_ppi,
        installed_ppi,
    ),
    "installed PPI exactly equals canonical source PPI",
)

source_hash = tensor_sha256(
    source_ppi
)

installed_hash = tensor_sha256(
    installed_ppi
)

check(
    source_hash
    == EXPECTED_PPI_HASH,
    "source PPI SHA256 matches certified hash",
)

check(
    installed_hash
    == EXPECTED_PPI_HASH,
    "installed PPI SHA256 matches certified hash",
)

check(
    source_hash
    == installed_hash,
    "source and installed PPI hashes are identical",
)

check(
    ppi_cert.get(
        "source_ppi_sha256"
    )
    == EXPECTED_PPI_HASH,
    "certification source hash matches actual PPI",
)

check(
    ppi_cert.get(
        "installed_ppi_sha256"
    )
    == EXPECTED_PPI_HASH,
    "certification installed hash matches actual PPI",
)


# =============================================================================
# PROTEIN NAMESPACE
# =============================================================================

ppi_min = int(
    installed_ppi.min()
)

ppi_max = int(
    installed_ppi.max()
)

dti_drug_min = int(
    master_dti[0].min()
)

dti_drug_max = int(
    master_dti[0].max()
)

dti_protein_min = int(
    master_dti[1].min()
)

dti_protein_max = int(
    master_dti[1].max()
)

check(
    ppi_min >= 0,
    "PPI contains no negative protein IDs",
)

check(
    ppi_max
    < EXPECTED_PROTEIN_NUM_NODES,
    "PPI max protein ID fits inside 2321-node namespace",
)

check(
    dti_protein_min >= 0,
    "DTI contains no negative protein IDs",
)

check(
    dti_protein_max
    < EXPECTED_PROTEIN_NUM_NODES,
    "DTI max protein ID fits inside 2321-node namespace",
)

check(
    dti_drug_min >= 0,
    "DTI contains no negative drug IDs",
)

check(
    dti_drug_max
    < EXPECTED_NUM_DRUGS,
    "DTI drug IDs fit inside common-1087 namespace",
)

ppi_nodes = tensor_set(
    installed_ppi
)

dti_proteins = tensor_set(
    master_dti[1]
)

print()
print("Protein namespace diagnostics")
print("-" * 110)
print(
    f"Protein num_nodes          : {EXPECTED_PROTEIN_NUM_NODES}"
)
print(
    f"PPI min/max                : {ppi_min} / {ppi_max}"
)
print(
    f"DTI protein min/max        : {dti_protein_min} / {dti_protein_max}"
)
print(
    f"PPI unique proteins        : {len(ppi_nodes)}"
)
print(
    f"DTI unique proteins        : {len(dti_proteins)}"
)
print(
    f"Intersection               : {len(ppi_nodes & dti_proteins)}"
)
print(
    f"PPI-only                   : {len(ppi_nodes - dti_proteins)}"
)
print(
    f"DTI-only                   : {len(dti_proteins - ppi_nodes)}"
)

check(
    len(ppi_nodes)
    == 2266,
    "PPI unique protein count = 2266",
)

check(
    len(dti_proteins)
    == 1072,
    "DTI unique protein count = 1072",
)

check(
    len(
        ppi_nodes
        & dti_proteins
    )
    == 1044,
    "PPI/DTI protein intersection = 1044",
)

check(
    len(
        ppi_nodes
        - dti_proteins
    )
    == 1222,
    "PPI-only protein count = 1222",
)

check(
    len(
        dti_proteins
        - ppi_nodes
    )
    == 28,
    "DTI-only protein count = 28",
)


# =============================================================================
# HISTORICAL PPI EXACT-MATCH AUDIT
# =============================================================================

print()
print("=" * 110)
print("HISTORICAL LEAKAGE-FREE PPI EXACT MATCH")
print("=" * 110)

historical_node_counts = []

for setting in HISTORICAL_SETTINGS:
    graph_path = (
        LEAKAGE_FREE_ROOT
        / f"{setting}_train_graph.pt"
    )

    if not graph_path.exists():
        failed(
            f"{setting}: historical leakage-free graph missing"
        )
        continue

    graph = load_torch(
        graph_path
    )

    hist_ppi = get_ppi_relation(
        graph
    )

    protein_nodes = get_protein_num_nodes(
        graph
    )

    historical_node_counts.append(
        protein_nodes
    )

    check(
        tuple(
            hist_ppi.shape
        )
        == EXPECTED_PPI_SHAPE,
        f"{setting}: historical PPI shape correct",
    )

    check(
        torch.equal(
            hist_ppi,
            installed_ppi,
        ),
        f"{setting}: historical PPI exactly equals installed canonical PPI",
    )

    check(
        tensor_sha256(
            hist_ppi
        )
        == EXPECTED_PPI_HASH,
        f"{setting}: historical PPI hash matches certified hash",
    )

    check(
        protein_nodes
        == EXPECTED_PROTEIN_NUM_NODES,
        f"{setting}: protein node count = 2321",
    )

check(
    len(
        historical_node_counts
    )
    == 3,
    "all three historical settings were audited",
)

check(
    set(
        historical_node_counts
    )
    == {
        EXPECTED_PROTEIN_NUM_NODES
    },
    "all historical graphs share identical 2321-node protein namespace",
)


# =============================================================================
# SETTING-SPECIFIC BIOLOGICAL AUDIT
# =============================================================================

print()
print("=" * 110)
print("STRICT COLD-START BIOLOGICAL DEVELOPMENT AUDIT")
print("=" * 110)

partitions = {}

for setting in SETTINGS:
    print()
    print("=" * 110)
    print(setting)
    print("=" * 110)

    exp = EXPECTED_PARTITIONS[
        setting
    ]

    inner_train = load_partition(
        setting,
        "inner_train",
    )

    inner_val = load_partition(
        setting,
        "inner_val",
    )

    outer_train = load_partition(
        setting,
        "outer_train",
    )

    outer_heldout = load_partition(
        setting,
        "outer_heldout",
    )

    partitions[
        setting
    ] = {
        "inner_train":
            tensor_set(
                inner_train
            ),
        "inner_val":
            tensor_set(
                inner_val
            ),
        "outer_train":
            tensor_set(
                outer_train
            ),
        "outer_heldout":
            tensor_set(
                outer_heldout
            ),
    }

    s_inner_train = partitions[
        setting
    ]["inner_train"]

    s_inner_val = partitions[
        setting
    ]["inner_val"]

    s_outer_train = partitions[
        setting
    ]["outer_train"]

    s_outer_heldout = partitions[
        setting
    ]["outer_heldout"]

    check(
        len(
            s_inner_train
        )
        == exp["inner_train"],
        f"{setting}: inner_train size = {exp['inner_train']}",
    )

    check(
        len(
            s_inner_val
        )
        == exp["inner_val"],
        f"{setting}: inner_val size = {exp['inner_val']}",
    )

    check(
        len(
            s_outer_train
        )
        == exp["outer_train"],
        f"{setting}: outer_train size = {exp['outer_train']}",
    )

    check(
        len(
            s_outer_heldout
        )
        == exp["outer_heldout"],
        f"{setting}: outer_heldout size = {exp['outer_heldout']}",
    )

    check(
        s_inner_train.isdisjoint(
            s_inner_val
        ),
        f"{setting}: inner train/val disjoint",
    )

    check(
        s_inner_train
        | s_inner_val
        == s_outer_train,
        f"{setting}: inner partitions reconstruct outer train",
    )

    check(
        s_outer_train.isdisjoint(
            s_outer_heldout
        ),
        f"{setting}: outer train/heldout disjoint",
    )

    check(
        len(
            s_outer_train
            | s_outer_heldout
        )
        == EXPECTED_ACTIVE_DRUGS,
        f"{setting}: outer partition reconstructs 1071 active drugs",
    )

    # -------------------------------------------------------------------------
    # FILTERED DTI
    # -------------------------------------------------------------------------

    dti_candidates = [
        BIO_ROOT
        / setting
        / "dti_edge_index.pt",

        BIO_ROOT
        / setting
        / "filtered_dti_edge_index.pt",

        BIO_ROOT
        / setting
        / "dti_dev_edge_index.pt",
    ]

    filtered_dti_path = None

    for path in dti_candidates:
        if path.exists():
            filtered_dti_path = path
            break

    if filtered_dti_path is None:
        failed(
            f"{setting}: cannot locate filtered DTI artifact"
        )

        continue

    filtered_dti = normalize_edge_index(
        load_torch(
            filtered_dti_path
        ),
        f"{setting} filtered DTI",
    )

    check(
        int(
            filtered_dti.shape[1]
        )
        == exp["dti_edges"],
        f"{setting}: filtered DTI edge count = {exp['dti_edges']}",
    )

    filtered_drugs = tensor_set(
        filtered_dti[0]
    )

    check(
        filtered_drugs.issubset(
            s_inner_train
        ),
        f"{setting}: every DTI drug endpoint is inner-train",
    )

    check(
        len(
            filtered_drugs
            & s_inner_val
        )
        == 0,
        f"{setting}: inner-validation DTI exposure = 0",
    )

    check(
        len(
            filtered_drugs
            & s_outer_heldout
        )
        == 0,
        f"{setting}: outer-heldout DTI exposure = 0",
    )

    expected_mask = torch.tensor(
        [
            int(x) in s_inner_train
            for x in master_dti[0].tolist()
        ],
        dtype=torch.bool,
    )

    independently_filtered = (
        master_dti[
            :,
            expected_mask,
        ]
        .contiguous()
    )

    check(
        torch.equal(
            filtered_dti,
            independently_filtered,
        ),
        f"{setting}: filtered DTI exactly equals independent master-DTI restriction",
    )

    # -------------------------------------------------------------------------
    # ZERO BIO INITIALIZATION
    # -------------------------------------------------------------------------

    zero_candidates = [
        BIO_ROOT
        / setting
        / "drug_bio_init_zero_128.pt",

        BIO_ROOT
        / "drug_bio_init_zero_128.pt",
    ]

    zero_path = None

    for path in zero_candidates:
        if path.exists():
            zero_path = path
            break

    if zero_path is None:
        failed(
            f"{setting}: zero biological initialization not found"
        )

    else:
        zero_bio = load_torch(
            zero_path
        )

        check(
            tuple(
                zero_bio.shape
            )
            == (
                EXPECTED_NUM_DRUGS,
                128,
            ),
            f"{setting}: zero bio initialization shape = (1087,128)",
        )

        check(
            torch.count_nonzero(
                zero_bio
            ).item()
            == 0,
            f"{setting}: biological drug initialization exactly all-zero",
        )

    # -------------------------------------------------------------------------
    # DDI TRAIN GRAPH
    # -------------------------------------------------------------------------

    ddi_candidates = [
        BIO_ROOT
        / setting
        / "ddi_train_edge_index.pt",

        BIO_ROOT
        / setting
        / "ddi_edge_index.pt",

        BIO_ROOT
        / setting
        / "train_ddi_edge_index.pt",
    ]

    ddi_path = None

    for path in ddi_candidates:
        if path.exists():
            ddi_path = path
            break

    if ddi_path is None:
        failed(
            f"{setting}: cannot locate DDI training graph"
        )

    else:
        ddi_graph = normalize_edge_index(
            load_torch(
                ddi_path
            ),
            f"{setting} DDI training graph",
        )

        check(
            int(
                ddi_graph.shape[1]
            )
            == exp["ddi_directed"],
            f"{setting}: DDI directed edges = {exp['ddi_directed']}",
        )

        ddi_nodes = tensor_set(
            ddi_graph
        )

        check(
            ddi_nodes.issubset(
                s_inner_train
            ),
            f"{setting}: DDI graph contains only inner-training drugs",
        )

        check(
            len(
                ddi_nodes
                & s_inner_val
            )
            == 0,
            f"{setting}: DDI graph inner-val exposure = 0",
        )

        check(
            len(
                ddi_nodes
                & s_outer_heldout
            )
            == 0,
            f"{setting}: DDI graph outer-heldout exposure = 0",
        )

    # -------------------------------------------------------------------------
    # PAIRS
    # -------------------------------------------------------------------------

    train_pos = load_pair(
        setting,
        "train_pos",
    )

    train_neg = load_pair(
        setting,
        "train_neg",
    )

    val_pos = load_pair(
        setting,
        "val_pos",
    )

    val_neg = load_pair(
        setting,
        "val_neg",
    )

    test_pos = load_pair(
        setting,
        "test_pos",
    )

    test_neg = load_pair(
        setting,
        "test_neg",
    )

    pair_tensors = {
        "train_pos":
            train_pos,

        "train_neg":
            train_neg,

        "val_pos":
            val_pos,

        "val_neg":
            val_neg,

        "test_pos":
            test_pos,

        "test_neg":
            test_neg,
    }

    for name, tensor in pair_tensors.items():
        expected_n = exp[
            name
        ]

        n = (
            tensor.shape[0]
            if tensor.shape[1] == 2
            else tensor.shape[1]
        )

        check(
            int(n)
            == expected_n,
            f"{setting}: {name} count = {expected_n}",
        )

    train_pos_set = pair_set(
        train_pos
    )

    train_neg_set = pair_set(
        train_neg
    )

    val_pos_set = pair_set(
        val_pos
    )

    val_neg_set = pair_set(
        val_neg
    )

    test_pos_set = pair_set(
        test_pos
    )

    test_neg_set = pair_set(
        test_neg
    )

    train_all = (
        train_pos_set
        | train_neg_set
    )

    val_all = (
        val_pos_set
        | val_neg_set
    )

    test_all = (
        test_pos_set
        | test_neg_set
    )

    check(
        len(
            train_all
            & val_all
        )
        == 0,
        f"{setting}: train/val pair overlap = 0",
    )

    check(
        len(
            train_all
            & test_all
        )
        == 0,
        f"{setting}: train/test pair overlap = 0",
    )

    check(
        len(
            val_all
            & test_all
        )
        == 0,
        f"{setting}: val/test pair overlap = 0",
    )

    check(
        len(
            train_pos_set
            & train_neg_set
        )
        == 0,
        f"{setting}: train positive/negative contradictions = 0",
    )

    check(
        len(
            val_pos_set
            & val_neg_set
        )
        == 0,
        f"{setting}: val positive/negative contradictions = 0",
    )

    check(
        len(
            test_pos_set
            & test_neg_set
        )
        == 0,
        f"{setting}: test positive/negative contradictions = 0",
    )

    val_pos_sem = unseen_semantics(
        val_pos,
        s_inner_val,
    )

    val_neg_sem = unseen_semantics(
        val_neg,
        s_inner_val,
    )

    test_pos_sem = unseen_semantics(
        test_pos,
        s_outer_heldout,
    )

    test_neg_sem = unseen_semantics(
        test_neg,
        s_outer_heldout,
    )

    check(
        val_pos_sem
        == exp["val_semantics"],
        f"{setting}: validation positive cold-start semantics correct",
    )

    check(
        val_neg_sem
        == exp["val_semantics"],
        f"{setting}: validation negative cold-start semantics correct",
    )

    check(
        test_pos_sem
        == exp["test_semantics"],
        f"{setting}: outer-test positive cold-start semantics preserved",
    )

    check(
        test_neg_sem
        == exp["test_semantics"],
        f"{setting}: outer-test negative cold-start semantics preserved",
    )


# =============================================================================
# CROSS-SETTING
# =============================================================================

print()
print("=" * 110)
print("CROSS-SETTING FINAL CHECK")
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
        & s2["outer_train"]
    )
    == 685,
    "S1/S2 outer-train intersection = 685",
)

check(
    len(
        s1["outer_heldout"]
        & s2["outer_heldout"]
    )
    == 44,
    "S1/S2 outer-heldout intersection = 44",
)

print(
    "INFO: S1/S2 inner-train intersection =",
    len(
        s1["inner_train"]
        & s2["inner_train"]
    ),
)

print(
    "INFO: S1/S2 inner-val intersection   =",
    len(
        s1["inner_val"]
        & s2["inner_val"]
    ),
)

check(
    (
        s1["inner_train"]
        | s1["inner_val"]
    )
    == s1["outer_train"],
    "S1 inner development partition reconstructs its own outer train",
)

check(
    (
        s2["inner_train"]
        | s2["inner_val"]
    )
    == s2["outer_train"],
    "S2 inner development partition reconstructs its own outer train",
)


# =============================================================================
# FINAL POLICY CHECKS
# =============================================================================

print()
print("=" * 110)
print("FINAL POLICY CHECK")
print("=" * 110)

check(
    manifest.get(
        "ppi_historical_exact_match_all_settings"
    )
    is True,
    "manifest records historical PPI exact-match certification",
)

important_policy = manifest.get(
    "important_policy",
    {},
)

check(
    important_policy.get(
        "ppi_certified"
    )
    is True,
    "manifest policy confirms PPI certified",
)

check(
    important_policy.get(
        "ppi_is_protein_only"
    )
    is True,
    "manifest policy confirms PPI is protein-only",
)

check(
    important_policy.get(
        "ppi_contains_drug_nodes"
    )
    is False,
    "manifest policy confirms PPI contains no drug nodes",
)

check(
    important_policy.get(
        "outer_test_used_for_ppi_certification"
    )
    is False,
    "manifest policy confirms outer test not used for PPI certification",
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 110)
print("FINAL BIOLOGICAL DEVELOPMENT CERTIFICATION SUMMARY")
print("=" * 110)

print(
    f"PASS checks : {PASS_COUNT}"
)

print(
    f"Warnings    : {WARN_COUNT}"
)

print(
    f"Failures    : {FAIL_COUNT}"
)

print()

if FAIL_COUNT == 0 and WARN_COUNT == 0:
    print(
        "FINAL BIOLOGICAL DEVELOPMENT CERTIFICATION: PASSED"
    )

    print(
        "DTI LAYER: CERTIFIED."
    )

    print(
        "DDI TRAINING LAYER: CERTIFIED."
    )

    print(
        "PPI LAYER: CERTIFIED."
    )

    print(
        "PROTEIN NAMESPACE: CERTIFIED."
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

    print()
    print(
        "BIOLOGICAL DATA LAYER MAY NOW BE FROZEN."
    )

    print(
        "SAFE NEXT STEP: architecture audit before HGT/model training."
    )

    sys.exit(0)

else:
    print(
        "FINAL BIOLOGICAL DEVELOPMENT CERTIFICATION: FAILED"
    )

    print(
        "DO NOT TRAIN."
    )

    sys.exit(1)
