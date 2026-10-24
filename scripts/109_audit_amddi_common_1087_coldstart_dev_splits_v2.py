from pathlib import Path
import hashlib
import json
import sys

import torch


# =============================================================================
# CONFIG
# =============================================================================

NUM_NODES = 1087
ACTIVE_DDI_DRUGS = 1071
OUTER_TRAIN_EXPECTED = 856
OUTER_HELDOUT_EXPECTED = 215
INNER_TRAIN_EXPECTED = 685
INNER_VAL_EXPECTED = 171

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTER_ROOT = PROJECT_ROOT / "data" / "amddi_common_splits"

DEV_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_coldstart_dev_splits_v2"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "amddi_common"
    / "drug_mapping.csv"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_leakage_free_features"
)

STRUCTURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_structure"
    / "molecular_graphs.pt"
)

SETTINGS = [
    "cold_start_S1",
    "cold_start_S2",
]

NAMES = [
    "train_pos",
    "train_neg",
    "val_pos",
    "val_neg",
    "test_pos",
    "test_neg",
]

failures = []
warnings = []
passes = []


# =============================================================================
# REPORTING
# =============================================================================

def PASS(msg):
    passes.append(msg)
    print(f"[PASS] {msg}")


def FAIL(msg):
    failures.append(msg)
    print(f"[FAIL] {msg}")


def WARN(msg):
    warnings.append(msg)
    print(f"[WARN] {msg}")


# =============================================================================
# HELPERS
# =============================================================================

def load_pair(path):
    if not path.exists():
        raise FileNotFoundError(path)

    x = torch.load(path, map_location="cpu")

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{path} is not Tensor")

    x = x.detach().cpu().long().contiguous()

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(
            f"{path}: expected [N,2], got {tuple(x.shape)}"
        )

    return x


def load_vector(path):
    if not path.exists():
        raise FileNotFoundError(path)

    x = torch.load(path, map_location="cpu")

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{path} is not Tensor")

    return x.detach().cpu().long().reshape(-1).contiguous()


def canonical(x):
    a = torch.minimum(x[:, 0], x[:, 1])
    b = torch.maximum(x[:, 0], x[:, 1])
    return torch.stack([a, b], dim=1)


def pair_set(x):
    return {
        (int(a), int(b))
        for a, b in canonical(x).tolist()
    }


def node_set(*xs):
    if not xs:
        return set()

    return {
        int(v)
        for v in torch.cat(xs, dim=0).reshape(-1).tolist()
    }


def tensor_hash(x):
    x = x.detach().cpu().contiguous()

    h = hashlib.sha256()
    h.update(x.numpy().tobytes())

    return h.hexdigest()


def membership_distribution(x, target):
    d = {0: 0, 1: 0, 2: 0}

    for a, b in x.tolist():
        n = int(a in target) + int(b in target)
        d[n] += 1

    return d


def audit_pair_tensor(label, x):
    if len(x) == 0:
        FAIL(f"{label}: empty")
        return

    if int(x.min()) < 0 or int(x.max()) >= NUM_NODES:
        FAIL(f"{label}: node IDs outside 0..1086")
    else:
        PASS(f"{label}: node IDs valid")

    self_pairs = int((x[:, 0] == x[:, 1]).sum())

    if self_pairs:
        FAIL(f"{label}: {self_pairs} self-pairs")
    else:
        PASS(f"{label}: no self-pairs")

    if len(pair_set(x)) != len(x):
        FAIL(f"{label}: duplicate undirected pairs")
    else:
        PASS(f"{label}: unique undirected pairs")


# =============================================================================
# HEADER
# =============================================================================

print("=" * 105)
print("AM-DDI COMMON-1087")
print("MASTER PRE-TRAINING CERTIFICATION V2")
print("=" * 105)

print(
    "\nThis audit independently reconstructs the original "
    "S1/S2 outer cold-start partitions."
)

print(
    "It does NOT trust Script 108 v2's PASS messages.\n"
)


# =============================================================================
# 1. LOAD MANIFEST
# =============================================================================

manifest_path = DEV_ROOT / "manifest.json"

if not manifest_path.exists():
    FAIL("v2 manifest missing")
    manifest = {}
else:
    with open(manifest_path) as f:
        manifest = json.load(f)

    PASS("v2 manifest exists")

    if manifest.get("protocol_version") == \
            "AMDDI_common1087_strict_coldstart_dev_v2":
        PASS("protocol version is v2")
    else:
        FAIL("unexpected protocol version")


# =============================================================================
# 2. INDEPENDENT SETTING-SPECIFIC AUDIT
# =============================================================================

outer_partitions = {}
dev_data = {}

for setting in SETTINGS:

    print()
    print("=" * 105)
    print(setting)
    print("=" * 105)

    # -------------------------------------------------------------------------
    # Load ORIGINAL outer tensors
    # -------------------------------------------------------------------------

    original = {
        name: load_pair(
            OUTER_ROOT / setting / f"{name}.pt"
        )
        for name in NAMES
    }

    original_train_nodes = node_set(
        original["train_pos"],
        original["train_neg"],
    )

    original_test_nodes = node_set(
        original["test_pos"],
        original["test_neg"],
    )

    original_heldout = (
        original_test_nodes
        - original_train_nodes
    )

    outer_partitions[setting] = {
        "train": original_train_nodes,
        "heldout": original_heldout,
    }

    print("Original outer train   :", len(original_train_nodes))
    print("Original outer heldout :", len(original_heldout))
    print(
        "Original overlap       :",
        len(original_train_nodes & original_heldout),
    )

    if len(original_train_nodes) == OUTER_TRAIN_EXPECTED:
        PASS(f"{setting}: original outer train = 856")
    else:
        FAIL(
            f"{setting}: original outer train "
            f"{len(original_train_nodes)} != 856"
        )

    if len(original_heldout) == OUTER_HELDOUT_EXPECTED:
        PASS(f"{setting}: original outer heldout = 215")
    else:
        FAIL(
            f"{setting}: original outer heldout "
            f"{len(original_heldout)} != 215"
        )

    if original_train_nodes & original_heldout:
        FAIL(f"{setting}: original outer train/heldout overlap")
    else:
        PASS(f"{setting}: original outer partition disjoint")

    if len(original_train_nodes | original_heldout) == ACTIVE_DDI_DRUGS:
        PASS(f"{setting}: outer partition reconstructs 1071 active drugs")
    else:
        FAIL(f"{setting}: outer partition does not reconstruct 1071 drugs")

    # -------------------------------------------------------------------------
    # Load V2 drug partitions
    # -------------------------------------------------------------------------

    root = DEV_ROOT / setting

    inner_train_t = load_vector(root / "inner_train_drugs.pt")
    inner_val_t = load_vector(root / "inner_val_drugs.pt")
    stored_outer_train_t = load_vector(root / "outer_train_drugs.pt")
    stored_outer_heldout_t = load_vector(root / "outer_heldout_drugs.pt")

    inner_train = set(inner_train_t.tolist())
    inner_val = set(inner_val_t.tolist())
    stored_outer_train = set(stored_outer_train_t.tolist())
    stored_outer_heldout = set(stored_outer_heldout_t.tolist())

    print()
    print("V2 inner train :", len(inner_train))
    print("V2 inner val   :", len(inner_val))

    if len(inner_train) == INNER_TRAIN_EXPECTED:
        PASS(f"{setting}: inner train = 685")
    else:
        FAIL(f"{setting}: inner train != 685")

    if len(inner_val) == INNER_VAL_EXPECTED:
        PASS(f"{setting}: inner val = 171")
    else:
        FAIL(f"{setting}: inner val != 171")

    if inner_train & inner_val:
        FAIL(f"{setting}: inner train/val drug overlap")
    else:
        PASS(f"{setting}: inner train/val drug-disjoint")

    if inner_train | inner_val == original_train_nodes:
        PASS(
            f"{setting}: inner train+val exactly reconstruct "
            "ORIGINAL setting-specific outer train"
        )
    else:
        FAIL(
            f"{setting}: inner partition does NOT reconstruct "
            "original outer train"
        )

    if stored_outer_train == original_train_nodes:
        PASS(f"{setting}: stored outer train matches original")
    else:
        FAIL(f"{setting}: stored outer train mismatch")

    if stored_outer_heldout == original_heldout:
        PASS(f"{setting}: stored outer heldout matches original")
    else:
        FAIL(f"{setting}: stored outer heldout mismatch")

    if (inner_train | inner_val) & original_heldout:
        FAIL(
            f"{setting}: final outer-heldout drugs entered "
            "inner development partition"
        )
    else:
        PASS(
            f"{setting}: final outer-heldout drugs excluded "
            "from inner development partition"
        )

    # -------------------------------------------------------------------------
    # Load V2 pair tensors
    # -------------------------------------------------------------------------

    v2 = {
        name: load_pair(root / f"{name}.pt")
        for name in NAMES
    }

    dev_data[setting] = v2

    print()
    print("V2 pair counts")

    for name in NAMES:
        print(f"  {name:10s}: {len(v2[name])}")
        audit_pair_tensor(f"{setting}/{name}", v2[name])

    # -------------------------------------------------------------------------
    # Balance
    # -------------------------------------------------------------------------

    for phase in ["train", "val", "test"]:

        p = len(v2[f"{phase}_pos"])
        n = len(v2[f"{phase}_neg"])

        if p == n:
            PASS(f"{setting}/{phase}: balanced {p}:{n}")
        else:
            FAIL(f"{setting}/{phase}: imbalance {p}:{n}")

    # -------------------------------------------------------------------------
    # Test immutability
    # -------------------------------------------------------------------------

    for name in ["test_pos", "test_neg"]:

        if torch.equal(v2[name], original[name]):
            PASS(f"{setting}/{name}: original final test unchanged")
        else:
            FAIL(f"{setting}/{name}: FINAL TEST CHANGED")

        if tensor_hash(v2[name]) == tensor_hash(original[name]):
            PASS(f"{setting}/{name}: SHA256 preserved")
        else:
            FAIL(f"{setting}/{name}: SHA256 mismatch")

    # -------------------------------------------------------------------------
    # Pair overlap
    # -------------------------------------------------------------------------

    train_pairs = pair_set(
        torch.cat(
            [v2["train_pos"], v2["train_neg"]],
            dim=0,
        )
    )

    val_pairs = pair_set(
        torch.cat(
            [v2["val_pos"], v2["val_neg"]],
            dim=0,
        )
    )

    test_pairs = pair_set(
        torch.cat(
            [v2["test_pos"], v2["test_neg"]],
            dim=0,
        )
    )

    overlaps = {
        "train/val": train_pairs & val_pairs,
        "train/test": train_pairs & test_pairs,
        "val/test": val_pairs & test_pairs,
    }

    print()
    print("Pair overlaps")

    for label, overlap in overlaps.items():
        print(f"  {label:10s}: {len(overlap)}")

        if overlap:
            FAIL(f"{setting}: {label} pair overlap")
        else:
            PASS(f"{setting}: {label} pair overlap = 0")

    # -------------------------------------------------------------------------
    # Label contradictions
    # -------------------------------------------------------------------------

    positives = (
        pair_set(v2["train_pos"])
        | pair_set(v2["val_pos"])
        | pair_set(v2["test_pos"])
    )

    negatives = (
        pair_set(v2["train_neg"])
        | pair_set(v2["val_neg"])
        | pair_set(v2["test_neg"])
    )

    contradictions = positives & negatives

    if contradictions:
        FAIL(
            f"{setting}: {len(contradictions)} "
            "positive/negative contradictions"
        )
    else:
        PASS(f"{setting}: no label contradictions")

    # -------------------------------------------------------------------------
    # Development node exposure
    # -------------------------------------------------------------------------

    train_nodes = node_set(
        v2["train_pos"],
        v2["train_neg"],
    )

    val_nodes = node_set(
        v2["val_pos"],
        v2["val_neg"],
    )

    dev_nodes = train_nodes | val_nodes

    print()
    print("Drug exposure")
    print("  train drugs                    :", len(train_nodes))
    print("  validation drugs               :", len(val_nodes))
    print(
        "  train ∩ inner-val             :",
        len(train_nodes & inner_val),
    )
    print(
        "  development ∩ outer-heldout   :",
        len(dev_nodes & original_heldout),
    )

    if train_nodes & inner_val:
        FAIL(f"{setting}: inner-val drug exposed during training")
    else:
        PASS(f"{setting}: zero inner-val drug exposure in training")

    if dev_nodes & original_heldout:
        FAIL(f"{setting}: final-test drug exposed during development")
    else:
        PASS(f"{setting}: zero final-test drug exposure during development")

    if train_nodes <= inner_train:
        PASS(f"{setting}: training endpoints restricted to inner train")
    else:
        FAIL(f"{setting}: training endpoints outside inner train")

    # -------------------------------------------------------------------------
    # Setting semantics
    # -------------------------------------------------------------------------

    print()
    print("Cold-start semantics")

    if setting == "cold_start_S1":

        for name in ["val_pos", "val_neg"]:

            d = membership_distribution(
                v2[name],
                inner_val,
            )

            print(
                f"  {name} inner-val drugs/pair:",
                d,
            )

            if (
                d[0] == 0
                and d[2] == 0
                and d[1] == len(v2[name])
            ):
                PASS(
                    f"S1/{name}: exactly one unseen "
                    "inner-validation drug per pair"
                )
            else:
                FAIL(f"S1/{name}: invalid S1 validation semantics")

        for name in ["test_pos", "test_neg"]:

            d = membership_distribution(
                v2[name],
                original_heldout,
            )

            print(
                f"  {name} outer-heldout drugs/pair:",
                d,
            )

            if (
                d[0] == 0
                and d[2] == 0
                and d[1] == len(v2[name])
            ):
                PASS(
                    f"S1/{name}: exactly one original "
                    "outer-heldout drug per pair"
                )
            else:
                FAIL(f"S1/{name}: invalid FINAL S1 semantics")

    else:

        for name in ["val_pos", "val_neg"]:

            d = membership_distribution(
                v2[name],
                inner_val,
            )

            print(
                f"  {name} inner-val drugs/pair:",
                d,
            )

            if (
                d[0] == 0
                and d[1] == 0
                and d[2] == len(v2[name])
            ):
                PASS(
                    f"S2/{name}: both drugs unseen "
                    "during inner validation"
                )
            else:
                FAIL(f"S2/{name}: invalid S2 validation semantics")

        for name in ["test_pos", "test_neg"]:

            d = membership_distribution(
                v2[name],
                original_heldout,
            )

            print(
                f"  {name} outer-heldout drugs/pair:",
                d,
            )

            if (
                d[0] == 0
                and d[1] == 0
                and d[2] == len(v2[name])
            ):
                PASS(
                    f"S2/{name}: both drugs belong to "
                    "original outer-heldout set"
                )
            else:
                FAIL(f"S2/{name}: invalid FINAL S2 semantics")


# =============================================================================
# 3. CROSS-SETTING CHECK
# =============================================================================

print()
print("=" * 105)
print("CROSS-SETTING OUTER-PARTITION AUDIT")
print("=" * 105)

s1_train = outer_partitions["cold_start_S1"]["train"]
s2_train = outer_partitions["cold_start_S2"]["train"]

s1_hold = outer_partitions["cold_start_S1"]["heldout"]
s2_hold = outer_partitions["cold_start_S2"]["heldout"]

print("S1 train ∩ S2 train     :", len(s1_train & s2_train))
print("S1 heldout ∩ S2 heldout :", len(s1_hold & s2_hold))
print("S1-only heldout         :", len(s1_hold - s2_hold))
print("S2-only heldout         :", len(s2_hold - s1_hold))

if s1_train == s2_train:
    FAIL(
        "S1/S2 outer training partitions unexpectedly identical; "
        "expected setting-specific partitions"
    )
else:
    PASS("S1/S2 outer training partitions correctly differ")

if s1_hold == s2_hold:
    FAIL(
        "S1/S2 outer heldout partitions unexpectedly identical"
    )
else:
    PASS("S1/S2 outer heldout partitions correctly differ")

if (
    len(s1_train & s2_train) == 685
    and len(s1_hold & s2_hold) == 44
    and len(s1_hold - s2_hold) == 171
    and len(s2_hold - s1_hold) == 171
):
    PASS("cross-setting outer partition fingerprint matches diagnosis")
else:
    FAIL("cross-setting outer partition counts changed unexpectedly")


# =============================================================================
# 4. COMMON-1087 MAPPING
# =============================================================================

print()
print("=" * 105)
print("COMMON-1087 MAPPING AUDIT")
print("=" * 105)

try:
    import pandas as pd

    mapping = pd.read_csv(MAPPING_PATH)

    if len(mapping) == NUM_NODES:
        PASS("mapping contains exactly 1087 rows")
    else:
        FAIL(f"mapping rows = {len(mapping)}")

    ids = mapping["node_id"].astype(int).tolist()

    if ids == list(range(NUM_NODES)):
        PASS("mapping node IDs exactly 0..1086")
    else:
        FAIL("mapping node IDs not exactly 0..1086")

    if mapping["drug_id"].duplicated().any():
        FAIL("mapping contains duplicate drug IDs")
    else:
        PASS("mapping drug IDs unique")

except Exception as e:
    FAIL(f"mapping audit exception: {e}")


# =============================================================================
# 5. MORGAN INTRINSIC FEATURE AUDIT
# =============================================================================

print()
print("=" * 105)
print("MORGAN FEATURE AUDIT")
print("=" * 105)

morgan = {}

for setting in [
    "transductive",
    "cold_start_S1",
    "cold_start_S2",
]:

    path = (
        FEATURE_ROOT
        / f"{setting}_morgan_2048.pt"
    )

    try:
        x = torch.load(path, map_location="cpu")
        x = x.detach().cpu().float().contiguous()

        morgan[setting] = x

        if tuple(x.shape) == (1087, 2048):
            PASS(f"{setting}: Morgan shape (1087,2048)")
        else:
            FAIL(f"{setting}: Morgan shape {tuple(x.shape)}")

        if torch.isfinite(x).all():
            PASS(f"{setting}: Morgan finite")
        else:
            FAIL(f"{setting}: Morgan contains NaN/Inf")

    except Exception as e:
        FAIL(f"{setting}: Morgan load error: {e}")

if len(morgan) == 3:

    if (
        torch.equal(morgan["transductive"], morgan["cold_start_S1"])
        and
        torch.equal(morgan["transductive"], morgan["cold_start_S2"])
    ):
        PASS("Morgan intrinsic features identical across settings")
    else:
        FAIL("Morgan features differ across settings")


# =============================================================================
# 6. MOLECULAR GRAPH AUDIT
# =============================================================================

print()
print("=" * 105)
print("MOLECULAR GRAPH AUDIT")
print("=" * 105)

try:
    try:
        graphs = torch.load(
            STRUCTURE_PATH,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        graphs = torch.load(
            STRUCTURE_PATH,
            map_location="cpu",
        )

    if len(graphs) == NUM_NODES:
        PASS("molecular graph bank = 1087")
    else:
        FAIL(f"molecular graph bank = {len(graphs)}")

    errors = []

    for idx, g in enumerate(graphs):

        if not hasattr(g, "x"):
            errors.append(f"{idx}: missing x")
            continue

        if not hasattr(g, "edge_index"):
            errors.append(f"{idx}: missing edge_index")
            continue

        if not hasattr(g, "edge_attr"):
            errors.append(f"{idx}: missing edge_attr")
            continue

        if g.x.ndim != 2 or g.x.shape[1] != 36:
            errors.append(
                f"{idx}: atom shape {tuple(g.x.shape)}"
            )

        if g.edge_attr.ndim != 2 or g.edge_attr.shape[1] != 12:
            errors.append(
                f"{idx}: bond shape {tuple(g.edge_attr.shape)}"
            )

        if hasattr(g, "node_id"):
            try:
                if int(g.node_id) != idx:
                    errors.append(
                        f"{idx}: node_id={int(g.node_id)}"
                    )
            except Exception:
                errors.append(f"{idx}: unreadable node_id")

    if errors:
        FAIL(f"{len(errors)} molecular graph schema/alignment errors")

        for e in errors[:10]:
            print("  ", e)
    else:
        PASS("all 1087 molecular graphs schema/alignment valid")

except Exception as e:
    FAIL(f"molecular graph audit exception: {e}")


# =============================================================================
# 7. MANIFEST SELECTION POLICY
# =============================================================================

print()
print("=" * 105)
print("MODEL-SELECTION POLICY")
print("=" * 105)

policy = manifest.get("selection_policy", {})

if policy.get("outer_test_locked") is True:
    PASS("outer test marked LOCKED")
else:
    FAIL("outer test is not marked locked")

if policy.get("outer_test_for_selection") is False:
    PASS("outer test forbidden for model selection")
else:
    FAIL("outer test selection policy invalid")

if policy.get("architecture_selection") == \
        "setting-specific inner validation only":
    PASS("architecture selection = setting-specific inner validation")
else:
    FAIL("architecture-selection policy mismatch")


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 105)
print("MASTER CERTIFICATION SUMMARY")
print("=" * 105)

print("PASS checks :", len(passes))
print("Warnings    :", len(warnings))
print("Failures    :", len(failures))

if warnings:
    print("\nWARNINGS")
    for w in warnings:
        print(" -", w)

if failures:
    print("\nFAILURES")
    for f in failures:
        print(" -", f)

    print()
    print("=" * 105)
    print("PRE-TRAINING CERTIFICATION V2: FAILED")
    print("DO NOT TRAIN")
    print("=" * 105)

    sys.exit(1)

print()
print("=" * 105)
print("PRE-TRAINING CERTIFICATION V2: PASSED")
print("SAFE FOR CHEMICAL-ONLY SEED-42 DEVELOPMENT")
print("=" * 105)

print()
print("Certified:")
print("  1087 common-drug index")
print("  1071 DDI-active drugs")
print("  setting-specific original S1/S2 outer partitions")
print("  856 outer-development drugs per setting")
print("  215 untouched final-heldout drugs per setting")
print("  685 inner-training drugs per setting")
print("  171 inner-validation drugs per setting")
print("  strict S1 inner-validation semantics")
print("  strict S2 inner-validation semantics")
print("  unchanged original outer test tensors")
print("  zero outer-test drug exposure during development")
print("  zero train/val/test pair overlap")
print("  zero positive/negative contradictions")
print("  aligned Morgan 2048 features")
print("  aligned molecular GINE graph bank")
print()
print(
    "NOTE: certification is for Morgan/structure chemical development. "
    "DTI/PPI/full-AM-DDI requires an additional biological leakage audit."
)
