from pathlib import Path
import hashlib
import json
import random
import torch


SEED = 42
NUM_NODES = 1087
INNER_VAL_FRACTION = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTER_ROOT = PROJECT_ROOT / "data" / "amddi_common_splits"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_coldstart_dev_splits_v2"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================================================
# HELPERS
# ======================================================================================

def load_tensor(path):
    x = torch.load(path, map_location="cpu")

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{path} is not a tensor")

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(
            f"{path}: expected [N,2], got {tuple(x.shape)}"
        )

    return x.long().contiguous()


def canonical_pairs(x):
    a = torch.minimum(x[:, 0], x[:, 1])
    b = torch.maximum(x[:, 0], x[:, 1])

    return torch.stack([a, b], dim=1)


def pair_set(x):
    x = canonical_pairs(x)

    return {
        (int(a), int(b))
        for a, b in x.tolist()
    }


def node_set(*xs):
    z = torch.cat(xs, dim=0)

    return set(
        int(v)
        for v in z.reshape(-1).tolist()
    )


def tensor_from_pairs(pairs):
    pairs = sorted(pairs)

    if len(pairs) == 0:
        return torch.empty((0, 2), dtype=torch.long)

    return torch.tensor(
        pairs,
        dtype=torch.long,
    )


def sha256_tensor(x):
    x = x.detach().cpu().contiguous()

    h = hashlib.sha256()
    h.update(x.numpy().tobytes())

    return h.hexdigest()


def save_tensor(x, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    torch.save(
        x.detach().cpu(),
        tmp,
    )

    tmp.replace(path)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )

    tmp.replace(path)


def generate_negatives(
    pool_a,
    pool_b,
    n_required,
    known_positive_pairs,
    used_negative_pairs,
    rng,
    same_pool,
):
    A = sorted(pool_a)
    B = sorted(pool_b)

    candidates = []

    if same_pool:

        for i in range(len(A)):
            for j in range(i + 1, len(A)):

                a = A[i]
                b = A[j]

                p = (
                    min(a, b),
                    max(a, b),
                )

                if p in known_positive_pairs:
                    continue

                if p in used_negative_pairs:
                    continue

                candidates.append(p)

    else:

        for a in A:
            for b in B:

                if a == b:
                    continue

                p = (
                    min(a, b),
                    max(a, b),
                )

                if p in known_positive_pairs:
                    continue

                if p in used_negative_pairs:
                    continue

                candidates.append(p)

    if len(candidates) < n_required:
        raise RuntimeError(
            f"Insufficient negatives: "
            f"need={n_required}, "
            f"available={len(candidates)}"
        )

    rng.shuffle(candidates)

    selected = candidates[:n_required]

    used_negative_pairs.update(selected)

    return tensor_from_pairs(selected)


# ======================================================================================
# LOAD ORIGINAL SPLITS
# ======================================================================================

print("=" * 100)
print("AM-DDI COMMON-1087")
print("STRICT SETTING-SPECIFIC COLD-START DEVELOPMENT SPLITS V2")
print("=" * 100)

settings = [
    "cold_start_S1",
    "cold_start_S2",
]

loaded = {}

for setting in settings:

    loaded[setting] = {}

    for name in [
        "train_pos",
        "train_neg",
        "val_pos",
        "val_neg",
        "test_pos",
        "test_neg",
    ]:

        loaded[setting][name] = load_tensor(
            OUTER_ROOT
            / setting
            / f"{name}.pt"
        )


# ======================================================================================
# GLOBAL KNOWN POSITIVE DDI SET
# ======================================================================================

known_positive_pairs = set()

for setting in settings:

    for name in [
        "train_pos",
        "val_pos",
        "test_pos",
    ]:

        known_positive_pairs.update(
            pair_set(
                loaded[setting][name]
            )
        )

# Also include transductive positives for maximum protection.

for name in [
    "train_pos",
    "val_pos",
    "test_pos",
]:

    path = (
        OUTER_ROOT
        / "transductive"
        / f"{name}.pt"
    )

    if path.exists():
        known_positive_pairs.update(
            pair_set(
                load_tensor(path)
            )
        )

print(
    "Known positive DDI pairs:",
    len(known_positive_pairs),
)


# ======================================================================================
# MANIFEST
# ======================================================================================

manifest = {
    "protocol_version":
        "AMDDI_common1087_strict_coldstart_dev_v2",

    "seed": SEED,

    "num_nodes": NUM_NODES,

    "settings": {},

    "selection_policy": {
        "architecture_selection":
            "setting-specific inner validation only",

        "outer_test_for_selection":
            False,

        "outer_test_locked":
            True,
    },
}


# ======================================================================================
# BUILD EACH SETTING INDEPENDENTLY
# ======================================================================================

for setting_idx, setting in enumerate(settings):

    print()
    print("=" * 100)
    print(setting)
    print("=" * 100)

    s = loaded[setting]

    # ------------------------------------------------------------------
    # ORIGINAL OUTER TRAIN DRUGS
    # ------------------------------------------------------------------

    outer_train_nodes = node_set(
        s["train_pos"],
        s["train_neg"],
    )

    if len(outer_train_nodes) != 856:
        raise RuntimeError(
            f"{setting}: expected 856 outer-train drugs, "
            f"found {len(outer_train_nodes)}"
        )

    # ------------------------------------------------------------------
    # ORIGINAL OUTER HELDOUT DRUGS
    # ------------------------------------------------------------------

    test_nodes = node_set(
        s["test_pos"],
        s["test_neg"],
    )

    outer_heldout_nodes = (
        test_nodes
        - outer_train_nodes
    )

    if len(outer_heldout_nodes) != 215:
        raise RuntimeError(
            f"{setting}: expected 215 outer-heldout drugs, "
            f"found {len(outer_heldout_nodes)}"
        )

    if outer_train_nodes & outer_heldout_nodes:
        raise RuntimeError(
            f"{setting}: outer train/heldout overlap"
        )

    print(
        "Outer train drugs   :",
        len(outer_train_nodes),
    )

    print(
        "Outer heldout drugs :",
        len(outer_heldout_nodes),
    )

    print(
        "Outer overlap       :",
        len(
            outer_train_nodes
            & outer_heldout_nodes
        ),
    )

    # ------------------------------------------------------------------
    # SETTING-SPECIFIC INNER DRUG SPLIT
    # ------------------------------------------------------------------

    rng_drugs = random.Random(
        SEED + 1000 * setting_idx
    )

    shuffled = sorted(
        outer_train_nodes
    )

    rng_drugs.shuffle(shuffled)

    n_inner_val = round(
        len(shuffled)
        * INNER_VAL_FRACTION
    )

    inner_val_nodes = set(
        shuffled[:n_inner_val]
    )

    inner_train_nodes = set(
        shuffled[n_inner_val:]
    )

    if len(inner_train_nodes) != 685:
        raise RuntimeError(
            f"{setting}: expected 685 inner-train drugs, "
            f"got {len(inner_train_nodes)}"
        )

    if len(inner_val_nodes) != 171:
        raise RuntimeError(
            f"{setting}: expected 171 inner-val drugs, "
            f"got {len(inner_val_nodes)}"
        )

    if inner_train_nodes & inner_val_nodes:
        raise RuntimeError(
            f"{setting}: inner train/val overlap"
        )

    if (
        inner_train_nodes
        | inner_val_nodes
    ) != outer_train_nodes:
        raise RuntimeError(
            f"{setting}: inner split does not reconstruct "
            "outer train universe"
        )

    print()
    print(
        "Inner train drugs :",
        len(inner_train_nodes),
    )

    print(
        "Inner val drugs   :",
        len(inner_val_nodes),
    )

    print(
        "Inner overlap     :",
        len(
            inner_train_nodes
            & inner_val_nodes
        ),
    )

    # ------------------------------------------------------------------
    # DEVELOPMENT POSITIVE POOL
    # ------------------------------------------------------------------

    dev_positive_pairs = {
        p
        for p in known_positive_pairs
        if (
            p[0] in outer_train_nodes
            and
            p[1] in outer_train_nodes
        )
    }

    seen_seen = []
    seen_val = []
    val_val = []

    for a, b in sorted(dev_positive_pairs):

        if (
            a in inner_train_nodes
            and
            b in inner_train_nodes
        ):

            seen_seen.append((a, b))

        elif (
            (
                a in inner_train_nodes
                and
                b in inner_val_nodes
            )
            or
            (
                b in inner_train_nodes
                and
                a in inner_val_nodes
            )
        ):

            seen_val.append((a, b))

        elif (
            a in inner_val_nodes
            and
            b in inner_val_nodes
        ):

            val_val.append((a, b))

    seen_seen = tensor_from_pairs(
        seen_seen
    )

    seen_val = tensor_from_pairs(
        seen_val
    )

    val_val = tensor_from_pairs(
        val_val
    )

    print()
    print("Development positive categories")
    print(
        "seen-seen       :",
        len(seen_seen),
    )

    print(
        "seen-inner-val  :",
        len(seen_val),
    )

    print(
        "inner-val pair  :",
        len(val_val),
    )

    # ------------------------------------------------------------------
    # SELECT VALIDATION POSITIVES
    # ------------------------------------------------------------------

    train_pos = seen_seen.clone()

    if setting == "cold_start_S1":

        val_pos = seen_val.clone()

    elif setting == "cold_start_S2":

        val_pos = val_val.clone()

    else:
        raise RuntimeError(setting)

    if len(train_pos) == 0:
        raise RuntimeError(
            f"{setting}: empty train positives"
        )

    if len(val_pos) == 0:
        raise RuntimeError(
            f"{setting}: empty validation positives"
        )

    # ------------------------------------------------------------------
    # NEGATIVE SAMPLING
    # ------------------------------------------------------------------

    used_negatives = set()

    train_neg = generate_negatives(
        pool_a=inner_train_nodes,
        pool_b=inner_train_nodes,
        n_required=len(train_pos),
        known_positive_pairs=known_positive_pairs,
        used_negative_pairs=used_negatives,
        rng=random.Random(
            SEED + 10000 + setting_idx
        ),
        same_pool=True,
    )

    if setting == "cold_start_S1":

        val_neg = generate_negatives(
            pool_a=inner_train_nodes,
            pool_b=inner_val_nodes,
            n_required=len(val_pos),
            known_positive_pairs=known_positive_pairs,
            used_negative_pairs=used_negatives,
            rng=random.Random(
                SEED + 20000 + setting_idx
            ),
            same_pool=False,
        )

    else:

        val_neg = generate_negatives(
            pool_a=inner_val_nodes,
            pool_b=inner_val_nodes,
            n_required=len(val_pos),
            known_positive_pairs=known_positive_pairs,
            used_negative_pairs=used_negatives,
            rng=random.Random(
                SEED + 30000 + setting_idx
            ),
            same_pool=True,
        )

    # ------------------------------------------------------------------
    # ORIGINAL FINAL TEST — UNCHANGED
    # ------------------------------------------------------------------

    test_pos = (
        s["test_pos"]
        .clone()
    )

    test_neg = (
        s["test_neg"]
        .clone()
    )

    # ------------------------------------------------------------------
    # SAVE SETTING
    # ------------------------------------------------------------------

    setting_root = (
        OUTPUT_ROOT
        / setting
    )

    setting_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    tensors = {
        "train_pos": train_pos,
        "train_neg": train_neg,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
    }

    for name, x in tensors.items():

        save_tensor(
            x,
            setting_root
            / f"{name}.pt",
        )

    save_tensor(
        torch.tensor(
            sorted(inner_train_nodes),
            dtype=torch.long,
        ),
        setting_root
        / "inner_train_drugs.pt",
    )

    save_tensor(
        torch.tensor(
            sorted(inner_val_nodes),
            dtype=torch.long,
        ),
        setting_root
        / "inner_val_drugs.pt",
    )

    save_tensor(
        torch.tensor(
            sorted(outer_heldout_nodes),
            dtype=torch.long,
        ),
        setting_root
        / "outer_heldout_drugs.pt",
    )

    save_tensor(
        torch.tensor(
            sorted(outer_train_nodes),
            dtype=torch.long,
        ),
        setting_root
        / "outer_train_drugs.pt",
    )

    # ------------------------------------------------------------------
    # IMMEDIATE SEMANTIC ASSERTIONS
    # ------------------------------------------------------------------

    # Development must never include original heldout drugs.

    development_nodes = node_set(
        train_pos,
        train_neg,
        val_pos,
        val_neg,
    )

    if development_nodes & outer_heldout_nodes:
        raise RuntimeError(
            f"{setting}: FINAL OUTER HELDOUT DRUG "
            "EXPOSED DURING DEVELOPMENT"
        )

    # S1 validation = exactly one inner-val drug.

    if setting == "cold_start_S1":

        for name, x in [
            ("val_pos", val_pos),
            ("val_neg", val_neg),
        ]:

            for a, b in x.tolist():

                n = (
                    int(a in inner_val_nodes)
                    +
                    int(b in inner_val_nodes)
                )

                if n != 1:
                    raise RuntimeError(
                        f"{setting}/{name}: "
                        f"expected exactly one inner-val drug"
                    )

        # Original S1 final test = exactly one OUTER heldout drug.

        for name, x in [
            ("test_pos", test_pos),
            ("test_neg", test_neg),
        ]:

            for a, b in x.tolist():

                n = (
                    int(a in outer_heldout_nodes)
                    +
                    int(b in outer_heldout_nodes)
                )

                if n != 1:
                    raise RuntimeError(
                        f"{setting}/{name}: "
                        "original S1 test semantics broken"
                    )

    # S2 validation/test = two heldout drugs.

    if setting == "cold_start_S2":

        for name, x in [
            ("val_pos", val_pos),
            ("val_neg", val_neg),
        ]:

            for a, b in x.tolist():

                if (
                    a not in inner_val_nodes
                    or
                    b not in inner_val_nodes
                ):

                    raise RuntimeError(
                        f"{setting}/{name}: "
                        "validation is not strict S2"
                    )

        for name, x in [
            ("test_pos", test_pos),
            ("test_neg", test_neg),
        ]:

            for a, b in x.tolist():

                if (
                    a not in outer_heldout_nodes
                    or
                    b not in outer_heldout_nodes
                ):

                    raise RuntimeError(
                        f"{setting}/{name}: "
                        "original S2 test semantics broken"
                    )

    # ------------------------------------------------------------------
    # MANIFEST
    # ------------------------------------------------------------------

    manifest["settings"][setting] = {

        "outer_train_drugs":
            len(outer_train_nodes),

        "outer_heldout_drugs":
            len(outer_heldout_nodes),

        "inner_train_drugs":
            len(inner_train_nodes),

        "inner_val_drugs":
            len(inner_val_nodes),

        "train_pos":
            len(train_pos),

        "train_neg":
            len(train_neg),

        "val_pos":
            len(val_pos),

        "val_neg":
            len(val_neg),

        "test_pos":
            len(test_pos),

        "test_neg":
            len(test_neg),

        "test_fingerprints": {
            "test_pos":
                sha256_tensor(test_pos),

            "test_neg":
                sha256_tensor(test_neg),
        },

        "outer_heldout_fingerprint":
            sha256_tensor(
                torch.tensor(
                    sorted(outer_heldout_nodes),
                    dtype=torch.long,
                )
            ),

        "inner_train_fingerprint":
            sha256_tensor(
                torch.tensor(
                    sorted(inner_train_nodes),
                    dtype=torch.long,
                )
            ),

        "inner_val_fingerprint":
            sha256_tensor(
                torch.tensor(
                    sorted(inner_val_nodes),
                    dtype=torch.long,
                )
            ),
    }

    print()
    print("Final development counts")
    print(
        "train_pos:",
        len(train_pos),
    )

    print(
        "train_neg:",
        len(train_neg),
    )

    print(
        "val_pos  :",
        len(val_pos),
    )

    print(
        "val_neg  :",
        len(val_neg),
    )

    print(
        "test_pos :",
        len(test_pos),
    )

    print(
        "test_neg :",
        len(test_neg),
    )


# ======================================================================================
# CROSS-SETTING INFORMATION ONLY
# ======================================================================================

s1_heldout = set(
    torch.load(
        OUTPUT_ROOT
        / "cold_start_S1"
        / "outer_heldout_drugs.pt",
        map_location="cpu",
    ).tolist()
)

s2_heldout = set(
    torch.load(
        OUTPUT_ROOT
        / "cold_start_S2"
        / "outer_heldout_drugs.pt",
        map_location="cpu",
    ).tolist()
)

manifest["cross_setting"] = {
    "S1_S2_outer_heldout_overlap":
        len(
            s1_heldout
            & s2_heldout
        ),

    "S1_only_outer_heldout":
        len(
            s1_heldout
            - s2_heldout
        ),

    "S2_only_outer_heldout":
        len(
            s2_heldout
            - s1_heldout
        ),
}


save_json(
    manifest,
    OUTPUT_ROOT / "manifest.json",
)


# ======================================================================================
# FINISH
# ======================================================================================

print()
print("=" * 100)
print("STRICT SETTING-SPECIFIC DEVELOPMENT SPLITS V2 CREATED")
print("=" * 100)

print(
    "Output:",
    OUTPUT_ROOT,
)

print()
print(
    "PASS: setting-specific S1/S2 outer partitions preserved."
)

print(
    "PASS: original outer test tensors preserved."
)

print(
    "PASS: development excludes outer-heldout drugs."
)

print()
print(
    "DO NOT TRAIN YET."
)

print(
    "Run Script 109 v2 master certification next."
)
