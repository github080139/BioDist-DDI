#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import torch
from sklearn.model_selection import train_test_split


ROOT = Path(".")
SPLIT_ROOT = ROOT / "data/amddi_common_1087_coldstart_dev_splits_v2"
BIO_ROOT = ROOT / "data/amddi_common_1087_biological_dev_v4"
FEAT_ROOT = ROOT / "data/amddi_common_1087_leakage_free_features"
TEACHER_ROOT = ROOT / "checkpoints/amddi_common_1087_v9d_teacher"

SEEDS = [42, 123, 456, 789, 2026]
SPLITS = ["cold_start_S1", "cold_start_S2"]

TRAINER = {
    "cold_start_S1":
        ROOT / "scripts/160_train_amddi_common_1087_v9d_s1_teacher_distillation.py",
    "cold_start_S2":
        ROOT / "scripts/158_train_amddi_common_1087_v9d_teacher_distillation.py",
}

TEACHER_SCRIPT = {
    "cold_start_S1":
        ROOT / "scripts/159_pretrain_amddi_common_1087_v9d_s1_bio_teacher.py",
    "cold_start_S2":
        ROOT / "scripts/157_pretrain_amddi_common_1087_v9d_bio_teacher.py",
}


passes = 0
fails = 0


def check(cond, name, detail=""):
    global passes, fails
    if cond:
        passes += 1
        print(f"[PASS] {name}" + (f" :: {detail}" if detail else ""))
    else:
        fails += 1
        print(f"[FAIL] {name}" + (f" :: {detail}" if detail else ""))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_drugs(path):
    x = torch.load(path, map_location="cpu", weights_only=True)
    return set(int(v) for v in x.reshape(-1).tolist())


def normalize_edges(x):
    """
    Accept [2,N] or [N,2] pair tensors.
    Returns list[(int,int)].
    """
    x = torch.as_tensor(x).cpu()

    if x.ndim != 2:
        raise RuntimeError(f"Unexpected pair tensor shape {tuple(x.shape)}")

    if x.shape[0] == 2:
        x = x.t()
    elif x.shape[1] != 2:
        raise RuntimeError(f"Unexpected pair tensor shape {tuple(x.shape)}")

    return [(int(a), int(b)) for a, b in x.tolist()]


def unordered(pair):
    a, b = pair
    return (a, b) if a <= b else (b, a)


def nonzero_rows(x):
    x = torch.as_tensor(x).cpu()
    mask = torch.any(x != 0, dim=1)
    return set(torch.where(mask)[0].tolist())


def static_forbidden_pair_audit(path):
    """
    Static source audit only.
    Does NOT open any test pair tensor.
    """
    text = path.read_text()

    executable = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if re.search(r'test_(pos|neg)\.pt', stripped):
            executable.append((lineno, stripped))

    return executable


print("=" * 100)
print("AM-DDI V9-D PROVENANCE / LEAKAGE CERTIFICATION")
print("=" * 100)
print("IMPORTANT: test_pos.pt and test_neg.pt are NEVER loaded by this audit.")
print()


for split in SPLITS:

    print()
    print("=" * 100)
    print(split)
    print("=" * 100)

    sdir = SPLIT_ROOT / split
    bdir = BIO_ROOT / split

    # ------------------------------------------------------------------
    # Drug partitions
    # ------------------------------------------------------------------

    inner_train = load_drugs(sdir / "inner_train_drugs.pt")
    inner_val = load_drugs(sdir / "inner_val_drugs.pt")
    outer = load_drugs(sdir / "outer_heldout_drugs.pt")
    outer_train = load_drugs(sdir / "outer_train_drugs.pt")

    check(
        len(inner_train) == 685,
        f"{split}: inner_train count",
        str(len(inner_train)),
    )

    check(
        len(inner_val) == 171,
        f"{split}: inner_val count",
        str(len(inner_val)),
    )

    check(
        len(outer) == 215,
        f"{split}: outer-heldout count",
        str(len(outer)),
    )

    check(
        inner_train.isdisjoint(inner_val),
        f"{split}: train/val drug disjointness",
    )

    check(
        inner_train.isdisjoint(outer),
        f"{split}: train/outer drug disjointness",
    )

    check(
        inner_val.isdisjoint(outer),
        f"{split}: val/outer drug disjointness",
    )

    check(
        inner_train | inner_val == outer_train,
        f"{split}: inner train+val exactly outer_train",
        f"{len(outer_train)} drugs",
    )

    # ------------------------------------------------------------------
    # Biological v4 partition consistency
    # ------------------------------------------------------------------

    bio_train = load_drugs(bdir / "inner_train_drugs.pt")
    bio_val = load_drugs(bdir / "inner_val_drugs.pt")
    bio_outer = load_drugs(bdir / "outer_heldout_drugs.pt")

    check(
        bio_train == inner_train,
        f"{split}: biological-v4 inner_train identity",
    )

    check(
        bio_val == inner_val,
        f"{split}: biological-v4 inner_val identity",
    )

    check(
        bio_outer == outer,
        f"{split}: biological-v4 outer identity",
    )

    # ------------------------------------------------------------------
    # RAW DTI exposure
    # IMPORTANT:
    # row 0 = AM-DDI drug index
    # row 1 = ESM-aligned protein index
    # ------------------------------------------------------------------

    dti = torch.load(
        bdir / "dti_edge_index.pt",
        map_location="cpu",
        weights_only=True,
    )

    check(
        dti.ndim == 2 and dti.shape[0] == 2,
        f"{split}: raw DTI shape",
        str(tuple(dti.shape)),
    )

    dti_drugs = set(int(v) for v in dti[0].tolist())

    expected_edges = 4805 if split == "cold_start_S1" else 4695

    check(
        dti.shape[1] == expected_edges,
        f"{split}: raw DTI edge count",
        str(dti.shape[1]),
    )

    check(
        dti_drugs == inner_train,
        f"{split}: raw DTI drug endpoints exactly inner_train",
        f"{len(dti_drugs)} exposed drugs",
    )

    check(
        dti_drugs.isdisjoint(inner_val),
        f"{split}: zero inner-val DTI exposure",
    )

    check(
        dti_drugs.isdisjoint(outer),
        f"{split}: zero outer-heldout DTI exposure",
    )

    # ------------------------------------------------------------------
    # Context exposure
    # ------------------------------------------------------------------

    base_path = (
        FEAT_ROOT /
        f"{split}_v9d_clean_dti_esm2_context_128.pt"
    )

    ppi_path = (
        FEAT_ROOT /
        f"{split}_ppi_esm2_context_128.pt"
    )

    base = torch.load(base_path, map_location="cpu", weights_only=True)
    ppi = torch.load(ppi_path, map_location="cpu", weights_only=True)

    check(
        tuple(base.shape) == (1087, 128),
        f"{split}: clean DTI-ESM2 context shape",
        str(tuple(base.shape)),
    )

    check(
        tuple(ppi.shape) == (1087, 128),
        f"{split}: PPI-ESM2 context shape",
        str(tuple(ppi.shape)),
    )

    base_nz = nonzero_rows(base)
    ppi_nz = nonzero_rows(ppi)

    check(
        base_nz == inner_train,
        f"{split}: clean DTI context exposure exactly inner_train",
        f"{len(base_nz)} nonzero rows",
    )

    check(
        ppi_nz == inner_train,
        f"{split}: PPI context exposure exactly inner_train",
        f"{len(ppi_nz)} nonzero rows",
    )

    check(
        base_nz.isdisjoint(inner_val | outer),
        f"{split}: clean context zero on val+outer",
    )

    check(
        ppi_nz.isdisjoint(inner_val | outer),
        f"{split}: PPI context zero on val+outer",
    )

    # ------------------------------------------------------------------
    # Official inner-training DDI pair integrity
    #
    # Only train_pos/train_neg are loaded.
    # val tensors are not required for teacher provenance.
    # test tensors are NEVER loaded.
    # ------------------------------------------------------------------

    train_pos = normalize_edges(
        torch.load(
            sdir / "train_pos.pt",
            map_location="cpu",
            weights_only=True,
        )
    )

    train_neg = normalize_edges(
        torch.load(
            sdir / "train_neg.pt",
            map_location="cpu",
            weights_only=True,
        )
    )

    expected_each = 14494 if split == "cold_start_S1" else 16903

    check(
        len(train_pos) == expected_each,
        f"{split}: official train positive count",
        str(len(train_pos)),
    )

    check(
        len(train_neg) == expected_each,
        f"{split}: official train negative count",
        str(len(train_neg)),
    )

    pos_u = [unordered(p) for p in train_pos]
    neg_u = [unordered(p) for p in train_neg]

    pos_set = set(pos_u)
    neg_set = set(neg_u)

    check(
        len(pos_set) == len(pos_u),
        f"{split}: no duplicate/reversed positive train pairs",
    )

    check(
        len(neg_set) == len(neg_u),
        f"{split}: no duplicate/reversed negative train pairs",
    )

    check(
        pos_set.isdisjoint(neg_set),
        f"{split}: no contradictory positive/negative labels",
    )

    endpoints = set()
    for a, b in train_pos + train_neg:
        endpoints.add(a)
        endpoints.add(b)

    check(
        endpoints.issubset(inner_train),
        f"{split}: all teacher DDI endpoints inside inner_train",
        f"{len(endpoints)} endpoint drugs",
    )

    # ------------------------------------------------------------------
    # Reproduce teacher fit/monitor split for each seed.
    #
    # Teacher protocol:
    # deterministic class-stratified 90/10 split of official
    # inner-training pairs.
    # ------------------------------------------------------------------

    pairs = np.array(train_pos + train_neg, dtype=np.int64)
    labels = np.array(
        [1] * len(train_pos) + [0] * len(train_neg),
        dtype=np.int64,
    )

    for seed in SEEDS:

        indices = np.arange(len(labels))

        fit_idx, monitor_idx = train_test_split(
            indices,
            test_size=0.10,
            random_state=seed,
            stratify=labels,
        )

        fit_pairs = {
            unordered(tuple(pairs[i]))
            for i in fit_idx
        }

        monitor_pairs = {
            unordered(tuple(pairs[i]))
            for i in monitor_idx
        }

        check(
            fit_pairs.isdisjoint(monitor_pairs),
            f"{split} seed {seed}: teacher fit/monitor unordered pair disjointness",
            f"fit={len(fit_idx)} monitor={len(monitor_idx)}",
        )

        fit_endpoints = set()
        monitor_endpoints = set()

        for i in fit_idx:
            fit_endpoints.update(pairs[i].tolist())

        for i in monitor_idx:
            monitor_endpoints.update(pairs[i].tolist())

        check(
            fit_endpoints.issubset(inner_train),
            f"{split} seed {seed}: teacher fit endpoints inside inner_train",
        )

        check(
            monitor_endpoints.issubset(inner_train),
            f"{split} seed {seed}: teacher monitor endpoints inside inner_train",
        )

        # Class-stratification sanity
        fit_y = labels[fit_idx]
        monitor_y = labels[monitor_idx]

        check(
            abs(float(fit_y.mean()) - 0.5) < 1e-3
            and abs(float(monitor_y.mean()) - 0.5) < 1e-3,
            f"{split} seed {seed}: teacher split class stratification",
            f"fit_pos={fit_y.mean():.6f} monitor_pos={monitor_y.mean():.6f}",
        )

        prefix = "s1" if split == "cold_start_S1" else "s2"

        ckpt = TEACHER_ROOT / f"{prefix}_seed_{seed}.pt"
        meta = TEACHER_ROOT / f"{prefix}_seed_{seed}.json"

        check(
            ckpt.exists(),
            f"{split} seed {seed}: teacher checkpoint exists",
        )

        if ckpt.exists():
            print(
                f"       SHA256 teacher {split} seed {seed}: "
                f"{sha256(ckpt)}"
            )

        # Metadata is useful but not mandatory if checkpoint exists.
        if meta.exists():
            try:
                md = json.loads(meta.read_text())

                if "seed" in md:
                    check(
                        int(md["seed"]) == seed,
                        f"{split} seed {seed}: teacher metadata seed",
                    )

                if "split" in md:
                    check(
                        md["split"] == split,
                        f"{split} seed {seed}: teacher metadata split",
                    )

            except Exception as exc:
                check(
                    False,
                    f"{split} seed {seed}: teacher metadata readable",
                    str(exc),
                )

    # ------------------------------------------------------------------
    # Static source audit for forbidden test-pair references.
    # This reads Python source only — NEVER test tensors.
    # ------------------------------------------------------------------

    for src_name, src in [
        ("teacher", TEACHER_SCRIPT[split]),
        ("student", TRAINER[split]),
    ]:
        check(
            src.exists(),
            f"{split}: {src_name} source exists",
            str(src),
        )

        if src.exists():
            hits = static_forbidden_pair_audit(src)

            check(
                len(hits) == 0,
                f"{split}: {src_name} has no executable test_pos/test_neg reference",
                str(hits) if hits else "none",
            )

            print(
                f"       SHA256 {src.name}: {sha256(src)}"
            )


print()
print("=" * 100)
print("FINAL V9-D AUDIT SUMMARY")
print("=" * 100)
print(f"PASS: {passes}")
print(f"FAIL: {fails}")
print("test_pos.pt loaded: NO")
print("test_neg.pt loaded: NO")
print("outer test status: SEALED")

if fails:
    raise SystemExit(1)

print("CERTIFICATION: PASS")
