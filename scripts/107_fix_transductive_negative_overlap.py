#!/usr/bin/env python3

from __future__ import annotations

import random
from pathlib import Path

import torch


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    ROOT
    / "data/amddi_common_splits/transductive"
)

N_DRUGS = 1087

SEED = 42


# ============================================================
# Helpers
# ============================================================


def load(name: str) -> torch.Tensor:

    path = SPLIT_DIR / f"{name}.pt"

    if not path.exists():
        raise FileNotFoundError(path)

    x = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    ).long()

    if (
        x.ndim != 2
        or x.shape[1] != 2
    ):
        raise RuntimeError(
            f"{name}: expected [N,2], "
            f"got {tuple(x.shape)}"
        )

    return x


def canonical_pair(a: int, b: int):

    a = int(a)
    b = int(b)

    if a == b:
        raise RuntimeError(
            f"Self-pair detected: {a},{b}"
        )

    return (
        min(a, b),
        max(a, b),
    )


def canonical_set(
    tensor: torch.Tensor,
):

    result = set()

    for a, b in tensor.tolist():

        result.add(
            canonical_pair(
                a,
                b,
            )
        )

    if len(result) != len(tensor):

        raise RuntimeError(
            "Duplicate undirected pairs "
            "inside tensor."
        )

    return result


def tensor_from_pairs(
    pairs,
):

    return torch.tensor(
        list(pairs),
        dtype=torch.long,
    )


def atomic_save(
    tensor: torch.Tensor,
    path: Path,
):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        tensor,
        tmp,
    )

    tmp.replace(
        path
    )


# ============================================================
# Load frozen split
# ============================================================

train_pos = load(
    "train_pos"
)

train_neg = load(
    "train_neg"
)

val_pos = load(
    "val_pos"
)

val_neg = load(
    "val_neg"
)

test_pos = load(
    "test_pos"
)

test_neg = load(
    "test_neg"
)


print()
print("=" * 90)
print(
    "TRANSDUCTIVE NEGATIVE-OVERLAP REPAIR"
)
print("=" * 90)


# ============================================================
# Original sizes
# ============================================================

target_val_neg = len(
    val_neg
)

target_test_neg = len(
    test_neg
)


print()
print("Original sizes")

print(
    "  train_pos:",
    len(train_pos),
)

print(
    "  train_neg:",
    len(train_neg),
)

print(
    "  val_pos  :",
    len(val_pos),
)

print(
    "  val_neg  :",
    len(val_neg),
)

print(
    "  test_pos :",
    len(test_pos),
)

print(
    "  test_neg :",
    len(test_neg),
)


# ============================================================
# Canonical sets
# ============================================================

train_pos_set = canonical_set(
    train_pos
)

train_neg_set = canonical_set(
    train_neg
)

val_pos_set = canonical_set(
    val_pos
)

val_neg_set = canonical_set(
    val_neg
)

test_pos_set = canonical_set(
    test_pos
)

test_neg_set = canonical_set(
    test_neg
)


all_positive = (
    train_pos_set
    | val_pos_set
    | test_pos_set
)


# ============================================================
# Fundamental positive audits
# ============================================================

if (
    train_pos_set
    & val_pos_set
):

    raise RuntimeError(
        "train_pos / val_pos overlap."
    )

if (
    train_pos_set
    & test_pos_set
):

    raise RuntimeError(
        "train_pos / test_pos overlap."
    )

if (
    val_pos_set
    & test_pos_set
):

    raise RuntimeError(
        "val_pos / test_pos overlap."
    )


if all_positive & train_neg_set:

    raise RuntimeError(
        "Existing train negatives "
        "contradict known positives."
    )


print()
print(
    "Positive split overlap: PASS"
)

print(
    "Train-negative label audit: PASS"
)


# ============================================================
# Determine active transductive drug universe
# ============================================================

active_drugs = set()

for pair in all_positive:

    active_drugs.update(
        pair
    )


active_drugs = sorted(
    active_drugs
)


print()
print(
    "Active transductive drugs:",
    len(active_drugs),
)

print(
    "Common node universe      :",
    N_DRUGS,
)


if not active_drugs:

    raise RuntimeError(
        "No active drugs found."
    )

if min(active_drugs) < 0:

    raise RuntimeError(
        "Negative node ID."
    )

if max(active_drugs) >= N_DRUGS:

    raise RuntimeError(
        "Node outside 1087 universe."
    )


# ============================================================
# Preserve clean validation negatives
# ============================================================

clean_val = []

used_negative = set(
    train_neg_set
)


for a, b in val_neg.tolist():

    pair = canonical_pair(
        a,
        b,
    )

    # Remove anything already used in train
    if pair in used_negative:
        continue

    # Defensive positive exclusion
    if pair in all_positive:
        continue

    clean_val.append(
        pair
    )

    used_negative.add(
        pair
    )


removed_val = (
    target_val_neg
    - len(clean_val)
)


print()
print(
    "Validation negatives"
)

print(
    "  target  :",
    target_val_neg,
)

print(
    "  retained:",
    len(clean_val),
)

print(
    "  replace :",
    removed_val,
)


# ============================================================
# Random sampler
# ============================================================

rng = random.Random(
    SEED
)


def sample_new_pair():

    while True:

        a, b = rng.sample(
            active_drugs,
            2,
        )

        pair = canonical_pair(
            a,
            b,
        )

        if pair in all_positive:
            continue

        if pair in used_negative:
            continue

        return pair


# ============================================================
# Fill validation
# ============================================================

while (
    len(clean_val)
    < target_val_neg
):

    pair = sample_new_pair()

    clean_val.append(
        pair
    )

    used_negative.add(
        pair
    )


# ============================================================
# Preserve clean test negatives
# ============================================================

clean_test = []


for a, b in test_neg.tolist():

    pair = canonical_pair(
        a,
        b,
    )

    # This now excludes:
    # train negatives +
    # corrected validation negatives
    if pair in used_negative:
        continue

    if pair in all_positive:
        continue

    clean_test.append(
        pair
    )

    used_negative.add(
        pair
    )


removed_test = (
    target_test_neg
    - len(clean_test)
)


print()
print(
    "Test negatives"
)

print(
    "  target  :",
    target_test_neg,
)

print(
    "  retained:",
    len(clean_test),
)

print(
    "  replace :",
    removed_test,
)


# ============================================================
# Fill test
# ============================================================

while (
    len(clean_test)
    < target_test_neg
):

    pair = sample_new_pair()

    clean_test.append(
        pair
    )

    used_negative.add(
        pair
    )


# ============================================================
# Convert back to tensors
# ============================================================

new_val_neg = tensor_from_pairs(
    clean_val
)

new_test_neg = tensor_from_pairs(
    clean_test
)


# ============================================================
# Final strict audit BEFORE saving
# ============================================================

new_val_set = canonical_set(
    new_val_neg
)

new_test_set = canonical_set(
    new_test_neg
)


if len(new_val_neg) != target_val_neg:

    raise RuntimeError(
        "Validation negative count changed."
    )

if len(new_test_neg) != target_test_neg:

    raise RuntimeError(
        "Test negative count changed."
    )


if train_neg_set & new_val_set:

    raise RuntimeError(
        "train_neg / new val_neg overlap."
    )

if train_neg_set & new_test_set:

    raise RuntimeError(
        "train_neg / new test_neg overlap."
    )

if new_val_set & new_test_set:

    raise RuntimeError(
        "new val_neg / new test_neg overlap."
    )


if all_positive & new_val_set:

    raise RuntimeError(
        "Positive / new val-negative "
        "contradiction."
    )

if all_positive & new_test_set:

    raise RuntimeError(
        "Positive / new test-negative "
        "contradiction."
    )


train_all = (
    train_pos_set
    | train_neg_set
)

val_all = (
    val_pos_set
    | new_val_set
)

test_all = (
    test_pos_set
    | new_test_set
)


if train_all & val_all:

    raise RuntimeError(
        "Final train/validation overlap."
    )

if train_all & test_all:

    raise RuntimeError(
        "Final train/test overlap."
    )

if val_all & test_all:

    raise RuntimeError(
        "Final validation/test overlap."
    )


print()
print(
    "Final strict audit"
)

print(
    "  train vs val overlap :",
    len(
        train_all
        & val_all
    ),
)

print(
    "  train vs test overlap:",
    len(
        train_all
        & test_all
    ),
)

print(
    "  val vs test overlap  :",
    len(
        val_all
        & test_all
    ),
)

print(
    "  positive contradiction:",
    len(
        all_positive
        & (
            train_neg_set
            | new_val_set
            | new_test_set
        )
    ),
)


# ============================================================
# Save ONLY corrected negative tensors
# ============================================================

atomic_save(
    new_val_neg,
    SPLIT_DIR
    / "val_neg.pt",
)

atomic_save(
    new_test_neg,
    SPLIT_DIR
    / "test_neg.pt",
)


print()
print(
    "Saved:"
)

print(
    " ",
    SPLIT_DIR
    / "val_neg.pt",
)

print(
    " ",
    SPLIT_DIR
    / "test_neg.pt",
)


print()
print("=" * 90)
print(
    "PASS: TRANSDUCTIVE NEGATIVE "
    "OVERLAP REPAIRED"
)
print("=" * 90)
