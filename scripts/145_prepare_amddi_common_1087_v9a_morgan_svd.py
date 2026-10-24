import os
import json
import hashlib
import torch

from sklearn.decomposition import TruncatedSVD


BASE = "data/amddi_common_1087_leakage_free_features"
SPLIT_BASE = "data/amddi_common_1087_coldstart_dev_splits_v2"

OUT = "data/amddi_common_1087_v9a_morgan_svd"

os.makedirs(OUT, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def prepare(split):

    print("=" * 100)
    print("V9-A TRAIN-ONLY MORGAN TRUNCATED-SVD PREPARATION")
    print("=" * 100)

    print("Split            :", split)
    print("Components       :", 128)
    print("SVD random state :", 42)

    morgan_file = (
        f"{BASE}/{split}_morgan_2048.pt"
    )

    split_dir = (
        f"{SPLIT_BASE}/{split}"
    )

    X = torch.load(
        morgan_file,
        weights_only=True
    )

    fit_ids = torch.load(
        f"{split_dir}/inner_train_drugs.pt",
        weights_only=True
    )

    fit_ids = fit_ids.long()

    print()
    print("Morgan shape     :", tuple(X.shape))
    print(
        "Unique fit drugs :",
        len(torch.unique(fit_ids))
    )

    X_fit = X[fit_ids]


    svd = TruncatedSVD(
        n_components=128,
        random_state=42
    )

    svd.fit(
        X_fit.numpy()
    )


    X_svd = svd.transform(
        X.numpy()
    )


    X_svd = torch.tensor(
        X_svd,
        dtype=torch.float32
    )


    feature_file = (
        f"{OUT}/{split}_morgan_svd128.pt"
    )

    component_file = (
        f"{OUT}/{split}_morgan_svd128_components.pt"
    )

    fit_file = (
        f"{OUT}/{split}_svd_fit_drug_ids.pt"
    )

    metadata_file = (
        f"{OUT}/{split}_morgan_svd128_metadata.json"
    )


    torch.save(
        X_svd,
        feature_file
    )

    torch.save(
        torch.tensor(
            svd.components_,
            dtype=torch.float32
        ),
        component_file
    )

    torch.save(
        fit_ids,
        fit_file
    )


    metadata = {

        "split":
        split,

        "method":
        "TruncatedSVD",

        "components":
        128,

        "random_state":
        42,

        "input_shape":
        list(X.shape),

        "output_shape":
        list(X_svd.shape),

        "fit_drug_count":
        int(len(torch.unique(fit_ids))),

        "explained_variance":
        float(
            svd.explained_variance_ratio_.sum()
        ),

        "validation_used":
        False,

        "outer_test_loaded":
        False,

        "outer_test_status":
        "SEALED",

        "source_sha256":
        sha256_file(morgan_file)

    }


    with open(metadata_file, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2
        )


    print()
    print("-" * 100)
    print("Output shape     :", tuple(X_svd.shape))
    print(
        "Explained var    :",
        metadata["explained_variance"]
    )
    print("Feature file     :", feature_file)
    print("Component file   :", component_file)
    print("Fit-ID file      :", fit_file)
    print("Metadata file    :", metadata_file)
    print("Outer test       : SEALED")
    print("PASS             :", split)



for split in [
    "cold_start_S1",
    "cold_start_S2"
]:

    prepare(split)


print()
print("=" * 100)
print("DONE — V9-A TRAIN-ONLY MORGAN SVD PREPARATION COMPLETE")
print("=" * 100)
