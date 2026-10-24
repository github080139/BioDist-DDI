from pathlib import Path
import copy
import hashlib
import json
import os
import shutil

import torch


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PPI_PATH = (
    PROJECT_ROOT
    / "graphs"
    / "dpddi"
    / "ppi_edge_index.pt"
)

MASTER_DTI_PATH = (
    PROJECT_ROOT
    / "graphs"
    / "amddi_common"
    / "dti_edge_index.pt"
)

BIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "amddi_common_1087_biological_dev_v2"
)

DEST_PPI_PATH = (
    BIO_ROOT
    / "ppi_edge_index.pt"
)

ROOT_MANIFEST_PATH = (
    BIO_ROOT
    / "manifest.json"
)

MANIFEST_BACKUP_PATH = (
    BIO_ROOT
    / "manifest_before_ppi_certification.json"
)

CERTIFICATION_PATH = (
    BIO_ROOT
    / "ppi_certification.json"
)

HISTORICAL_GRAPH_ROOT = (
    PROJECT_ROOT
    / "graphs"
    / "amddi_common_1087_leakage_free_training"
)


# =============================================================================
# AUTHORITATIVE CONSTANTS
# =============================================================================

SETTINGS = [
    "transductive",
    "cold_start_S1",
    "cold_start_S2",
]

EXPECTED_PPI_EDGES = 107944
EXPECTED_DTI_EDGES = 7408
EXPECTED_NUM_DRUGS = 1087

# DTI endpoint semantics were independently certified in Script 113.
DTI_DRUG_ROW = 0
DTI_PROTEIN_ROW = 1

PPI_EDGE_TYPE = (
    "protein",
    "ppi",
    "protein",
)


# =============================================================================
# IO HELPERS
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


def atomic_json_save(obj, path):
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


def atomic_torch_save(obj, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        obj,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


# =============================================================================
# TENSOR HELPERS
# =============================================================================

def normalize_edge_index(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{name}: expected torch.Tensor, "
            f"got {type(x)}"
        )

    x = (
        x.detach()
        .cpu()
        .long()
        .contiguous()
    )

    if x.ndim != 2:
        raise RuntimeError(
            f"{name}: expected rank-2 tensor, "
            f"got shape {tuple(x.shape)}"
        )

    if x.shape[0] == 2:
        pass

    elif x.shape[1] == 2:
        x = x.t().contiguous()

    else:
        raise RuntimeError(
            f"{name}: expected [2,E] or [E,2], "
            f"got {tuple(x.shape)}"
        )

    if x.shape[1] == 0:
        raise RuntimeError(
            f"{name}: graph contains zero edges"
        )

    if x.min().item() < 0:
        raise RuntimeError(
            f"{name}: negative node index detected"
        )

    return x


def tensor_sha256(x):
    x = (
        x.detach()
        .cpu()
        .contiguous()
    )

    return hashlib.sha256(
        x.numpy().tobytes()
    ).hexdigest()


# =============================================================================
# HETEROGRAPH HELPERS
# =============================================================================

def get_protein_num_nodes(graph, graph_name):
    """
    Determine protein node-space size from an existing PyG heterogeneous graph.

    We intentionally do not infer the protein universe merely from PPI max ID,
    because DTI can contain proteins that are isolated from PPI.
    """

    if not hasattr(graph, "node_types"):
        raise RuntimeError(
            f"{graph_name}: object does not expose node_types"
        )

    if "protein" not in graph.node_types:
        raise RuntimeError(
            f"{graph_name}: missing protein node type"
        )

    store = graph["protein"]

    # Prefer explicitly stored num_nodes if available.
    try:
        num_nodes = store.num_nodes
    except Exception:
        num_nodes = None

    if num_nodes is not None:
        num_nodes = int(num_nodes)

        if num_nodes <= 0:
            raise RuntimeError(
                f"{graph_name}: invalid protein num_nodes={num_nodes}"
            )

        return num_nodes

    # Otherwise infer from x.
    if hasattr(store, "x") and store.x is not None:
        if store.x.ndim < 1:
            raise RuntimeError(
                f"{graph_name}: malformed protein x tensor"
            )

        return int(
            store.x.shape[0]
        )

    raise RuntimeError(
        f"{graph_name}: unable to determine protein node count"
    )


def get_ppi_from_historical_graph(graph, graph_name):
    if not hasattr(graph, "edge_types"):
        raise RuntimeError(
            f"{graph_name}: object does not expose edge_types"
        )

    if PPI_EDGE_TYPE not in graph.edge_types:
        raise RuntimeError(
            f"{graph_name}: missing edge type "
            f"{PPI_EDGE_TYPE}"
        )

    store = graph[
        PPI_EDGE_TYPE
    ]

    if not hasattr(store, "edge_index"):
        raise RuntimeError(
            f"{graph_name}: PPI relation has no edge_index"
        )

    return normalize_edge_index(
        store.edge_index,
        f"{graph_name}/PPI",
    )


# =============================================================================
# HISTORICAL GRAPH PATH RESOLUTION
# =============================================================================

def resolve_historical_graph_path(setting):
    """
    We know the corrected leakage-free training graphs were stored setting-wise,
    but this function remains conservative about the exact filename.

    It first tests the authoritative expected path. If absent, it looks only
    inside that setting-specific leakage-free graph directory/root and requires
    exactly one defensible candidate.
    """

    expected = (
        HISTORICAL_GRAPH_ROOT
        / f"{setting}_train_graph.pt"
    )

    if expected.exists():
        return expected

    if not HISTORICAL_GRAPH_ROOT.exists():
        raise FileNotFoundError(
            f"Historical leakage-free graph root missing:\n"
            f"{HISTORICAL_GRAPH_ROOT}"
        )

    candidates = []

    setting_lower = setting.lower()

    for path in sorted(
        HISTORICAL_GRAPH_ROOT.rglob("*.pt")
    ):
        p = str(path).lower()

        if (
            setting_lower in p
            and "train" in p
            and "graph" in p
        ):
            candidates.append(
                path
            )

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "\nCould not uniquely resolve historical "
        f"leakage-free graph for {setting}.\n"
        f"Expected path:\n{expected}\n"
        f"Candidate count: {len(candidates)}\n"
        f"Candidates:\n"
        + "\n".join(
            str(x)
            for x in candidates
        )
        + "\n\nSTOPPING rather than guessing."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 110)
    print("AM-DDI COMMON-1087")
    print(
        "SCRIPT 114 — CERTIFY AND INSTALL CANONICAL PPI "
        "FOR BIOLOGICAL DEV V2"
    )
    print("=" * 110)

    # =========================================================================
    # REQUIRED PATHS
    # =========================================================================

    required = [
        SOURCE_PPI_PATH,
        MASTER_DTI_PATH,
        BIO_ROOT,
        ROOT_MANIFEST_PATH,
        HISTORICAL_GRAPH_ROOT,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact:\n{path}"
            )

    print()
    print(
        "PASS: all required source paths exist."
    )

    # =========================================================================
    # LOAD ROOT MANIFEST
    # =========================================================================

    root_manifest = load_json(
        ROOT_MANIFEST_PATH
    )

    if (
        root_manifest.get("version")
        != "biological_dev_v2"
    ):
        raise RuntimeError(
            "Biological root manifest is not "
            "biological_dev_v2."
        )

    if (
        int(
            root_manifest.get(
                "num_drugs",
                -1,
            )
        )
        != EXPECTED_NUM_DRUGS
    ):
        raise RuntimeError(
            "Biological root manifest does not "
            "preserve common-1087 universe."
        )

    print(
        "PASS: biological_dev_v2 manifest identified."
    )

    # =========================================================================
    # LOAD SOURCE PPI
    # =========================================================================

    source_ppi = normalize_edge_index(
        load_torch(
            SOURCE_PPI_PATH
        ),
        "source PPI",
    )

    print()
    print(
        "Source PPI:"
    )

    print(
        f"  path         : {SOURCE_PPI_PATH}"
    )

    print(
        f"  shape        : {tuple(source_ppi.shape)}"
    )

    print(
        f"  edges        : {source_ppi.shape[1]}"
    )

    print(
        f"  min index    : {int(source_ppi.min())}"
    )

    print(
        f"  max index    : {int(source_ppi.max())}"
    )

    print(
        f"  unique nodes : "
        f"{int(torch.unique(source_ppi).numel())}"
    )

    if (
        source_ppi.shape[1]
        != EXPECTED_PPI_EDGES
    ):
        raise RuntimeError(
            f"Source PPI has {source_ppi.shape[1]} edges; "
            f"expected {EXPECTED_PPI_EDGES}."
        )

    print(
        "PASS: PPI edge count exactly matches "
        "historical 107944."
    )

    source_ppi_hash = tensor_sha256(
        source_ppi
    )

    # =========================================================================
    # LOAD MASTER DTI
    # =========================================================================

    master_dti = normalize_edge_index(
        load_torch(
            MASTER_DTI_PATH
        ),
        "master DTI",
    )

    if (
        master_dti.shape[1]
        != EXPECTED_DTI_EDGES
    ):
        raise RuntimeError(
            f"Master DTI has {master_dti.shape[1]} edges; "
            f"expected {EXPECTED_DTI_EDGES}."
        )

    if (
        int(master_dti[DTI_DRUG_ROW].max())
        >= EXPECTED_NUM_DRUGS
    ):
        raise RuntimeError(
            "Master DTI drug endpoint exceeds common-1087 "
            "drug namespace."
        )

    print()
    print(
        "PASS: master DTI shape and drug endpoint "
        "remain consistent."
    )

    dti_proteins = set(
        int(x)
        for x in torch.unique(
            master_dti[
                DTI_PROTEIN_ROW
            ]
        ).tolist()
    )

    ppi_proteins = set(
        int(x)
        for x in torch.unique(
            source_ppi
        ).tolist()
    )

    dti_protein_max = int(
        master_dti[
            DTI_PROTEIN_ROW
        ].max()
    )

    ppi_protein_max = int(
        source_ppi.max()
    )

    print()
    print(
        "Protein overlap diagnostics:"
    )

    print(
        f"  PPI unique proteins      : "
        f"{len(ppi_proteins)}"
    )

    print(
        f"  DTI unique proteins      : "
        f"{len(dti_proteins)}"
    )

    print(
        f"  intersection             : "
        f"{len(ppi_proteins & dti_proteins)}"
    )

    print(
        f"  PPI-only proteins        : "
        f"{len(ppi_proteins - dti_proteins)}"
    )

    print(
        f"  DTI-only proteins        : "
        f"{len(dti_proteins - ppi_proteins)}"
    )

    # =========================================================================
    # EXACT HISTORICAL PPI PROVENANCE CHECK
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "EXACT HISTORICAL LEAKAGE-FREE GRAPH MATCH"
    )
    print("=" * 110)

    historical_matches = {}
    protein_node_counts = {}

    for setting in SETTINGS:
        graph_path = (
            resolve_historical_graph_path(
                setting
            )
        )

        print()
        print(
            f"{setting}:"
        )

        print(
            f"  graph: {graph_path}"
        )

        graph = load_torch(
            graph_path
        )

        historical_ppi = (
            get_ppi_from_historical_graph(
                graph,
                str(graph_path),
            )
        )

        num_proteins = (
            get_protein_num_nodes(
                graph,
                str(graph_path),
            )
        )

        protein_node_counts[
            setting
        ] = num_proteins

        exact = torch.equal(
            historical_ppi,
            source_ppi,
        )

        historical_matches[
            setting
        ] = {
            "graph_path":
                str(graph_path),

            "ppi_shape":
                list(
                    historical_ppi.shape
                ),

            "ppi_sha256":
                tensor_sha256(
                    historical_ppi
                ),

            "exact_match":
                bool(exact),

            "protein_num_nodes":
                int(num_proteins),
        }

        print(
            f"  PPI shape       : "
            f"{tuple(historical_ppi.shape)}"
        )

        print(
            f"  protein nodes   : "
            f"{num_proteins}"
        )

        print(
            f"  exact PPI match : "
            f"{exact}"
        )

        if not exact:
            raise RuntimeError(
                f"{setting}: source PPI is NOT exactly equal "
                f"to historical leakage-free PPI relation."
            )

    print()
    print(
        "PASS: source PPI is bit-for-bit identical "
        "to PPI relation in all three historical "
        "leakage-free AM-DDI graphs."
    )

    # =========================================================================
    # PROTEIN NAMESPACE CONSISTENCY
    # =========================================================================

    unique_num_proteins = set(
        protein_node_counts.values()
    )

    if len(unique_num_proteins) != 1:
        raise RuntimeError(
            "Historical graphs disagree on protein "
            f"node-space size: {protein_node_counts}"
        )

    protein_num_nodes = int(
        next(
            iter(
                unique_num_proteins
            )
        )
    )

    if protein_num_nodes <= 0:
        raise RuntimeError(
            "Invalid protein node-space size."
        )

    if (
        ppi_protein_max
        >= protein_num_nodes
    ):
        raise RuntimeError(
            f"PPI max protein ID {ppi_protein_max} "
            f"exceeds protein node space "
            f"[0,{protein_num_nodes - 1}]."
        )

    if (
        dti_protein_max
        >= protein_num_nodes
    ):
        raise RuntimeError(
            f"DTI max protein ID {dti_protein_max} "
            f"exceeds protein node space "
            f"[0,{protein_num_nodes - 1}]."
        )

    print()
    print(
        "Protein namespace:"
    )

    print(
        f"  historical protein num_nodes : "
        f"{protein_num_nodes}"
    )

    print(
        f"  valid ID range               : "
        f"0..{protein_num_nodes - 1}"
    )

    print(
        f"  PPI max protein ID           : "
        f"{ppi_protein_max}"
    )

    print(
        f"  DTI max protein ID           : "
        f"{dti_protein_max}"
    )

    print()
    print(
        "PASS: PPI and DTI protein IDs fit inside "
        "the same historical protein node space."
    )

    # =========================================================================
    # DESTINATION SAFETY CHECK
    # =========================================================================

    if DEST_PPI_PATH.exists():
        existing_ppi = normalize_edge_index(
            load_torch(
                DEST_PPI_PATH
            ),
            "existing biological-v2 PPI",
        )

        if not torch.equal(
            existing_ppi,
            source_ppi,
        ):
            raise RuntimeError(
                "\nDestination PPI already exists but differs "
                "from certified source.\n"
                f"Destination:\n{DEST_PPI_PATH}\n\n"
                "STOPPING. No file was overwritten."
            )

        print()
        print(
            "PASS: destination PPI already exists and is "
            "identical to certified source."
        )

    # =========================================================================
    # BACKUP ROOT MANIFEST BEFORE FIRST MODIFICATION
    # =========================================================================

    if not MANIFEST_BACKUP_PATH.exists():
        shutil.copy2(
            ROOT_MANIFEST_PATH,
            MANIFEST_BACKUP_PATH,
        )

        print()
        print(
            "PASS: original biological-v2 manifest backed up."
        )

        print(
            f"  backup: {MANIFEST_BACKUP_PATH}"
        )

    else:
        print()
        print(
            "INFO: manifest backup already exists; "
            "it was not overwritten."
        )

    # =========================================================================
    # ATOMIC INSTALLATION
    # =========================================================================

    atomic_torch_save(
        source_ppi,
        DEST_PPI_PATH,
    )

    installed_ppi = normalize_edge_index(
        load_torch(
            DEST_PPI_PATH
        ),
        "installed PPI",
    )

    if not torch.equal(
        installed_ppi,
        source_ppi,
    ):
        raise RuntimeError(
            "Post-copy verification failed: installed "
            "PPI differs from certified source."
        )

    installed_hash = tensor_sha256(
        installed_ppi
    )

    if installed_hash != source_ppi_hash:
        raise RuntimeError(
            "Post-copy PPI tensor hash mismatch."
        )

    print()
    print(
        "PASS: certified PPI copied atomically."
    )

    print(
        f"  destination: {DEST_PPI_PATH}"
    )

    print(
        f"  SHA256     : {installed_hash}"
    )

    # =========================================================================
    # CERTIFICATION RECORD
    # =========================================================================

    certification = {
        "certification":
            "ppi_biological_dev_v2",

        "status":
            "PASSED",

        "source_ppi_path":
            str(
                SOURCE_PPI_PATH
            ),

        "installed_ppi_path":
            str(
                DEST_PPI_PATH
            ),

        "source_ppi_sha256":
            source_ppi_hash,

        "installed_ppi_sha256":
            installed_hash,

        "ppi_shape":
            list(
                installed_ppi.shape
            ),

        "ppi_edges":
            int(
                installed_ppi.shape[1]
            ),

        "ppi_unique_proteins":
            len(
                ppi_proteins
            ),

        "ppi_min_protein_id":
            int(
                installed_ppi.min()
            ),

        "ppi_max_protein_id":
            ppi_protein_max,

        "protein_namespace": {
            "num_nodes":
                protein_num_nodes,

            "min_valid_id":
                0,

            "max_valid_id":
                protein_num_nodes - 1,

            "source":
                "historical leakage-free AM-DDI heterogeneous graphs",
        },

        "dti_reference": {
            "path":
                str(
                    MASTER_DTI_PATH
                ),

            "edges":
                int(
                    master_dti.shape[1]
                ),

            "protein_row":
                DTI_PROTEIN_ROW,

            "unique_proteins":
                len(
                    dti_proteins
                ),

            "min_protein_id":
                int(
                    master_dti[
                        DTI_PROTEIN_ROW
                    ].min()
                ),

            "max_protein_id":
                dti_protein_max,
        },

        "protein_overlap": {
            "ppi_unique":
                len(
                    ppi_proteins
                ),

            "dti_unique":
                len(
                    dti_proteins
                ),

            "intersection":
                len(
                    ppi_proteins
                    & dti_proteins
                ),

            "ppi_only":
                len(
                    ppi_proteins
                    - dti_proteins
                ),

            "dti_only":
                len(
                    dti_proteins
                    - ppi_proteins
                ),
        },

        "historical_exact_matches":
            historical_matches,

        "policy": {
            "ppi_is_protein_only":
                True,

            "ppi_filtered_by_drug_split":
                False,

            "reason":
                (
                    "PPI connects protein nodes only and contains "
                    "no drug endpoint; the same certified protein-only "
                    "PPI topology is therefore retained for both "
                    "cold-start settings."
                ),

            "inner_validation_drug_exposure_through_ppi":
                False,

            "outer_heldout_drug_exposure_through_ppi":
                False,

            "dti_remains_setting_specific":
                True,

            "ddi_remains_setting_specific":
                True,

            "old_dti_derived_bio128_used":
                False,

            "outer_test_used":
                False,
        },
    }

    atomic_json_save(
        certification,
        CERTIFICATION_PATH,
    )

    print()
    print(
        "PASS: PPI certification record written."
    )

    print(
        f"  {CERTIFICATION_PATH}"
    )

    # =========================================================================
    # UPDATE ROOT BIOLOGICAL MANIFEST
    # =========================================================================

    updated_manifest = copy.deepcopy(
        root_manifest
    )

    updated_manifest[
        "ppi_status"
    ] = "loaded"

    updated_manifest[
        "ppi_path"
    ] = str(
        DEST_PPI_PATH
    )

    updated_manifest[
        "ppi_source_path"
    ] = str(
        SOURCE_PPI_PATH
    )

    updated_manifest[
        "ppi_certification_path"
    ] = str(
        CERTIFICATION_PATH
    )

    updated_manifest[
        "ppi_sha256"
    ] = installed_hash

    updated_manifest[
        "ppi_edges"
    ] = int(
        installed_ppi.shape[1]
    )

    updated_manifest[
        "ppi_unique_proteins"
    ] = len(
        ppi_proteins
    )

    updated_manifest[
        "protein_num_nodes"
    ] = protein_num_nodes

    updated_manifest[
        "ppi_certified"
    ] = True

    updated_manifest[
        "ppi_historical_exact_match_all_settings"
    ] = True

    updated_manifest[
        "ppi_is_protein_only"
    ] = True

    if (
        "important_policy"
        not in updated_manifest
    ):
        updated_manifest[
            "important_policy"
        ] = {}

    updated_manifest[
        "important_policy"
    ][
        "ppi_certified"
    ] = True

    updated_manifest[
        "important_policy"
    ][
        "ppi_is_protein_only"
    ] = True

    updated_manifest[
        "important_policy"
    ][
        "ppi_contains_drug_nodes"
    ] = False

    updated_manifest[
        "important_policy"
    ][
        "ppi_requires_drug_split_filtering"
    ] = False

    updated_manifest[
        "important_policy"
    ][
        "outer_test_used_for_ppi_certification"
    ] = False

    atomic_json_save(
        updated_manifest,
        ROOT_MANIFEST_PATH,
    )

    # Reload to confirm.
    verify_manifest = load_json(
        ROOT_MANIFEST_PATH
    )

    if (
        verify_manifest.get(
            "ppi_status"
        )
        != "loaded"
    ):
        raise RuntimeError(
            "Manifest post-write verification failed: "
            "ppi_status != loaded"
        )

    if (
        verify_manifest.get(
            "ppi_sha256"
        )
        != installed_hash
    ):
        raise RuntimeError(
            "Manifest post-write verification failed: "
            "PPI hash mismatch"
        )

    if (
        verify_manifest.get(
            "ppi_certified"
        )
        is not True
    ):
        raise RuntimeError(
            "Manifest post-write verification failed: "
            "ppi_certified != True"
        )

    print()
    print(
        "PASS: biological-v2 root manifest updated "
        "and verified."
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 110)
    print("SCRIPT 114 PPI CERTIFICATION SUMMARY")
    print("=" * 110)

    print(
        "PASS: canonical source PPI identified."
    )

    print(
        f"PASS: PPI directed edges = "
        f"{EXPECTED_PPI_EDGES}."
    )

    print(
        "PASS: exact PPI match confirmed against "
        "all three historical leakage-free AM-DDI graphs."
    )

    print(
        f"PASS: common protein node space = "
        f"{protein_num_nodes} nodes."
    )

    print(
        "PASS: PPI IDs are inside protein namespace."
    )

    print(
        "PASS: DTI protein IDs are inside same "
        "protein namespace."
    )

    print(
        "PASS: PPI is protein-only and therefore "
        "does not directly expose cold-start drugs."
    )

    print(
        "PASS: certified PPI installed into "
        "biological_dev_v2."
    )

    print(
        "PASS: biological-v2 manifest updated."
    )

    print(
        "PASS: outer test was not used."
    )

    print()
    print(
        "PPI CERTIFICATION: PASSED"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT start biological/full AM-DDI training yet."
    )

    print(
        "First rerun Script 113 and require:"
    )

    print(
        "  Warnings : 0"
    )

    print(
        "  Failures : 0"
    )


if __name__ == "__main__":
    main()
