from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data


ROOT = Path("/home/fec-rl/DDI-project")

MAPPING_FILE = (
    ROOT / "data/amddi_common/drug_mapping.csv"
)

SMILES_FILE = (
    ROOT
    / "data/processed/updated_smiles/"
      "drug_smiles_final_current.csv"
)

OUTDIR = (
    ROOT
    / "data/amddi_common_1087_structure"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

GRAPH_FILE = OUTDIR / "molecular_graphs.pt"
MANIFEST_FILE = OUTDIR / "structure_manifest.csv"
SUMMARY_FILE = OUTDIR / "structure_summary.txt"


# ============================================================
# Atom features
# ============================================================

ALLOWABLE_ELEMENTS = [
    "H", "B", "C", "N", "O", "F",
    "Si", "P", "S", "Cl", "Br", "I",
]


def one_hot_with_unknown(value, choices):
    """
    One-hot encoding with an additional unknown category.
    """
    out = [0] * (len(choices) + 1)

    if value in choices:
        out[choices.index(value)] = 1
    else:
        out[-1] = 1

    return out


def atom_features(atom):

    features = []

    # Element
    features += one_hot_with_unknown(
        atom.GetSymbol(),
        ALLOWABLE_ELEMENTS,
    )

    # Degree: 0..5 + unknown
    features += one_hot_with_unknown(
        atom.GetDegree(),
        [0, 1, 2, 3, 4, 5],
    )

    # Formal charge as scalar
    features.append(
        float(atom.GetFormalCharge())
    )

    # Number of hydrogens
    features += one_hot_with_unknown(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4],
    )

    # Hybridization
    features += one_hot_with_unknown(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
    )

    # Aromatic
    features.append(
        float(atom.GetIsAromatic())
    )

    # Ring membership
    features.append(
        float(atom.IsInRing())
    )

    # Chirality possible
    features.append(
        float(
            atom.HasProp(
                "_ChiralityPossible"
            )
        )
    )

    return features


# ============================================================
# Bond features
# ============================================================

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def bond_features(bond):

    features = []

    features += one_hot_with_unknown(
        bond.GetBondType(),
        BOND_TYPES,
    )

    features.append(
        float(bond.GetIsConjugated())
    )

    features.append(
        float(bond.IsInRing())
    )

    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]

    features += one_hot_with_unknown(
        bond.GetStereo(),
        stereo_choices,
    )

    return features


# ============================================================
# SMILES -> PyG graph
# ============================================================

def smiles_to_graph(
    smiles,
    node_id,
    drug_id,
):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"RDKit failed: "
            f"node={node_id}, "
            f"drug={drug_id}"
        )

    # ----------------------------
    # Node features
    # ----------------------------

    x = torch.tensor(
        [
            atom_features(atom)
            for atom in mol.GetAtoms()
        ],
        dtype=torch.float32,
    )

    # ----------------------------
    # Edges
    # ----------------------------

    src = []
    dst = []
    edge_attr = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = bond_features(bond)

        # Both directions
        src.extend([i, j])
        dst.extend([j, i])

        edge_attr.append(bf)
        edge_attr.append(bf)

    if len(src) > 0:

        edge_index = torch.tensor(
            [src, dst],
            dtype=torch.long,
        )

        edge_attr = torch.tensor(
            edge_attr,
            dtype=torch.float32,
        )

    else:

        # Infer bond feature dimension
        bond_dim = (
            len(BOND_TYPES) + 1
            + 1
            + 1
            + 4 + 1
        )

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_attr = torch.empty(
            (0, bond_dim),
            dtype=torch.float32,
        )

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    graph.node_id = int(node_id)
    graph.drug_id = str(drug_id)

    return graph


# ============================================================
# Load authoritative mapping
# ============================================================

mapping = pd.read_csv(MAPPING_FILE)

assert len(mapping) == 1087
assert mapping["node_id"].nunique() == 1087
assert mapping["drug_id"].nunique() == 1087

mapping = mapping.sort_values(
    "node_id"
).reset_index(drop=True)

expected_ids = list(range(1087))

assert (
    mapping["node_id"].tolist()
    == expected_ids
), "node_id is not exactly 0..1086"


# ============================================================
# Load structures
# ============================================================

smiles_df = pd.read_csv(SMILES_FILE)

required = [
    "drug_id",
    "smiles",
    "rdkit_valid",
    "structure_source",
]

for col in required:
    if col not in smiles_df.columns:
        raise RuntimeError(
            f"Missing column: {col}"
        )

selected = mapping.merge(
    smiles_df[
        [
            "drug_id",
            "smiles",
            "rdkit_valid",
            "structure_source",
            "drug_name",
        ]
    ],
    on="drug_id",
    how="left",
    validate="one_to_one",
)

assert len(selected) == 1087
assert selected["smiles"].notna().all()


# ============================================================
# Build molecular graphs
# ============================================================

graphs = []
manifest_rows = []

atom_dim = None
bond_dim = None

total_atoms = 0
total_directed_edges = 0

print("=" * 90)
print("BUILDING COMMON-1087 MOLECULAR GRAPHS")
print("=" * 90)

for row in selected.itertuples(
    index=False
):

    graph = smiles_to_graph(
        row.smiles,
        row.node_id,
        row.drug_id,
    )

    if atom_dim is None:
        atom_dim = graph.x.shape[1]

    if graph.x.shape[1] != atom_dim:
        raise RuntimeError(
            "Inconsistent atom feature dimension"
        )

    if bond_dim is None:
        bond_dim = graph.edge_attr.shape[1]

    if graph.edge_attr.shape[1] != bond_dim:
        raise RuntimeError(
            "Inconsistent bond feature dimension"
        )

    graphs.append(graph)

    num_atoms = graph.num_nodes
    num_edges = graph.edge_index.shape[1]

    total_atoms += num_atoms
    total_directed_edges += num_edges

    manifest_rows.append(
        {
            "node_id": row.node_id,
            "drug_id": row.drug_id,
            "drug_name": row.drug_name,
            "smiles": row.smiles,
            "structure_source":
                row.structure_source,
            "num_atoms": num_atoms,
            "num_directed_edges":
                num_edges,
        }
    )


# ============================================================
# Final alignment audit
# ============================================================

assert len(graphs) == 1087

for expected_node_id, graph in enumerate(
    graphs
):
    assert graph.node_id == expected_node_id

    expected_drug = mapping.iloc[
        expected_node_id
    ]["drug_id"]

    assert graph.drug_id == expected_drug


# ============================================================
# Save
# ============================================================

torch.save(
    graphs,
    GRAPH_FILE,
)

manifest = pd.DataFrame(
    manifest_rows
)

manifest.to_csv(
    MANIFEST_FILE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

summary = f"""
AM-DDI COMMON-1087 MOLECULAR GRAPH SUMMARY
===========================================

Number of drugs       : {len(graphs)}
Atom feature dimension: {atom_dim}
Bond feature dimension: {bond_dim}

Total atoms           : {total_atoms}
Total directed edges  : {total_directed_edges}

Mean atoms/drug       : {total_atoms / len(graphs):.4f}
Mean edges/drug       : {total_directed_edges / len(graphs):.4f}

Graph file:
{GRAPH_FILE}

Manifest:
{MANIFEST_FILE}

Alignment:
node_id 0..1086 = PASS
drug_id mapping  = PASS
RDKit conversion = PASS
"""

SUMMARY_FILE.write_text(summary)

print(summary)

print(
    "First graph:",
    graphs[0]
)

print(
    "\nPASS: all 1087 molecular graphs "
    "built and aligned."
)
