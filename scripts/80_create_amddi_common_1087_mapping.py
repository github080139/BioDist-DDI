import pandas as pd
import os


os.makedirs(
    "data/amddi_common",
    exist_ok=True
)


# -----------------------------
# Load drug universes
# -----------------------------

ddi = pd.read_csv(
    "graphs/ddi_graph/drug_node_mapping.csv"
)


morgan = pd.read_csv(
    "features/chemical/morgan_drug_mapping_final.csv"
)


bio = pd.read_csv(
    "features/amddi_v3/drug_protein_mapping.csv"
)



ddi_drugs = set(
    ddi["drug_id"].astype(str)
)


morgan_drugs = set(
    morgan["drug_id"].astype(str)
)


bio_drugs = set(
    bio["drug_id"].astype(str)
)



# -----------------------------
# Common modality
# -----------------------------

common_drugs = sorted(
    ddi_drugs
    &
    morgan_drugs
    &
    bio_drugs
)



print("="*70)
print("FINAL AM-DDI COMMON MODALITY")
print("="*70)

print(
    "DDI drugs:",
    len(ddi_drugs)
)

print(
    "Morgan drugs:",
    len(morgan_drugs)
)

print(
    "Biological drugs:",
    len(bio_drugs)
)

print(
    "Common drugs:",
    len(common_drugs)
)



mapping = pd.DataFrame(
    {
        "node_id": range(len(common_drugs)),
        "drug_id": common_drugs
    }
)



mapping.to_csv(
    "data/amddi_common/drug_mapping.csv",
    index=False
)



print("\nSaved:")
print(
    "data/amddi_common/drug_mapping.csv"
)


print("\nFirst 5:")
print(mapping.head())

print("\nLast 5:")
print(mapping.tail())
