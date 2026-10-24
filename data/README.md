# Data

This repository contains code and frozen split definitions required to reproduce the BioDist-DDI experiments.

The full source datasets and derived feature tensors are not redistributed here where redistribution rights may be unclear.

BioDist-DDI uses the following data modalities:

- DDI interaction data
- Drug-target interaction (DTI) data
- Protein-protein interaction (PPI) data
- Molecular structures
- Morgan fingerprints
- ESM-2 protein embeddings

The published split definitions are provided under:

- `splits/cold_start_S1/`
- `splits/cold_start_S2/`
- `splits/transductive/`

The exact hashes of these split files are stored in:

`metadata/split_sha256.txt`

Feature-generation and preprocessing scripts are provided under `scripts/`.

Users should obtain the corresponding source datasets from their original providers and regenerate derived features using the supplied scripts.
