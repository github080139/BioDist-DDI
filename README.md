# BioDist-DDI

**BioDist-DDI: Cold-Start Drug–Drug Interaction Prediction for Medication-Safety Screening Using Training-Only Biological Distillation**

BioDist-DDI is a drug–drug interaction (DDI) prediction framework designed for cold-start evaluation. Biological information derived from drug–target interactions (DTI), ESM-2 protein representations, and protein–protein interactions (PPI) is used only during training through representation distillation. At validation, testing, and inference time, the student model uses chemical information only.

## Model Overview

The BioDist-DDI student contains two chemical branches:

1. A molecular-graph branch using a 3-layer GINE encoder.
2. A Morgan fingerprint branch using 2048-bit Morgan fingerprints reduced to 128 dimensions using TruncatedSVD.

For each drug pair, each branch constructs a pair representation:

```text
[h_i, h_j, |h_i - h_j|]
