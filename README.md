# Wildlife Re-Identification (Problem Statement 3)
## Machine Learning Course Project - IIIT Delhi

This repository contains our team's complete, reproducible code and final submissions for the **Wildlife Re-Identification** Kaggle competition.

---

## Team Members

| Name | Roll Number | Email | Role |
| :--- | :---: | :--- | :--- |
| **R S Maneesh** (Group Representative) | **2024449** | maneesh24449@iiitd.ac.in | Model 1 (Baseline k-NN), Exploratory Data Analysis |
| **Raunak Mishra** | **2024455** | raunak24455@iiitd.ac.in | Model 3 (Multi-Space GBDT), Pipeline Architecture, Notebook |
| **Rishabh Khannawalia** | **2024463** | rishabh24463@iiitd.ac.in | Encounter-Disjoint Open-Set CV, Novelty Thresholds |
| **Shakti** | **2024519** | shakti24519@iiitd.ac.in | Viewpoint Flank Analysis, View-Split Centroids, Error Auditing |

---

## 1. Project Overview & Approach

The goal is to identify individual spotted hyenas and leopards from camera-trap sightings using precomputed 2,152-dimensional MiewID feature descriptors. The test set has 2,791 images. An animal can either belong to an individual seen during training (of the same species) or be an unseen animal labeled as `new_individual`. Evaluation is strictly on **MAP@5**.

### Key Pipeline Components:
1. **Feature Normalization & Species Separation**: All MiewID descriptors are L2-normalized onto the unit hypersphere so dot products compute exact cosine similarity. Retrieval runs strictly within species (`hyena` vs `leopard`).
2. **Encounter-Disjoint Open-Set Validation**:
   - Standard random image splits leak same-burst photos (`39.8%` of leopard queries share a `>= 0.80` gallery twin locally vs only `8.9%` on Kaggle test).
   - We cluster same-identity burst frames (`cos >= 0.82`) into discrete encounters via connected components, hold out ~25-26% of identities completely as `new_individual`, and hold out entire encounters for known queries across 3 stratified folds.
3. **Viewpoint Flank Asymmetry & View-Split Prototypes**:
   - Spotted hyenas and leopards have asymmetric coat markings across left and right flanks.
   - Model 2 applies an opposite-flank penalty (`-0.05`), while Model 3 builds separate `left`, `right`, and `neutral` identity prototypes so opposite flanks never blur together.
4. **Multi-Space Whitened Retrieval & Two-Stage GBDT + Bayes Expected MAP@5**:
   - Model 3 combines shrinkage LDA whitening, PCA whitening, CSLS hubness suppression, balanced kernel ridge separation, inductive alpha-QE/DBA, and k-reciprocal Jaccard re-ranking (`k1=8, k2=2, lam=0.55`).
   - A Stage-A ensemble (`LGBMClassifier`, `HistGradientBoostingClassifier`, `LGBMRanker`) scores candidates across 37+ retrieval/margin/coherence features, a Stage-B novelty ensemble estimates $P(\text{new\_individual} \mid q)$, and Bayes Expected MAP@5 decoding blends the inductive (`55%`) and clean hybrid k-reciprocal (`45%`) branches to reach **0.8177** on Kaggle.

---

## 2. Model Evolution & Results (The 3 Model Stages)

As required by the project instructions, we developed three distinct model versions showing clear progression:

| Model Version | Description | Key Additions | Hyena CV | Leopard CV | Weighted CV MAP@5 | Kaggle Public Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Model 1: Baseline** | Cosine Similarity k-NN | L2-normalization, nearest neighbors, naive rank-5 `new_individual` | 0.5765 | 0.5327 | 0.5518 | — |
| **Model 2: Enhanced k-NN** | Viewpoint + Calibrated Thresholding | Opposite-flank penalty (`-0.05`), species-specific `new_individual` cutoffs (`0.52` / `0.54`) | 0.7898 | 0.7571 | 0.7714 | 0.75821 |
| **Model 3: Multi-Space GBDT** | Whitened Retrieval + Two-Stage GBDT | Shrinkage LDA/PCA, view-split prototypes, CSLS, balanced ridge, k-reciprocal + Stage A/B GBDT & Bayes MAP@5 | 0.8863 | 0.8304 | **0.8547** | **0.8177** (Best) |

*\*Note: Model 1 was evaluated on local encounter-disjoint CV; to preserve daily submission limits, our enhanced pipelines (Model 2 and Model 3) were submitted to Kaggle.*

---

## 3. Individual Contributions Breakdown

* **R S Maneesh (2024449)** — *Group Representative*:
  * Team lead, repository hosting, and group coordination.
  * Conducted Exploratory Data Analysis (EDA) on species distributions, identity frequencies, and class imbalance (noting that 99 leopard identities are singletons with only 1 photo).
  * Built and benchmarked **Model 1** (`submission_model1.csv`) using L2-normalized cosine k-NN with naive Rank-5 `new_individual`.
  * Performed error analysis on Model 1 showing that failing to prioritize unseen animals drops encounter-disjoint MAP@5 to `0.5518`.

* **Raunak Mishra (2024455)**:
  * Formulated and implemented **Model 3** (`submission_model3.csv` / `submission.csv`): multi-space shrinkage LDA and PCA whitening, CSLS hubness suppression, balanced ridge regression, and inductive + k-reciprocal two-stage GBDT ensembles (`LGBMClassifier`, `HistGradientBoostingClassifier`, `LGBMRanker`, `GradientBoostingClassifier`).
  * Replaced fixed heuristic novelty thresholds with Stage-B posterior estimation and Bayes Expected MAP@5 list decoding.
  * Refactored the end-to-end `run_pipeline.py` script and `Wildlife_ReID_Pipeline.ipynb` notebook, bringing our Kaggle public score up from `0.75821` to **0.8177**.

* **Rishabh Khannawalia (2024463)**:
  * Audited camera-trap burst leakage during random splits (`39.8%` vs `8.9%` high-similarity leopard twins) and built the encounter-disjoint 3-fold open-set validation protocol.
  * Implemented and tuned the Rank-1 novelty thresholds for **Model 2** ($T = 0.52$ for Hyena, $0.54$ for Leopard) to cleanly separate seen vs unseen animals.
  * Verified zero-leakage submission checks (`2,791` test rows, unique valid same-species IDs, zero nulls) and fold-strict inductive whitening boundaries.

* **Shakti (2024519)**:
  * Audited viewpoint metadata and verified biological flank asymmetry between `left` and `right` sightings in hyenas and leopards.
  * Implemented Model 2's opposite-flank penalty (`-0.05`) and Model 3's viewpoint-split `left` / `right` / `neutral` prototypes so asymmetric flanks do not blur centroids together.
  * Analyzed Model 2 failure modes (background lighting variance, singleton suppression, and flat top-5 confidence curves) that motivated Model 3.

---

## 4. Setup & Execution Guide

### Prerequisites
- **Python**: Version `3.11` is required (tested on Python `3.11.9`).
- **Data & Environment Notice**: In compliance with course rules and repository size limits, the dataset (`data/`) and Python virtual environments (`.venv*`) are **not** committed to GitHub and must be set up locally.

### Step 1: Environment Setup (Windows)
Create and activate a virtual environment:
```powershell
python -m venv .venv311
.venv311\Scripts\activate
```

Install the required dependencies:
```powershell
pip install -r requirements.txt
```

### Step 2: Expected Dataset Directory Structure
Extract the competition dataset into the project root so the structure matches either layout (both are auto-detected by `run_pipeline.py` and the notebook):
```
ML-Project3-Wildlife-ReID/
└── data/
    └── projectwild3/
        ├── reid_student_data_main/
        │   ├── train.csv                # Training sighting labels & metadata
        │   ├── test.csv                 # Test query sightings
        │   ├── miewid_ids.csv           # Sighting ID to embedding index mapping
        │   ├── miewid_features.npy      # Precomputed 2,152-dim MiewID descriptors
        │   └── sample_submission.csv    # Competition submission format reference
        ├── sift_features_train.npz/     # Directory of local keypoints (optional)
        └── sift_features_test.npz/      # Directory of local keypoints (optional)
```

### Step 3: Running the Standalone Pipeline
To run the full end-to-end pipeline, evaluate cross-validation across all three stages, and regenerate all CSVs:
```powershell
python run_pipeline.py
```
This outputs `submission_model1.csv`, `submission_model2.csv`, `submission_model3.csv`, and `submission.csv` (final best Model 3 predictions) directly in the project root.

### Step 4: Running the Notebook
To interactively explore the three model progression stages:
```powershell
jupyter notebook Wildlife_ReID_Pipeline.ipynb
```
Launch the notebook directly from the project root. All relative paths automatically resolve against `data/projectwild3/reid_student_data_main/` (or `data/` fallback).

---

## 5. Repository Structure

```
ML-Project3-Wildlife-ReID/
├── .gitignore                    # Ignores heavy .npy, .zip, venv, and scratch scripts
├── LICENSE                       # MIT License
├── README.md                     # Clean, authentic student project overview
├── requirements.txt              # Standard permitted packages (numpy, pandas, scipy, scikit-learn, lightgbm)
├── Wildlife_ReID_Pipeline.ipynb  # Reproducible notebook covering all 3 models
├── run_pipeline.py               # Standalone runner script for all 3 models
├── submission_model1.csv         # Model 1 baseline predictions (2,791 rows)
├── submission_model2.csv         # Model 2 viewpoint + threshold predictions (0.75821 Kaggle)
├── submission_model3.csv         # Model 3 Multi-Space Two-Stage GBDT predictions (0.8177 Kaggle)
└── submission.csv                # Final competition submission matching Model 3 (0.8177 Kaggle)
```

---

## 6. Course Rules Compliance Checklist

- [x] **No raw images used**: Strictly used the provided `miewid_features.npy` descriptors and metadata.
- [x] **No deep learning frameworks**: No PyTorch, TensorFlow, Keras, or neural network layers.
- [x] **No external data or pre-trained models**: Trained only on the provided training features.
- [x] **Permitted libraries only**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`.
- [x] **Correct submission format**: Exactly 2,791 rows matching `test.csv` IDs, valid labels, no NaNs.
