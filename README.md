# Wildlife Re-Identification (Problem Statement 3)
## Machine Learning Course Project - IIIT Delhi

This repository contains our team's complete, reproducible code and final submissions for the **Wildlife Re-Identification** Kaggle competition.

---

## Team Members

| Name | Roll number | Email |
| --- | --- | --- |
| **R S Maneesh** (Group Representative) | 2024449 | maneesh24449@iiitd.ac.in |
| **Raunak Mishra** | 2024455 | raunak24455@iiitd.ac.in |
| **Rishabh Khannawalia** | 2024463 | rishabh24463@iiitd.ac.in |
| **Shakti** | 2024519 | shakti24519@iiitd.ac.in |

## 1. Project Overview & Approach

The goal is to identify individual spotted hyenas and leopards from camera-trap sightings using supplied 2,152-dimensional MiewID embeddings and precomputed SIFT descriptors. The test set has 2,791 images. An animal can either belong to an individual seen during training (of the same species) or be an unseen animal labeled as `new_individual`. Evaluation is strictly on **MAP@5**.

### Key Pipeline Components
1. **Feature Normalization & Species Separation**: All MiewID descriptors are L2-normalized onto the unit hypersphere so dot products compute exact cosine similarity. Retrieval runs strictly within species (`hyena` vs `leopard`).
2. **Encounter-Disjoint Open-Set Validation**:
   - Standard random image splits leak same-burst photos (`39.8%` of leopard queries share a `>= 0.80` gallery twin locally vs only `8.9%` on Kaggle test).
   - We cluster same-identity burst frames (`cos >= 0.82`) into encounters, hold out ~25–26% of identities completely as `new_individual`, and hold out entire encounters for known queries. Model 3 uses five repeated development splits plus a separate three-fold confirmation audit for the leopard refinement.
3. **Viewpoint Flank Asymmetry & View-Split Prototypes**:
   - Spotted hyenas and leopards have asymmetric coat markings across left and right flanks.
   - Model 2 applies an opposite-flank penalty of `-0.05` for hyenas. Model 3 builds separate `left`, `right`, and `neutral` identity prototypes.
4. **Dual-Branch Manifold Diffusion + k-Reciprocal Retrieval & Two-Stage GBDT**:
   - **Branch 1 (`kRec`)**: Shrinkage LDA (`lda_sample`), gallery PCA whitening, viewpoint-split centroids, CSLS hubness suppression, balanced kernel ridge separation, inductive $\alpha$-QE/DBA, and raw k-reciprocal Jaccard re-ranking (`k1=8, k2=2, lam=0.55`).
   - **Branch 2 (`Diff`)**: Identity-balanced LDA (`lda_bal`) + transductive PCA (`[G; Q]`), impostor max-bias debiasing ($-0.35 \cdot P_{98}$), encounter-collapsed k-reciprocal (`k1=10`), and **Sparse Manifold Graph Diffusion** ($(I - \alpha P) Z = (1 - \alpha) Y$) with a **Bhattacharyya Diffusion Kernel** ($\cos(\sqrt{Z_q}, \sqrt{Z_g})$) and same-burst query pooling (`cos >= 0.82`).
   - A Stage-A ensemble (`LGBMClassifier`, `HistGradientBoostingClassifier`, `LGBMRanker`) scores candidates across both branches, a Stage-B novelty ensemble estimates $P(\text{new\_individual} \mid q)$, and Bayes Expected MAP@5 decoding blends Diffusion (`65%` Hyena / `50%` Leopard) with k-Reciprocal (`35%` Hyena / `50%` Leopard). This global backbone scored **0.83547** on Kaggle.
   - The final Model 3 verifies up to six leopard candidates using RootSIFT and up to two compatible gallery encounters per candidate. A regularized correction adjusts the full candidate union, and a three-outcome model estimates novel, known covered, and known missed. The final submission scored **0.84209** on Kaggle.

---

## 2. Model Progression

| Model | Main change | Hyena CV | Leopard CV | Weighted CV | Kaggle public MAP@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| **1. Cosine k-NN** | Retrieve normalized embeddings and put `new_individual` fifth. | 0.5765 | 0.5327 | 0.5518 | — |
| **2. Viewpoint-aware k-NN** | Add a hyena opposite-flank penalty and species-specific novelty thresholds. | 0.7898 | 0.7571 | 0.7714 | 0.75821 |
| **3. Dual-branch GBDT + SIFT** | Blend diffusion and k-reciprocal scores, then verify leopard candidates and account for shortlist coverage. | 0.9018 | 0.8353 | 0.8643 | **0.84209** |

Model 3 CV shows the global backbone, which scored **0.83547** on Kaggle before the leopard SIFT refinement.

In a separate nested encounter-disjoint audit, the leopard refinement gained **+0.00574 weighted MAP@5** and **+0.00508** with thinner galleries. The three leopard fold gains were +0.01314, +0.01236, and +0.00503. Known query encounters were unused in the earlier audit. Novel identities partly overlapped it.

See [MODEL_LOG.md](MODEL_LOG.md) for intermediate versions, rejected experiments, and demo notes.

## 3. Individual Contributions Breakdown

* **R S Maneesh (2024449)** — *Group Representative*:
  * Team lead, repository hosting, and group coordination.
  * Conducted Exploratory Data Analysis (EDA) on species distributions, identity frequencies, and class imbalance (noting that 99 leopard identities are singletons with only 1 photo).
  * Built and benchmarked **Model 1** (`submission_model1.csv`) using L2-normalized cosine k-NN with naive Rank-5 `new_individual`.
  * Performed error analysis on Model 1 showing that failing to prioritize unseen animals drops encounter-disjoint MAP@5 to `0.5518`.

* **Raunak Mishra (2024455)**:
  * Formulated and implemented **Model 3** (`submission_model3.csv` / `submission.csv`): identity-balanced & multi-space LDA/PCA whitening, impostor max-bias debiasing, encounter-collapsed sparse manifold diffusion with Bhattacharyya kernel scoring, k-reciprocal re-ranking, and two-stage GBDT ensembles (`LGBMClassifier`, `HistGradientBoostingClassifier`, `LGBMRanker`, `GradientBoostingClassifier`).
  * Replaced fixed heuristic novelty thresholds with Stage-B posterior estimation, query-burst pooling, and calibrated Bayes Expected MAP@5 list decoding.
  * Refactored the end-to-end `run_pipeline.py` script and `Wildlife_ReID_Pipeline.ipynb` notebook, raising our Kaggle public score from `0.75821` to **0.835**.

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
- **Python**: Use Python `3.11` or newer. The final verification environment used Python 3.14.4, NumPy 2.5.2, pandas 3.0.5, SciPy 1.17.1, scikit-learn 1.9.0, and LightGBM 4.7.0 on Windows.
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
Extract the competition dataset into the project root. The runner and notebook support flat `data/`, `data/projectwild3/reid_student_data_main/`, and `projectwild3/reid_student_data_main/`. For another location, such as a Kaggle input mount, set `REID_DATA_DIR` to the directory that contains `train.csv`.
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
        ├── sift_features_train.npz/     # Extracted SIFT arrays for the 0.84209 model
        └── sift_features_test.npz/      # Extracted SIFT arrays for the 0.84209 model
```

### Step 3: Running the Standalone Pipeline
To run the full end-to-end pipeline, evaluate cross-validation across all three stages, and regenerate all CSVs:
```powershell
python run_pipeline.py
```
This outputs `submission_model1.csv`, `submission_model2.csv`, `submission_model3.csv`, and `submission.csv` (final best Model 3 predictions) directly in the project root.
Set `REID_OUTPUT_DIR` to write the regenerated CSVs to a separate directory. Both SIFT stores must be extracted directories with per-image `_xy.npy` and `_desc.npy` files for the **0.84209** result. If both stores are absent, the runner prints that it is using the **0.83547** global-only Model 3.

### Step 4: Running the Notebook
To interactively explore the three model progression stages:
```powershell
jupyter notebook Wildlife_ReID_Pipeline.ipynb
```
Launch the notebook from the project root, or set `REID_DATA_DIR` to the mounted competition input. It includes the same three-model code and final leopard refinement as the runner.

---

## 5. Repository Files

- [`run_pipeline.py`](run_pipeline.py) runs all three models using [`sift_rerank.py`](sift_rerank.py) and the packages in [`requirements.txt`](requirements.txt).
- [`Wildlife_ReID_Pipeline.ipynb`](Wildlife_ReID_Pipeline.ipynb) explains and runs the same models.
- [`MODEL_LOG.md`](MODEL_LOG.md) records experiment results and decisions for the demo.
- [`submission_model1.csv`](submission_model1.csv) and [`submission_model2.csv`](submission_model2.csv) contain the earlier predictions.
- [`submission.csv`](submission.csv) and [`submission_model3.csv`](submission_model3.csv) contain the same 0.84209 predictions.

## 6. Course Rules Compliance Checklist

- [x] **No raw images used**: The pipeline uses only the supplied MiewID embeddings, SIFT arrays and metadata.
- [x] **No deep learning frameworks**: No PyTorch, TensorFlow, Keras, or neural network layers.
- [x] **No external data or pre-trained models**: Trained only on the provided training features.
- [x] **Permitted libraries only**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`.
- [x] **Correct submission format**: Exactly 2,791 rows matching `test.csv` IDs, valid labels, no NaNs.
