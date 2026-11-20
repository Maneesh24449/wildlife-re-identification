# Wildlife Re-Identification (Problem Statement 3)
## Machine Learning Course Project - IIIT Delhi

This repository contains our team's complete, reproducible code and final submissions for the **Wildlife Re-Identification** Kaggle competition.

---

## 1. Project Overview & Approach

The goal is to identify individual spotted hyenas and leopards from camera-trap sightings using precomputed 2,152-dimensional MiewID feature descriptors. The test set has 2,791 images. An animal can either belong to an individual seen during training (of the same species) or be an unseen animal labeled as `new_individual`. Evaluation is strictly on **MAP@5**.

### Key Pipeline Components:
1. **Feature Normalization**: All MiewID descriptors are L2-normalized so their dot product directly gives cosine similarity. Retrieval is strictly separated by species (hyenas only compare to hyenas, leopards to leopards).
2. **Open-Set Validation Split**:
   - We hold out 20% of training identities completely to simulate `new_individual` queries.
   - For remaining identities, we hold out 30% of their images as queries.
   - Evaluated across 3 distinct random seeds (42, 100, 2024).
3. **Flank Viewpoint Penalty**:
   - Spotted hyenas and leopards have asymmetric spot patterns on their left vs right sides.
   - An opposite-flank penalty (-0.05) reduces spurious background matches across different sides.
4. **Calibrated Thresholding & LightGBM Re-Ranking**:
   - If similarity is below a calibrated threshold, `new_individual` is placed at Rank 1.
   - Top candidates are re-ranked using pairwise features and a LightGBM classifier to produce our best competition score.

---

## 2. Model Evolution & Results (The 3 Model Stages)

As required by the project instructions, we developed three distinct model versions showing clear progression:

| Model Version | Description | Key Additions | Local CV MAP@5 | Kaggle Public Score |
| :--- | :--- | :--- | :---: | :---: |
| **Model 1: Baseline** | Cosine Similarity k-NN | L2-normalization, nearest neighbors, naive rank-5 `new_individual` | 0.5834 | ~0.58 |
| **Model 2: Enhanced k-NN** | Viewpoint + Calibrated Thresholding | Flank asymmetry penalty, binary decision boundary for novel animals | 0.8305 | 0.75821 |
| **Model 3: Pairwise GBDT** | LightGBM Re-Ranker Fusion | 11 pairwise interaction features, learned match probability | **0.8667** | **0.78151** (Best) |

---

## 3. Repository Structure

```
├── README.md                     # Project overview and instructions
├── PROJECT_REPORT.md             # Detailed report with EDA, error analysis, and viva prep
├── Wildlife_ReID_Pipeline.ipynb  # Main end-to-end reproducible notebook
├── run_pipeline.py               # Complete standalone script running all stages
├── eda.py                        # Exploratory data analysis script
├── test_baseline.py              # Model 1 baseline script
├── tune_multi_threshold.py       # Model 2 threshold calibration script
├── test_lightgbm_ranker.py       # Model 3 LightGBM pairwise ranker
└── submission.csv                # Final top-scoring submission file (0.78151)
```

---

## 4. How to Run

### Setup Environment:
```bash
pip install numpy pandas scipy scikit-learn lightgbm
```

### Run Full Pipeline:
```bash
python run_pipeline.py
```
* Execution time: ~14 seconds on standard CPU.
* Output: Generates `submission.csv` (2,791 rows matching test set IDs).

---

## 5. Course Rules Compliance Checklist

- [x] **No raw images used**: Strictly used the provided `miewid_features.npy` descriptors and metadata.
- [x] **No deep learning frameworks**: No PyTorch, TensorFlow, Keras, or neural network layers.
- [x] **No external data or pre-trained models**: Trained only on the provided training features.
- [x] **Permitted libraries only**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`.
- [x] **Correct submission format**: Exactly 2,791 rows matching `test.csv` IDs, valid labels, no NaNs.
