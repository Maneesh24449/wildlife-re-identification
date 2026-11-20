# Wildlife Re-Identification (Problem Statement 3)
## Machine Learning Course Project - IIIT Delhi

This repository contains our team's complete, reproducible code and final submissions for the **Wildlife Re-Identification** Kaggle competition.

---

## Team Members

| Name | Roll Number | Email | Role |
| :--- | :---: | :--- | :--- |
| **R S Maneesh** (Group Representative) | **2024449** | maneesh24449@iiitd.ac.in | Model 1 (Baseline k-NN), Exploratory Data Analysis |
| **Raunak Mishra** | **2024455** | raunak24455@iiitd.ac.in | Model 3 (GBDT Ranker), Pipeline Architecture, Notebook |
| **Rishabh Khannawalia** | **2024463** | rishabh24463@iiitd.ac.in | Open-Set Validation Protocol, Threshold Tuning |
| **Shakti** | **2024519** | shakti24519@iiitd.ac.in | Viewpoint Analysis, Flank Penalty, Error Auditing |

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
   - Top candidates are re-ranked using pairwise features and a LightGBM classifier to produce our best competition score (**0.78151**).

---

## 2. Model Evolution & Results (The 3 Model Stages)

As required by the project instructions, we developed three distinct model versions showing clear progression:

| Model Version | Description | Key Additions | Local CV MAP@5 | Kaggle Public Score |
| :--- | :--- | :--- | :---: | :---: |
| **Model 1: Baseline** | Cosine Similarity k-NN | L2-normalization, nearest neighbors, naive rank-5 `new_individual` | 0.5834 | — |
| **Model 2: Enhanced k-NN** | Viewpoint + Calibrated Thresholding | Flank asymmetry penalty, binary decision boundary for novel animals | 0.8305 | 0.75821 |
| **Model 3: Pairwise GBDT** | LightGBM Re-Ranker Fusion | 11 pairwise interaction features, learned match probability | **0.8667** | **0.78151** (Best) |

*\*Note: Model 1 was evaluated on local CV; to preserve daily submission limits, only our enhanced pipelines (Model 2 and Model 3) were submitted to Kaggle.*

---

## 3. Individual Contributions Breakdown

* **R S Maneesh (2024449)** — *Group Representative*:
  * Team lead and group coordination.
  * Conducted Exploratory Data Analysis (EDA) on species distributions, identity frequencies, and class imbalance (noting that 99 leopard identities are singletons with only 1 photo).
  * Built and benchmarked **Model 1** (Baseline Cosine Similarity k-NN with naive Rank-5 `new_individual`).
  * Performed error analysis on Model 1 to show that failing to detect novel animals degraded the score to ~0.58.

* **Raunak Mishra (2024455)**:
  * Repository configuration, pipeline modularization, and Git version control setup.
  * Formulated and implemented **Model 3**: engineered 11 pairwise interaction features (`s_max`, `s_mean`, `diff_to_top1`, `top1_top2_gap`, etc.) and trained the LightGBM classifier across 3 cross-validation folds.
  * Generated and verified the final competition submission achieving our peak score of **0.78151** on Kaggle.

* **Rishabh Khannawalia (2024463)**:
  * Designed the realistic open-set 3-fold cross-validation split that holds out 20% of identities entirely to simulate unseen `new_individual` test queries.
  * Implemented and tuned the decision thresholds for **Model 2** ($T = 0.52$ for Hyena, $0.54$ for Leopard) to accurately identify novel animals at Rank 1.
  * Benchmarked threshold sensitivities, proving that calibrated binary boundaries outperform multi-stage hedging.

* **Shakti (2024519)**:
  * Audited viewpoint metadata and identified that left and right flank coat patterns in spotted hyenas and leopards are biologically asymmetric.
  * Implemented the opposite-flank penalty matrix to suppress cross-viewpoint background false positives.
  * Conducted error analysis on candidate coherence across multi-image identities, demonstrating that single-image outlier similarities create false positives that require coherence verification.

---

## 4. Repository Structure

```
RounakMishra03/ML-Project3-Wildlife-ReID/
├── .gitignore                    # Ignores heavy .npy, .zip, venv, and scratch scripts
├── README.md                     # Clean, authentic student project overview
├── requirements.txt              # Standard permitted packages (numpy, pandas, scipy, scikit-learn, lightgbm)
├── Wildlife_ReID_Pipeline.ipynb  # The core, human-written reproducible notebook covering all 3 models
├── run_pipeline.py               # Standalone runner script
└── submission.csv                # Final top-scoring Kaggle submission (0.78151)
```

---

## 5. Course Rules Compliance Checklist

- [x] **No raw images used**: Strictly used the provided `miewid_features.npy` descriptors and metadata.
- [x] **No deep learning frameworks**: No PyTorch, TensorFlow, Keras, or neural network layers.
- [x] **No external data or pre-trained models**: Trained only on the provided training features.
- [x] **Permitted libraries only**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`.
- [x] **Correct submission format**: Exactly 2,791 rows matching `test.csv` IDs, valid labels, no NaNs.
