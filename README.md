# Wildlife Re-Identification (Problem Statement 3)

Machine Learning course project, IIIT Delhi.

Our final [submission.csv](submission.csv) scored **0.84780 public MAP@5**. The runner and notebook use the supplied competition features to build the three models.

## Team

| Name | Roll number | Email |
| --- | --- | --- |
| R S Maneesh (Group Representative) | 2024449 | maneesh24449@iiitd.ac.in |
| Raunak Mishra | 2024455 | raunak24455@iiitd.ac.in |
| Rishabh Khannawalia | 2024463 | rishabh24463@iiitd.ac.in |
| Shakti | 2024519 | shakti24519@iiitd.ac.in |

## The problem

We have to identify individual spotted hyenas and leopards from camera-trap sightings. We're given 2152-dimensional MiewID embeddings and precomputed SIFT descriptors. The test set has 2791 images, and each one is either an individual seen in training (of the same species) or an unseen animal, which should be labelled `new_individual`. The metric is MAP@5.

## Our approach

### Basics

We L2 normalise all MiewID descriptors, so a dot product gives cosine similarity. Hyenas and leopards are always retrieved separately.

### Validation

A random image split doesn't work well here because photos from the same burst end up on both sides. Locally, 39.8% of leopard queries had a gallery twin with similarity of at least 0.80, compared with only 8.9% on the Kaggle test set.

To get around this, we group frames of the same animal into encounters (cosine ≥ 0.82). We then hold out about 25–26% of identities completely to act as `new_individual`, and hold out whole encounters for the known queries. For Model 3 we used five repeated development splits, plus a separate three-fold audit to confirm the leopard refinement.

### Left and right flanks

Hyenas and leopards have different coat markings on their left and right sides. Model 2 handles this with a -0.05 penalty for opposite flank matches in hyenas. Model 3 instead builds separate `left`, `right` and `neutral` prototypes for each identity.

### Model 3

Model 3 combines two retrieval branches.

**k-reciprocal branch (`kRec`)**
- Shrinkage LDA (`lda_sample`) and PCA whitening fitted on the gallery
- Viewpoint-split centroids
- CSLS hubness suppression and balanced kernel ridge separation
- Inductive α-QE/DBA
- k-reciprocal Jaccard re-ranking (`k1=8, k2=2, lam=0.55`)

**Diffusion branch (`Diff`)**
- Identity-balanced LDA (`lda_bal`) and transductive PCA on gallery + queries (`[G; Q]`)
- Impostor max-bias debiasing (-0.35 · P98)
- Encounter-collapsed k-reciprocal (`k1=10`)
- Sparse graph diffusion, `(I - αP) Z = (1 - α) Y`, scored with a Bhattacharyya kernel, `cos(√Z_q, √Z_g)`
- Pooling of queries from the same burst (cosine ≥ 0.82)

**Putting them together**

A first-stage ensemble (`LGBMClassifier`, `HistGradientBoostingClassifier`, `LGBMRanker`) scores candidates from both branches. A second stage ensemble estimates the probability that a query is a new individual. We then decode the top-5 list by maximising expected MAP@5, blending the two branches as follows:

| Species | Diffusion | k-reciprocal |
| --- | ---: | ---: |
| Hyena | 65% | 35% |
| Leopard | 50% | 50% |

This global model scored **0.83547** on Kaggle.

**SIFT check for leopards**

In the first local refinement, we verify up to six leopard candidates with RootSIFT, using up to two compatible gallery encounters per candidate. A regularised correction then adjusts the full candidate set, and a three outcome model estimates whether the query is novel, known and covered by the shortlist, or known but missed. This brought our score to **0.84209**.

**Final local-descriptor scorer**

We add encounter-balanced Gaussian identity scores and up to four proposals each from Fisher vectors and LNBNN. The encoders fit on each fold's gallery. Leopard gets extra SIFT verification for leading proposals. Hyena uses the dual-branch scores as its anchor. We blend that evidence with a gradient-boosted candidate scorer and a three-outcome novelty model trained on encounter-disjoint inner folds.

## Results

| Model | Main change | Hyena CV | Leopard CV | Weighted CV | Kaggle public MAP@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1. Cosine k-NN | Nearest neighbours, with `new_individual` always fifth | 0.5765 | 0.5327 | 0.5518 | — |
| 2. Viewpoint-aware k-NN | Hyena flank penalty and per-species novelty thresholds | 0.7898 | 0.7571 | 0.7714 | 0.75821 |
| 3. Final two-branch GBDT | Gaussian identity scores, SIFT, Fisher vectors and LNBNN | 0.9018 | 0.8353 | 0.8643 | **0.84780** |

The Model 3 CV figures describe the global backbone, which scored **0.83547** on Kaggle. The leopard SIFT release scored **0.84209**, and the Gaussian candidate scoring and verification release scored **0.84604**.

We tested the leopard refinement separately in a nested encounter disjoint audit. It gained +0.00574 weighted MAP@5, and +0.00508 with thinner galleries. The gains on the three leopard folds were +0.01314, +0.01236 and +0.00503. The known query encounters hadn't been used in the earlier audit, though the novel identities partly overlapped with it.

Intermediate versions, rejected experiments and demo notes are in [MODEL_LOG.md](MODEL_LOG.md).

## Who did what

**R S Maneesh (2024449), Group Representative**
- Led the team, hosted the repository and coordinated the group
- Did the EDA on species distribution, identity frequencies and class imbalance (for example, 99 leopard identities have only one photo)
- Built and benchmarked Model 1 (`submission_model1.csv`)
- Analysed its errors and showed that not prioritising unseen animals leaves encounter-disjoint MAP@5 at 0.5518

**Raunak Mishra (2024455)**
- Designed and implemented the Model 3 backbone including LDA/PCA whitening, debiasing, diffusion, k-reciprocal re-ranking and two-stage GBDT ensembles (including `GradientBoostingClassifier`)
- Replaced the fixed novelty thresholds with estimated probabilities, query burst pooling and expected MAP@5 decoding
- Refactored `run_pipeline.py` and the notebook, taking our public score from 0.75821 to 0.835

**Rishabh Khannawalia (2024463)**
- Found the burst leakage in random splits and built the encounter-disjoint 3-fold open set validation
- Implemented and tuned Model 2's novelty thresholds (0.52 for hyenas, 0.54 for leopards)
- Checked the submissions (2,791 rows, valid same species IDs, no nulls) and that whitening respected fold boundaries

**Shakti (2024519)**
- Audited the viewpoint metadata and verified the left/right flank asymmetry in both species
- Implemented Model 2's flank penalty and Model 3's viewpoint split prototypes
- Analysed where Model 2 failed (background lighting, suppressed singletons, flat top-5 confidence), which motivated Model 3

## Running the code

### 1. Set up the environment

You need Python 3.11 or newer. Our final verification was on Windows with Python 3.14.4, NumPy 2.5.2, pandas 3.0.5, SciPy 1.17.1, scikit-learn 1.9.0 and LightGBM 4.7.0.

```powershell
python -m venv .venv311
.venv311\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the data

The dataset (`data/`) and virtual environments (`.venv*`) aren't committed, because of course rules and repo size limits. Extract the competition data into the project root like this:

```
ML-Project3-Wildlife-ReID/
└── data/
    └── projectwild3/
        ├── reid_student_data_main/
        │   ├── train.csv                # training labels and metadata
        │   ├── test.csv                 # test queries
        │   ├── miewid_ids.csv           # sighting ID to embedding index
        │   ├── miewid_features.npy      # 2,152-dim MiewID descriptors
        │   └── sample_submission.csv    # submission format
        ├── sift_features_train.npz/     # extracted per-image SIFT arrays
        └── sift_features_test.npz/
```

A flat `data/` folder, `data/projectwild3/reid_student_data_main/` and `projectwild3/reid_student_data_main/` all work. If your data is somewhere else (such as a Kaggle input mount), set `REID_DATA_DIR` to the folder containing `train.csv`.

### 3. Run the pipeline

```powershell
python run_pipeline.py
```

This cross-validates all three models and writes `submission_model1.csv`, `submission_model2.csv`, `submission_model3.csv` and `submission.csv` to `outputs/`. Set `REID_OUTPUT_DIR` to use another destination.

Both SIFT stores must be extracted folders with `_xy.npy` and `_desc.npy` files for each image. Set `REID_MODEL3=sift` to run the historical **0.84209** version. If both stores are missing, the script uses the global-only Model 3.

The checked-in `submission.csv` contains the exact scored predictions. Its SHA-256 is `b9e7ab82f8fb98f1fe2ad8d8909068c751b39223d98f1d0b9e1eb7f1d4955011`.

### 4. Or use the notebook

```powershell
jupyter notebook Wildlife_ReID_Pipeline.ipynb
```

Start it from the project root, or set `REID_DATA_DIR`. It has the same three models and final local-descriptor scorer as the script.

## Files

| File | What it is |
| --- | --- |
| [`run_pipeline.py`](run_pipeline.py) | Runs all three models |
| [`sift_rerank.py`](sift_rerank.py) | SIFT re-ranking used by the pipeline |
| [`final_rerank.py`](final_rerank.py), [`local_encoders.py`](local_encoders.py) | Final candidate scoring and local descriptor encoders |
| [`requirements.txt`](requirements.txt) | Dependencies |
| [`Wildlife_ReID_Pipeline.ipynb`](Wildlife_ReID_Pipeline.ipynb) | Walkthrough of the same models |
| [`MODEL_LOG.md`](MODEL_LOG.md) | Experiment results and decisions |
| [`submission_model1.csv`](submission_model1.csv), [`submission_model2.csv`](submission_model2.csv) | Earlier predictions |
| [`submission.csv`](submission.csv), [`submission_model3.csv`](submission_model3.csv) | Final predictions (identical, 0.84780) |

## Course rules

- We used only the supplied MiewID embeddings, SIFT arrays and metadata, with no raw images.
- No deep learning frameworks or neural network layers (no PyTorch, TensorFlow or Keras).
- No external data or pretrained models; everything is trained on the provided training features.
- Libraries used are from the permitted list: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`.
- The submission has exactly 2791 rows matching the IDs in `test.csv`, with valid labels and no NaNs.
