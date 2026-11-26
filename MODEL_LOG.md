# Wildlife Re-ID model log

Our final submission scored **0.84209 public MAP@5**. The three models in the notebook are cosine k-NN, viewpoint-aware k-NN, and the dual-branch GBDT with leopard SIFT verification. The intermediate versions below show how we got there.

## Model progression

| Stage | Main change | Result and decision |
| --- | --- | --- |
| **1. Cosine k-NN** | Normalize MiewID embeddings, retrieve within species, and put `new_individual` fifth. | **0.5518 local MAP@5**. Keep as the baseline and handle unseen animals earlier. |
| **2. Viewpoint and novelty thresholds** | Penalize opposite-flank hyena matches by 0.05. Use novelty cutoffs of 0.52 for hyenas and 0.54 for leopards. | **0.7714 local, 0.75821 public**. Keep as Model 2. Fixed cutoffs still struggle with sparse galleries. |
| Early learned reranking | Train LightGBM on candidate-pair features. | **0.78311 public**. Improve retrieval and give novelty its own model. |
| Multi-space GBDT revisions | Add LDA/PCA whitening, viewpoint prototypes, a separate novelty model, and k-reciprocal reranking. | **0.80794 public** (inductive), **0.81771** (blended), **0.82346** (clean k-reciprocal). Use these revisions as the basis for the backbone. |
| Dual-branch global backbone | Combine k-reciprocal retrieval with encounter-collapsed graph diffusion. Add identity-balanced LDA and subtract a gallery-estimated impostor bias. | **0.8643 development CV, 0.83547 public**. Keep both branches and compare later changes on a separate outer audit. |
| **3. Final Model 3 with leopard SIFT** | Verify up to six candidates using supplied RootSIFT and rescore the combined shortlist before selecting five. Model novelty, retrieved known identities, and missed known identities separately. | **0.84209 public**. Keep the leopard refinement and the original hyena predictions. Confirmation gained **+0.00574 weighted MAP@5**, with **+0.00508** on thinner galleries. |

The runner reports **0.8643** as backbone development CV and **0.84209** as the historical Kaggle public score.

## Experiment decisions

| Experiment | What we saw | Decision |
| --- | --- | --- |
| Pseudo-gallery expansion and alternate diffusion | Pseudo-label errors spread across the graph. Raw diffusion strength helped flag novel animals. | Keep the existing diffusion design. |
| Exact viewpoint prototypes, full impostor normalization, and dropping leading PCA components | Smaller viewpoint groups had fewer examples. Full normalization amplified some impostor matches, and dropping PCA components removed useful signal. | Keep broad flank groups, additive impostor bias, and regularization. |
| Larger galleries, gallery-support and singleton features, and calibration | The 80%-gallery variant scored **0.87068** versus **0.88094** on the same probe. The screened global challenger lost **0.00341 weighted MAP@5**. | Keep the original training and split recipe. |
| One-position SIFT rerank after selecting five | **+0.00172 weighted MAP@5**. This could only move labels already in the final five. | Verify candidates before selecting five. |
| Wider raw cosine, cross-view, and local descriptor retrieval | Candidate recall improved. The newly found true identities stayed outside the final five. | Keep the existing candidate pool. |
| VLAD scoring and hyena transfer | **+0.00012** and **+0.00082** weighted MAP@5, below the **+0.002** promotion threshold. | Keep the 0.84209 variant. |


- **Why we split by encounter.** Random image splits can put near-duplicate camera-trap frames on both sides. For known animals we hold out whole encounters, and for novel animals we hold out whole identities.
- **Why we moved SIFT earlier.** On 3,030 leopard query-fold rows, 199 known queries missed both shortlists and another 96 had the true identity shortlisted but outside the final five. Verifying before the cut gave those shortlisted identities a chance to move up.
- **What still limits the model.** Few gallery photos, opposite flanks, and missing the true identity at retrieval can all cause errors. Known encounters in the final confirmation were unused in the earlier audit. Some novel identities overlapped, so we also reported the gain on previously unused rows.
- **How to reproduce it.** The runner and standalone notebook generate all three model CSVs. The final file has 2,791 test IDs in order and five unique valid predictions each. A full local rerun matched the shipped CSV byte for byte.

For model code, setup, and the team's contribution summary, see [README.md](README.md) and [the project notebook](Wildlife_ReID_Pipeline.ipynb).
