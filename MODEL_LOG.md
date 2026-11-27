# Wildlife Re-ID model log

Our [final submission](submission.csv) scored **0.84780 public MAP@5**. We kept the encounter-disjoint validation protocol and built the final scorer on the dual-branch Model 3 backbone.

## How the model progressed

| Stage | What changed | Result and what we did next |
| --- | --- | --- |
| 1. Cosine k-NN | Normalised the MiewID embeddings, retrieved within species and always put `new_individual` fifth. | 0.5518 local MAP@5. Kept as the baseline; unseen animals needed to be ranked higher. |
| 2. Viewpoint and novelty thresholds | Added a 0.05 penalty for opposite-flank hyena matches, with novelty cutoffs of 0.52 (hyena) and 0.54 (leopard). | 0.7714 local, 0.75821 public. Kept as Model 2, though fixed cutoffs still struggle when an animal has few gallery photos. |
| Early learned re-ranking | Trained LightGBM on features of candidate pairs. | 0.78311 public. Next step was better retrieval and a separate novelty model. |
| Multi-space GBDT revisions | Added LDA/PCA whitening, viewpoint prototypes, a separate novelty model and k-reciprocal re-ranking. | Public scores of 0.80794 (inductive), 0.81771 (blended) and 0.82346 (clean k-reciprocal). These became the basis of the backbone. |
| Two-branch global backbone | Combined k-reciprocal retrieval with encounter-collapsed graph diffusion, added identity-balanced LDA and subtracted an impostor bias estimated from the gallery. | 0.8643 development CV, 0.83547 public. Kept both branches and judged later changes on a separate outer audit. |
| 3. Initial leopard SIFT refinement | Verified up to six candidates with the supplied RootSIFT, then rescored the whole shortlist before picking five. Novel, retrieved known and missed known cases are modelled separately. | **0.84209** public. Kept this for leopards and left the hyena predictions as they were. Confirmation showed +0.00574 weighted MAP@5, or +0.00508 with thinner galleries. |
| Gaussian identity scoring | Add encounter-balanced Gaussian proposals, verify the leading proposals, and fit a HistGBM candidate scorer. | **0.84604 public**. Keep the expanded leopard shortlist. |
| **Final Model 3 with Fisher vectors and LNBNN** | Add up to four candidates per local encoder. Use gallery-only PCA, a diagonal GMM for Fisher vectors, and a local descriptor index for LNBNN. Verify leading additional proposals for leopards. Blend the new heads with the existing leopard and hyena scores. | **0.84780 public**. Keep this file as the final submission. |

The script reports **0.8643** for backbone development CV. The **0.84780** number is the measured public score for the final CSV.

## What we tried

| Experiment | What we saw | Decision |
| --- | --- | --- |
| Pseudo-gallery expansion and alternative diffusion | Wrong pseudo-labels spread through the graph. Raw diffusion strength did help flag novel animals. | Kept the existing diffusion design. |
| Exact viewpoint prototypes, full impostor normalisation, dropping the leading PCA components | Smaller viewpoint groups had too few examples, full normalisation boosted some impostor matches, and dropping components lost useful signal. | Kept the broad flank groups, additive impostor bias and regularisation. |
| Larger galleries, gallery-support and singleton features, calibration | The 80% gallery version scored 0.87068, against 0.88094 on the same probe. The global challenger we screened lost 0.00341 weighted MAP@5. | Kept the original training and split setup. |
| SIFT re-ranking of one position after picking the top five | +0.00172 weighted MAP@5, but it could only reorder labels already in the top five. | Moved verification to before the top five are picked. |
| Wider raw cosine, cross-view and local descriptor retrieval | Candidate recall went up, but the newly found true identities still didn't reach the top five. | Kept the existing candidate pool. |
| VLAD scoring and transfer to hyenas | +0.00012 and +0.00082 weighted MAP@5, both under our +0.002 bar. | Kept the 0.84209 version. |
| Local encoders as extra features only | Adding Fisher and LNBNN scores without their proposed identities slightly reduced development MAP@5. | Let both encoders propose new identities before reranking. |
| Query-neighbor features, whitening and extra SIFT checks | The hyena query-neighbor trial gained **+0.00238** weighted MAP@5 in development and **+0.00037** on its reserved folds. The other changes had mixed or small gains. | Leave these changes out of the release. |
| Gallery-relative evidence and a learned encounter metric | Discounts for common visual matches, direct query support, and a gallery-trained pair distance gave small or inconsistent changes. Leopard query support lost two of three confirmation folds. | Keep the existing leopard ranking. |
| Calibrated known-versus-novel pairs | The hyena trial gained **+0.00261** weighted MAP@5 on confirmation folds with about 91% novel queries. It lost **0.00595** MAP@5 on known queries and had roughly zero weighted gain at a 45% novel mix. The team reported a public drop of about **0.005**. | Drop the last trial and keep **0.84780**. |

## Notes

### Why we split by encounter

A random image split can put near-identical camera-trap frames on both sides. So for known animals we hold out whole encounters, and for novel animals we hold out whole identities.

### Why we moved SIFT earlier

Out of 3,030 leopard query-fold rows, 199 known queries were missing from both shortlists. Another 96 had the correct identity shortlisted but not in the final five. Verifying before the cut gave those 96 a chance to move up.

### What still limits the model

Errors can still come from animals with few gallery photos, from opposite-flank views, and from the correct identity never being retrieved. The known encounters in the final confirmation hadn't been used in the earlier audit, but some novel identities overlapped, so we also reported the gain on previously unused rows.

### Reproducing the results

Both the script and the standalone notebook generate the CSVs for all three models. The final file has all 2,791 test IDs in order, each with five unique, valid predictions. A replay with the saved Phase 5 score tables reproduced every ranking. The standalone runner and notebook rebuild the scorer from the supplied features.

For the code, setup steps and team contributions, see [README.md](README.md) and [the notebook](Wildlife_ReID_Pipeline.ipynb).
