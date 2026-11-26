"""
Wildlife Re-Identification (Problem Statement 3)
Reproducible End-to-End Pipeline & Submission Generator

Builds and evaluates three progressive classical ML / GBDT model stages:
  - Model 1: L2-Normalized Cosine k-NN Baseline (naive Rank-5 new_individual)
  - Model 2: Viewpoint-Aware Cosine k-NN + Calibrated Thresholding (Kaggle LB: 0.75821)
  - Model 3: Dual-Branch GBDT + Bayes MAP@5 with leopard SIFT refinement

Strict Rules Compliance:
  Uses only supplied MiewID features, SIFT descriptors and metadata with numpy, pandas, scipy,
  scikit-learn, and lightgbm. Zero deep learning frameworks or external datasets.
"""

import os

# Keep OpenMP / BLAS thread count bounded so LightGBM and sparse/dense ops run smoothly
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from sift_rerank import find_sift_directory, rerank_leopards

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent
data_override = os.environ.get("REID_DATA_DIR")
dataset_locations = [Path(data_override)] if data_override else [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "data" / "projectwild3" / "reid_student_data_main",
    PROJECT_ROOT / "projectwild3" / "reid_student_data_main",
]
DATA_DIR = next((path for path in dataset_locations if (path / "train.csv").is_file()), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Dataset not found. Set REID_DATA_DIR to the directory containing train.csv "
        "or extract the supplied dataset under data/ or projectwild3/."
    )
OUTPUT_DIR = Path(os.environ.get("REID_OUTPUT_DIR", PROJECT_ROOT))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("=" * 72)
print(" WILDLIFE RE-IDENTIFICATION: REPRODUCIBLE 3-MODEL PIPELINE")
print("=" * 72)


# ---------------------------------------------------------------------------
# 1. Data Loading & Normalization
# ---------------------------------------------------------------------------
print("\n[1/5] Loading and aligning MiewID embeddings...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
mids_df = pd.read_csv(DATA_DIR / "miewid_ids.csv")
raw_features = np.load(DATA_DIR / "miewid_features.npy")

norms = np.linalg.norm(raw_features, axis=1, keepdims=True)
norms[norms == 0] = 1.0
features = (raw_features / norms).astype(np.float32)

id_to_idx = {val: i for i, val in enumerate(mids_df["id"])}
train_df["feat_idx"] = train_df["id"].map(id_to_idx).astype(int)
test_df["feat_idx"] = test_df["id"].map(id_to_idx).astype(int)

for sp in ["hyena", "leopard"]:
    sp_tr = train_df[train_df["species"] == sp]
    sp_te = test_df[test_df["species"] == sp]
    print(f"  [{sp.upper()}] Train: {len(sp_tr)} images ({sp_tr['identity'].nunique()} IDs) | Test: {len(sp_te)} images")


# ---------------------------------------------------------------------------
# 2. Evaluation Metric & Encounter-Disjoint Open-Set Validation
# ---------------------------------------------------------------------------
def map5_score(predictions, ground_truths):
    total = 0.0
    for preds, gt in zip(predictions, ground_truths):
        for rank, pred in enumerate(preds[:5], start=1):
            if pred == gt:
                total += 1.0 / rank
                break
    return total / len(ground_truths) if len(ground_truths) > 0 else 0.0


def assign_encounters(sp_df, burst_thr=0.82):
    """Group camera-trap burst photos of the same ID into discrete encounters."""
    enc_ids = np.zeros(len(sp_df), dtype=int)
    cur_enc = 0
    for _, grp in sp_df.groupby("identity"):
        idxs = grp.index.values
        if len(idxs) == 1:
            enc_ids[idxs[0]] = cur_enc
            cur_enc += 1
        else:
            F = features[sp_df.loc[idxs, "feat_idx"].values]
            adj = (F @ F.T) >= burst_thr
            n_comp, labels = connected_components(adj, directed=False)
            for li, gi in enumerate(idxs):
                enc_ids[gi] = cur_enc + labels[li]
            cur_enc += n_comp
    return enc_ids


def make_open_set_split(sp_df, holdout_id_ratio=0.25, query_enc_ratio=0.38, max_per_id=10, seed=42):
    """
    Hold out entire identities for new_individual queries and hold out entire encounters
    (rather than individual burst frames) so CV mirrors unseen test encounters.
    """
    df = sp_df if "enc" in sp_df.columns else sp_df.assign(enc=assign_encounters(sp_df))
    rng = np.random.RandomState(seed)

    enc_counts = df.groupby("identity")["enc"].nunique()
    buckets = [
        enc_counts[enc_counts == 1].index.values,
        enc_counts[(enc_counts >= 2) & (enc_counts <= 4)].index.values,
        enc_counts[enc_counts > 4].index.values,
    ]
    new_ids = set()
    for grp in buckets:
        if len(grp) > 0:
            shuf = rng.permutation(grp)
            k = max(1, int(np.round(len(grp) * holdout_id_ratio)))
            new_ids.update(shuf[:k])

    gal_idx, q_idx, q_labs = [], [], []
    for identity, grp in df.groupby("identity"):
        if identity in new_ids:
            idxs = grp.index.values
            if len(idxs) > max_per_id:
                idxs = rng.choice(idxs, size=max_per_id, replace=False)
            q_idx.extend(idxs)
            q_labs.extend(["new_individual"] * len(idxs))
        else:
            u_encs = grp["enc"].unique()
            if len(u_encs) == 1:
                gal_idx.extend(grp.index.values)
            else:
                shuf_encs = rng.permutation(u_encs)
                nq = min(len(u_encs) - 1, max(1, int(np.round(len(u_encs) * query_enc_ratio))))
                q_encs = set(shuf_encs[:nq])
                id_q = []
                for idx in grp.index.values:
                    if df.loc[idx, "enc"] in q_encs:
                        id_q.append(idx)
                    else:
                        gal_idx.append(idx)
                if len(id_q) > max_per_id:
                    id_q = list(rng.choice(id_q, size=max_per_id, replace=False))
                q_idx.extend(id_q)
                q_labs.extend([identity] * len(id_q))

    return df.loc[gal_idx].reset_index(drop=True), df.loc[q_idx].reset_index(drop=True), np.array(q_labs)


def flank_side(v_series):
    arr = v_series.fillna("").to_numpy(dtype=str)
    out = np.zeros(len(arr), dtype=int)
    for i, val in enumerate(arr):
        if "left" in val:
            out[i] = -1
        elif "right" in val:
            out[i] = 1
    return out


def validate_submission(sub_df, filename):
    assert len(sub_df) == 2791, f"Expected 2791 rows, got {len(sub_df)}"
    assert sub_df["id"].nunique() == 2791, "Duplicate IDs found!"
    assert sub_df["id"].equals(test_df["id"]), "Submission IDs must follow test.csv order"
    assert not sub_df["predictions"].isna().any(), "Missing predictions found!"
    valid_h = set(train_df[train_df["species"] == "hyena"]["identity"].unique()) | {"new_individual"}
    valid_l = set(train_df[train_df["species"] == "leopard"]["identity"].unique()) | {"new_individual"}
    id2sp = dict(zip(test_df["id"], test_df["species"]))
    for _, row in sub_df.iterrows():
        preds = row["predictions"].split()
        assert len(preds) == 5 and len(set(preds)) == 5
        allowed = valid_h if id2sp[row["id"]] == "hyena" else valid_l
        for p in preds:
            assert p in allowed
    out_path = OUTPUT_DIR / filename
    sub_df.to_csv(out_path, index=False)
    print(f"  -> Validated and saved {filename} ({len(sub_df)} rows)")


# ---------------------------------------------------------------------------
# 3. Model 1: Baseline Cosine Similarity k-NN
# ---------------------------------------------------------------------------
print("\n[2/5] Running Model 1 (Baseline Cosine k-NN, Rank-5 new_individual)...")
m1_cv = {}
m1_preds = {}

for sp, hr in [("hyena", 0.25), ("leopard", 0.27)]:
    sp_df = train_df[train_df["species"] == sp].reset_index(drop=True)
    sp_df["enc"] = assign_encounters(sp_df)
    fold_sc = []
    for seed in [42, 100, 2024]:
        gal, que, labs = make_open_set_split(sp_df, holdout_id_ratio=hr, query_enc_ratio=0.45, seed=seed)
        S = features[que["feat_idx"].values] @ features[gal["feat_idx"].values].T
        uids, inv = np.unique(gal["identity"].values, return_inverse=True)
        id_sc = np.column_stack([S[:, inv == c].max(axis=1) for c in range(len(uids))])
        top4 = np.argsort(-id_sc, axis=1)[:, :4]
        preds = [list(uids[r]) + ["new_individual"] for r in top4]
        fold_sc.append(map5_score(preds, labs))
    m1_cv[sp] = np.mean(fold_sc)

    sp_te = test_df[test_df["species"] == sp].reset_index(drop=True)
    S_te = features[sp_te["feat_idx"].values] @ features[sp_df["feat_idx"].values].T
    uids, inv = np.unique(sp_df["identity"].values, return_inverse=True)
    id_sc = np.column_stack([S_te[:, inv == c].max(axis=1) for c in range(len(uids))])
    top4 = np.argsort(-id_sc, axis=1)[:, :4]
    for i, qid in enumerate(sp_te["id"].values):
        m1_preds[qid] = " ".join(list(uids[top4[i]]) + ["new_individual"])

    print(f"  [Model 1 - {sp.upper()}] Encounter-Disjoint CV MAP@5 = {m1_cv[sp]:.4f}")

sub_m1 = pd.DataFrame([{"id": qid, "predictions": m1_preds[qid]} for qid in test_df["id"].values])
validate_submission(sub_m1, "submission_model1.csv")


# ---------------------------------------------------------------------------
# 4. Model 2: Viewpoint Flank Penalty + Calibrated Novelty Thresholding
# ---------------------------------------------------------------------------
print("\n[3/5] Running Model 2 (Viewpoint Flank Penalty + Thresholding)...")
m2_cv = {}
m2_preds = {}
m2_thresholds = {"hyena": 0.52, "leopard": 0.54}

def apply_flank_penalty(S, que_df, gal_df, sp):
    if sp != "hyena":
        return S
    qv = que_df["viewpoint"].fillna("").to_numpy(dtype=str)[:, None]
    gv = gal_df["viewpoint"].fillna("").to_numpy(dtype=str)[None, :]
    opp = ((qv == "left") & (gv == "right")) | ((qv == "right") & (gv == "left"))
    return S - 0.05 * opp

for sp, hr in [("hyena", 0.25), ("leopard", 0.27)]:
    sp_df = train_df[train_df["species"] == sp].reset_index(drop=True)
    sp_df["enc"] = assign_encounters(sp_df)
    thr = m2_thresholds[sp]
    fold_sc = []
    for seed in [42, 100, 2024]:
        gal, que, labs = make_open_set_split(sp_df, holdout_id_ratio=hr, query_enc_ratio=0.45, seed=seed)
        S = apply_flank_penalty(features[que["feat_idx"].values] @ features[gal["feat_idx"].values].T, que, gal, sp)
        uids, inv = np.unique(gal["identity"].values, return_inverse=True)
        id_sc = np.column_stack([S[:, inv == c].max(axis=1) for c in range(len(uids))])
        top4 = np.argsort(-id_sc, axis=1)[:, :4]
        top1 = id_sc[np.arange(len(que)), top4[:, 0]]
        preds = [
            (["new_individual"] + list(uids[top4[i]])) if top1[i] < thr else (list(uids[top4[i]]) + ["new_individual"])
            for i in range(len(que))
        ]
        fold_sc.append(map5_score(preds, labs))
    m2_cv[sp] = np.mean(fold_sc)

    sp_te = test_df[test_df["species"] == sp].reset_index(drop=True)
    S_te = apply_flank_penalty(features[sp_te["feat_idx"].values] @ features[sp_df["feat_idx"].values].T, sp_te, sp_df, sp)
    uids, inv = np.unique(sp_df["identity"].values, return_inverse=True)
    id_sc = np.column_stack([S_te[:, inv == c].max(axis=1) for c in range(len(uids))])
    top4 = np.argsort(-id_sc, axis=1)[:, :4]
    top1 = id_sc[np.arange(len(sp_te)), top4[:, 0]]
    for i, qid in enumerate(sp_te["id"].values):
        p = (["new_individual"] + list(uids[top4[i]])) if top1[i] < thr else (list(uids[top4[i]]) + ["new_individual"])
        m2_preds[qid] = " ".join(p)

    print(f"  [Model 2 - {sp.upper()}] Encounter-Disjoint CV MAP@5 = {m2_cv[sp]:.4f} (Kaggle LB: 0.75821)")

sub_m2 = pd.DataFrame([{"id": qid, "predictions": m2_preds[qid]} for qid in test_df["id"].values])
validate_submission(sub_m2, "submission_model2.csv")


# ---------------------------------------------------------------------------
# 5. Model 3: Dual-Branch Manifold Diffusion & k-Reciprocal GBDT + Bayes MAP@5
# ---------------------------------------------------------------------------
print("\n[4/5] Running Model 3 (Dual-Branch Manifold Diffusion + k-Reciprocal GBDT + Bayes MAP@5)...")


def whiten_multi(G, Q, gal_ids=None, regs=(0.0002, 0.0006, 0.003), mode="lda_sample"):
    """
    Center and whiten embeddings using zero-leak covariance estimation:
      - 'lda_sample': standard within-class covariance on gallery IDs
      - 'lda_bal':    identity-balanced within-class covariance (equal weight per ID)
      - 'pca_gal':    unsupervised gallery PCA covariance
      - 'pca_all':    transductive unlabeled PCA covariance over [G; Q] (zero label leak)
    """
    if mode == "pca_all":
        A = np.vstack([G, Q])
        mu = A.mean(axis=0, keepdims=True)
        Gc, Qc = G - mu, Q - mu
        Ac = A - mu
        cov = (Ac.T @ Ac) / max(1, len(Ac))
    else:
        mu = G.mean(axis=0, keepdims=True)
        Gc, Qc = G - mu, Q - mu
        if mode == "lda_bal":
            u_ids, inv = np.unique(gal_ids, return_inverse=True)
            cov = np.zeros((G.shape[1], G.shape[1]), dtype=np.float64)
            n_multi = 0
            for c in range(len(u_ids)):
                mask = (inv == c)
                cnt = mask.sum()
                if cnt > 1:
                    d = Gc[mask] - Gc[mask].mean(axis=0, keepdims=True)
                    cov += (d.T @ d) / cnt
                    n_multi += 1
            cov /= max(1, n_multi)
        elif mode == "lda_sample":
            u_ids, inv = np.unique(gal_ids, return_inverse=True)
            G_w = np.zeros_like(Gc)
            n_multi = 0
            for c in range(len(u_ids)):
                mask = (inv == c)
                cnt = mask.sum()
                if cnt > 1:
                    G_w[mask] = Gc[mask] - Gc[mask].mean(axis=0, keepdims=True)
                    n_multi += cnt
            cov = (G_w.T @ G_w) / max(1, n_multi)
        else:
            cov = (Gc.T @ Gc) / max(1, len(Gc))

    vals, vecs = np.linalg.eigh(cov.astype(np.float64))
    vals = np.maximum(vals, 0.0)
    Gp, Qp = Gc @ vecs, Qc @ vecs
    out = []
    for r in regs:
        sc = 1.0 / np.sqrt(vals + r)
        Gw = ((Gp * sc) @ vecs.T).astype(np.float32)
        Qw = ((Qp * sc) @ vecs.T).astype(np.float32)
        Gw /= np.maximum(np.linalg.norm(Gw, axis=1, keepdims=True), 1e-12)
        Qw /= np.maximum(np.linalg.norm(Qw, axis=1, keepdims=True), 1e-12)
        out.append((Gw, Qw))
    return out


def fast_k_reciprocal_rerank(S_qg, S_gg, S_qq, k1=8, k2=2, lambda_value=0.55):
    """CVPR 2017 k-reciprocal Jaccard re-ranking over similarity submatrices."""
    all_sim = np.block([[S_qq, S_qg], [S_qg.T, S_gg]])
    orig_dist = np.maximum(0.0, 2.0 - 2.0 * all_sim)
    orig_dist = orig_dist / np.maximum(orig_dist.max(axis=0, keepdims=True), 1e-12)

    N = orig_dist.shape[0]
    N_q = S_qg.shape[0]
    top_m = min(k1 + 1, N)
    p_idx = np.argpartition(orig_dist, top_m - 1, axis=1)[:, :top_m]
    p_dist = np.take_along_axis(orig_dist, p_idx, axis=1)
    initial_rank = np.take_along_axis(p_idx, np.argsort(p_dist, axis=1), axis=1)

    half_m = int(np.round(k1 / 2)) + 1
    in_k1 = np.zeros((N, N), dtype=bool)
    in_k1[np.arange(N)[:, None], initial_rank] = True
    recip_mask = in_k1 & in_k1.T

    in_half = np.zeros((N, N), dtype=bool)
    in_half[np.arange(N)[:, None], initial_rank[:, :half_m]] = True
    half_recip_mask = in_half & in_half.T

    k_lists = [np.where(recip_mask[i])[0] for i in range(N)]
    h_lists = [np.where(half_recip_mask[i])[0] for i in range(N)]

    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        kr = k_lists[i]
        kr_exp = set(kr)
        row_m = recip_mask[i]
        for j in kr:
            cr = h_lists[j]
            if len(cr) > 0 and row_m[cr].sum() > (2.0 / 3.0) * len(cr):
                kr_exp.update(cr)
        arr = np.fromiter(kr_exp, dtype=int)
        w = np.exp(-orig_dist[i, arr])
        V[i, arr] = w / np.sum(w)

    if k2 > 1:
        V_qe = np.empty_like(V)
        nbrs_k2 = initial_rank[:, :k2]
        for i in range(N):
            V_qe[i] = V[nbrs_k2[i]].mean(axis=0)
        V = V_qe

    V_q, V_g = V[:N_q], V[N_q:]
    jaccard = np.empty((N_q, N - N_q), dtype=np.float32)
    for i in range(N_q):
        vq = V_q[i]
        nz = vq > 0
        ms = np.minimum(vq[nz][None, :], V_g[:, nz]).sum(axis=1)
        jaccard[i] = ms / np.maximum(2.0 - ms, 1e-8)

    final_dist = (1.0 - lambda_value) * (1.0 - jaccard) + lambda_value * orig_dist[:N_q, N_q:]
    return 1.0 - final_dist


def _row_znorm(A):
    return (A - A.mean(axis=1, keepdims=True)) / np.maximum(A.std(axis=1, keepdims=True), 1e-6)


def _trim_znorm(A, excl=3):
    """Row-wise z-score excluding top matches so strong true hits do not skew background stats."""
    k = min(excl, A.shape[1] - 2)
    s = np.sort(A, axis=1)[:, :-k]
    return (A - s.mean(axis=1, keepdims=True)) / np.maximum(s.std(axis=1, keepdims=True), 1e-6)


def _glob_znorm(A):
    return (A - A.mean()) / max(A.std(), 1e-6)


def solve_diffusion_sparse(
    S_all, N_q, enc_c, enc_s, q_side, vp_p, M,
    k_nbr=10, alpha=0.82, gamma=3.5, asym_w=0.20, deg_damp=0.0,
    use_softmax=False, tau=0.075, n_iter=28
):
    """
    Solve random-walk label absorption (I - alpha P) Z = (1 - alpha) Y over the joint
    [Queries; Encounter-Prototypes] manifold graph via sparse matrix power iteration.
    Returns class-aggregated diffusion densities, identity share, and Bhattacharyya kernel scores.
    """
    N = len(S_all)
    N_e = len(enc_c)
    S = S_all.copy()
    np.fill_diagonal(S, -1e9 if use_softmax else 0.0)
    km = min(k_nbr, N - 1)
    idx = np.argpartition(S, -km, axis=1)[:, -km:]

    if use_softmax:
        W = np.zeros((N, N), dtype=np.float32)
        v = np.take_along_axis(S, idx, axis=1)
        ev = np.exp((v - v.max(axis=1, keepdims=True)) / tau)
        W[np.arange(N)[:, None], idx] = ev / ev.sum(axis=1, keepdims=True)
        W = 0.5 * (W + W.T)
    else:
        A = np.zeros((N, N), dtype=np.float32)
        A[np.arange(N)[:, None], idx] = np.power(np.maximum(np.take_along_axis(S, idx, axis=1), 0.0), gamma)
        mut = (A > 0) & (A.T > 0)
        W = np.where(mut, np.sqrt(A * A.T + 1e-12), asym_w * (A + A.T))

    d = np.maximum(W.sum(axis=1) + deg_damp, 1e-8)
    di = 1.0 / np.sqrt(d)
    P_sp = csr_matrix((di[:, None] * W) * di[None, :])

    rhs = np.zeros((N, N_e), dtype=np.float32)
    rhs[N_q:, :] = (1.0 - alpha) * np.eye(N_e, dtype=np.float32)
    Z_full = rhs.copy()
    for _ in range(n_iter):
        Z_full = alpha * (P_sp @ Z_full) + rhs
    Z_q, Z_g = Z_full[:N_q, :], Z_full[N_q:, :]

    e_counts = np.bincount(enc_c, minlength=M)
    c_max = np.column_stack([Z_q[:, enc_c == c].max(axis=1) for c in range(M)])
    c_sum = np.column_stack([Z_q[:, enc_c == c].sum(axis=1) for c in range(M)])
    c_bal = c_sum / np.sqrt(e_counts[None, :])
    tot_gal = Z_q.sum(axis=1)
    share = c_sum / np.maximum(tot_gal[:, None], 1e-6)

    # Bhattacharyya diffusion kernel: cosine between Hellinger-mapped walk distributions
    Hz_q = np.sqrt(np.maximum(Z_q, 0.0))
    Hz_q /= np.maximum(np.linalg.norm(Hz_q, axis=1, keepdims=True), 1e-12)
    Hz_g = np.sqrt(np.maximum(Z_g, 0.0))
    Hz_g /= np.maximum(np.linalg.norm(Hz_g, axis=1, keepdims=True), 1e-12)
    S_bh = (Hz_q @ Hz_g.T) - vp_p * ((q_side[:, None] * enc_s[None, :]) == -1)
    sc_bh = np.column_stack([S_bh[:, enc_c == c].max(axis=1) for c in range(M)])

    return (
        np.log1p(150.0 * np.maximum(c_max, 0.0)),
        np.log1p(150.0 * np.maximum(c_sum, 0.0)),
        np.log1p(150.0 * np.maximum(c_bal, 0.0)),
        share,
        sc_bh,
    )


def _compute_branch_base_scores(G_raw, Q_raw, G_hyb, Q_hyb, G_l02, Q_l02, G_l30, Q_l30, G_pa, Q_pa,
                                class_to_idx, id_inv, id_counts, q_side, g_side, same_view, opp_broad, pen, M, N_q):
    """Compute core retrieval views (multi-reg LDA/PCA, AQE, DBA, centroids, balanced ridge, CSLS)."""
    S_raw = (Q_raw @ G_raw.T) - pen * opp_broad + 0.010 * same_view
    S_l02 = (Q_l02 @ G_l02.T) - pen * opp_broad + 0.015 * same_view
    S_l30 = (Q_l30 @ G_l30.T) - pen * opp_broad + 0.015 * same_view
    S_pca = (Q_pa  @ G_pa.T)  - pen * opp_broad + 0.010 * same_view
    S_hyb = (Q_hyb @ G_hyb.T) - pen * opp_broad + 0.015 * same_view

    top2_g = np.argpartition(S_hyb, -2, axis=1)[:, -2:]
    Q_aqe = Q_hyb + 0.20 * G_hyb[top2_g].mean(axis=1)
    Q_aqe /= np.maximum(np.linalg.norm(Q_aqe, axis=1, keepdims=True), 1e-12)
    S_aqe = (Q_aqe @ G_hyb.T) - pen * opp_broad + 0.015 * same_view

    G_dba = G_hyb.copy()
    for c, idx in enumerate(class_to_idx):
        if len(idx) > 1:
            gs = g_side[idx]
            for li, gi in enumerate(idx):
                compat = idx[gs == gs[li]]
                if len(compat) > 1:
                    G_dba[gi] = 0.80 * G_hyb[gi] + 0.20 * G_hyb[compat].mean(axis=0)
    G_dba /= np.maximum(np.linalg.norm(G_dba, axis=1, keepdims=True), 1e-12)
    S_dba = (Q_hyb @ G_dba.T) - pen * opp_broad + 0.015 * same_view

    sc_raw = np.empty((N_q, M), dtype=np.float32)
    sc_l02 = np.empty((N_q, M), dtype=np.float32)
    sc_l30 = np.empty((N_q, M), dtype=np.float32)
    sc_pca = np.empty((N_q, M), dtype=np.float32)
    sc_hyb = np.empty((N_q, M), dtype=np.float32)
    sc_hyb_mean = np.empty((N_q, M), dtype=np.float32)
    sc_hyb_top2 = np.empty((N_q, M), dtype=np.float32)
    sc_aqe = np.empty((N_q, M), dtype=np.float32)
    sc_dba = np.empty((N_q, M), dtype=np.float32)
    has_same_view = np.zeros((N_q, M), dtype=np.float32)
    opp_ratio = np.zeros((N_q, M), dtype=np.float32)
    strong_hit_frac = np.zeros((N_q, M), dtype=np.float32)

    q_bg_thr = np.percentile(S_hyb, 97, axis=1, keepdims=True)
    for c, idx in enumerate(class_to_idx):
        sc_raw[:, c] = S_raw[:, idx].max(axis=1)
        sc_l02[:, c] = S_l02[:, idx].max(axis=1)
        sc_l30[:, c] = S_l30[:, idx].max(axis=1)
        sc_pca[:, c] = S_pca[:, idx].max(axis=1)
        sub_h = S_hyb[:, idx]
        sc_hyb[:, c] = sub_h.max(axis=1)
        sc_hyb_mean[:, c] = sub_h.mean(axis=1)
        sc_hyb_top2[:, c] = np.partition(sub_h, -2, axis=1)[:, -2:].mean(axis=1) if len(idx) > 1 else sc_hyb[:, c]
        sc_aqe[:, c] = S_aqe[:, idx].max(axis=1)
        sc_dba[:, c] = S_dba[:, idx].max(axis=1)
        has_same_view[:, c] = same_view[:, idx].any(axis=1).astype(np.float32)
        opp_ratio[:, c] = opp_broad[:, idx].mean(axis=1).astype(np.float32)
        strong_hit_frac[:, c] = (sub_h >= q_bg_thr).mean(axis=1).astype(np.float32)

    C_glob = np.vstack([G_hyb[idx].mean(axis=0) for idx in class_to_idx])
    C_glob /= np.maximum(np.linalg.norm(C_glob, axis=1, keepdims=True), 1e-12)
    sc_cent = Q_hyb @ C_glob.T

    sc_vcent = np.empty((N_q, M), dtype=np.float32)
    for c, idx in enumerate(class_to_idx):
        gs = g_side[idx]
        c_l = G_hyb[idx[gs == -1]].mean(axis=0) if (gs == -1).any() else C_glob[c]
        c_r = G_hyb[idx[gs == 1]].mean(axis=0) if (gs == 1).any() else C_glob[c]
        c_o = G_hyb[idx[gs == 0]].mean(axis=0) if (gs == 0).any() else C_glob[c]
        c_l /= max(np.linalg.norm(c_l), 1e-12)
        c_r /= max(np.linalg.norm(c_r), 1e-12)
        c_o /= max(np.linalg.norm(c_o), 1e-12)
        sc_vcent[:, c] = np.where(q_side == -1, Q_hyb @ c_l, np.where(q_side == 1, Q_hyb @ c_r, Q_hyb @ c_o))

    Y = np.zeros((len(G_hyb), M), dtype=np.float32)
    Y[np.arange(len(G_hyb)), id_inv] = 1.0 / np.sqrt(id_counts[id_inv])
    lam_r = 4.0 * np.sqrt(len(G_hyb) / 1500.0)
    if len(G_hyb) > G_hyb.shape[1]:
        W_r = np.linalg.solve((G_hyb.T @ G_hyb) + lam_r * np.eye(G_hyb.shape[1], dtype=np.float32), G_hyb.T @ Y)
        sc_ridge = Q_hyb @ W_r
    else:
        A_r = np.linalg.solve((G_hyb @ G_hyb.T) + lam_r * np.eye(len(G_hyb), dtype=np.float32), Y)
        sc_ridge = (Q_hyb @ G_hyb.T) @ A_r

    S_gg_h = G_hyb @ G_hyb.T
    r_G = np.sort(S_gg_h, axis=1)[:, -16:-1].mean(axis=1)
    S_csls_img = 2.0 * S_hyb - 0.55 * r_G[None, :]
    r_Q = np.sort(sc_hyb, axis=1)[:, -15:].mean(axis=1)
    sc_csls = np.column_stack([S_csls_img[:, idx].max(axis=1) for idx in class_to_idx]) - 0.55 * r_Q[:, None]

    return {
        "S_hyb": S_hyb, "S_gg_h": S_gg_h, "sc_raw": sc_raw, "sc_l02": sc_l02, "sc_l30": sc_l30,
        "sc_pca": sc_pca, "sc_hyb": sc_hyb, "sc_hyb_mean": sc_hyb_mean, "sc_hyb_top2": sc_hyb_top2,
        "sc_aqe": sc_aqe, "sc_dba": sc_dba, "sc_cent": sc_cent, "sc_vcent": sc_vcent,
        "sc_ridge": sc_ridge, "sc_csls": sc_csls, "r_Q": r_Q,
        "has_same_view": has_same_view, "opp_ratio": opp_ratio, "strong_hit_frac": strong_hit_frac,
    }


def compute_dual_branch_views(gal_df, que_df, sp):
    """
    Build both complementary feature-geometry views once per split:
      - 'krec': sample-weighted LDA + gallery PCA + raw k-Reciprocal (proven 0.8235 Kaggle geometry)
      - 'diff': identity-balanced LDA + transductive PCA + Manifold Diffusion + Bhattacharyya +
                Impostor Max-Bias Debiasing + Encounter-Collapsed k-Reciprocal + QQ Burst Clusters
    """
    G_raw = features[gal_df["feat_idx"].values]
    Q_raw = features[que_df["feat_idx"].values]
    gal_ids = gal_df["identity"].to_numpy(dtype=str)
    unique_ids, id_inv = np.unique(gal_ids, return_inverse=True)
    M, N_q = len(unique_ids), len(que_df)
    class_to_idx = [np.where(id_inv == c)[0] for c in range(M)]
    id_counts = np.array([len(idx) for idx in class_to_idx])

    qv_arr = que_df["viewpoint"].fillna("").to_numpy(dtype=str)
    gv_arr = gal_df["viewpoint"].fillna("").to_numpy(dtype=str)
    q_side, g_side = flank_side(que_df["viewpoint"]), flank_side(gal_df["viewpoint"])
    opp_broad = (q_side[:, None] * g_side[None, :]) == -1
    same_view = (np.where(qv_arr == "", "X", qv_arr)[:, None] == np.where(gv_arr == "", "Y", gv_arr)[None, :])
    pen = 0.04 if sp == "hyena" else 0.02

    # --- Branch 1 ('krec'): sample-weighted LDA + gallery PCA ---
    (G1_02, Q1_02), (G1_05, Q1_05), (G1_30, Q1_30) = whiten_multi(G_raw, Q_raw, gal_ids, mode="lda_sample")
    [(G1_pa, Q1_pa)] = whiten_multi(G_raw, Q_raw, regs=(0.003,), mode="pca_gal")
    G1_hyb = 0.40 * G1_05 + 0.25 * G1_02 + 0.20 * G1_30 + 0.15 * G1_pa
    Q1_hyb = 0.40 * Q1_05 + 0.25 * Q1_02 + 0.20 * Q1_30 + 0.15 * Q1_pa
    G1_hyb /= np.maximum(np.linalg.norm(G1_hyb, axis=1, keepdims=True), 1e-12)
    Q1_hyb /= np.maximum(np.linalg.norm(Q1_hyb, axis=1, keepdims=True), 1e-12)

    b1 = _compute_branch_base_scores(
        G_raw, Q_raw, G1_hyb, Q1_hyb, G1_02, Q1_02, G1_30, Q1_30, G1_pa, Q1_pa,
        class_to_idx, id_inv, id_counts, q_side, g_side, same_view, opp_broad, pen, M, N_q
    )
    S_fk1 = fast_k_reciprocal_rerank(Q1_hyb @ G1_hyb.T, b1["S_gg_h"], Q1_hyb @ Q1_hyb.T, k1=8, k2=2, lambda_value=0.55) - pen * opp_broad
    sc_fk1 = np.column_stack([S_fk1[:, idx].max(axis=1) for idx in class_to_idx])
    comp_ind1 = (
        0.35 * _glob_znorm(b1["sc_hyb"]) + 0.15 * _glob_znorm(b1["sc_cent"])
        + 0.20 * _glob_znorm(b1["sc_vcent"]) + 0.20 * _glob_znorm(b1["sc_csls"])
        + 0.10 * _glob_znorm(b1["sc_ridge"]) + 0.15 * _glob_znorm(b1["sc_dba"])
    )
    comp_krec = 0.78 * comp_ind1 + 0.28 * _glob_znorm(sc_fk1)
    krec_view = {
        "unique_ids": unique_ids, "id_counts": id_counts,
        **{k: b1[k] for k in ["sc_raw", "sc_l02", "sc_l30", "sc_pca", "sc_hyb", "sc_hyb_mean", "sc_hyb_top2",
                               "sc_aqe", "sc_dba", "sc_cent", "sc_vcent", "sc_ridge", "sc_csls",
                               "has_same_view", "opp_ratio", "strong_hit_frac", "r_Q"]},
        "sc_fkrec": sc_fk1, "comp": comp_krec,
        "z_hyb": _row_znorm(b1["sc_hyb"]), "z_l30": _row_znorm(b1["sc_l30"]), "z_pca": _row_znorm(b1["sc_pca"]),
        "z_csls": _row_znorm(b1["sc_csls"]), "z_cent": _row_znorm(b1["sc_cent"]), "z_vcent": _row_znorm(b1["sc_vcent"]),
        "z_ridge": _row_znorm(b1["sc_ridge"]), "z_fkrec": _row_znorm(sc_fk1), "z_comp": _row_znorm(comp_krec),
    }

    # --- Branch 2 ('diff'): identity-balanced LDA + transductive PCA + Diffusion + Bhattacharyya ---
    (G2_02, Q2_02), (G2_05, Q2_05), (G2_30, Q2_30) = whiten_multi(G_raw, Q_raw, gal_ids, mode="lda_bal")
    [(G2_pa, Q2_pa)] = whiten_multi(G_raw, Q_raw, regs=(0.003,), mode="pca_all")
    G2_hyb = 0.25 * G2_02 + 0.40 * G2_05 + 0.20 * G2_30 + 0.15 * G2_pa
    Q2_hyb = 0.25 * Q2_02 + 0.40 * Q2_05 + 0.20 * Q2_30 + 0.15 * Q2_pa
    G2_hyb /= np.maximum(np.linalg.norm(G2_hyb, axis=1, keepdims=True), 1e-12)
    Q2_hyb /= np.maximum(np.linalg.norm(Q2_hyb, axis=1, keepdims=True), 1e-12)

    b2 = _compute_branch_base_scores(
        G_raw, Q_raw, G2_hyb, Q2_hyb, G2_02, Q2_02, G2_30, Q2_30, G2_pa, Q2_pa,
        class_to_idx, id_inv, id_counts, q_side, g_side, same_view, opp_broad, pen, M, N_q
    )
    S_gg_vp = b2["S_gg_h"] - pen * ((g_side[:, None] * g_side[None, :]) == -1)
    sc_gg = np.column_stack([S_gg_vp[:, idx].max(axis=1) for idx in class_to_idx])
    imp_p98 = np.array([np.percentile(sc_gg[id_inv != c, c], 98) for c in range(M)], dtype=np.float32)
    sc_imp_deb = b2["sc_hyb"] - 0.35 * imp_p98[None, :]

    # Collapse gallery burst twins (cos >= 0.80 within class) into balanced encounter prototypes
    enc_vecs, enc_c, enc_s = [], [], []
    for c, idx in enumerate(class_to_idx):
        if len(idx) == 1:
            enc_vecs.append(G2_hyb[idx[0]]); enc_c.append(c); enc_s.append(g_side[idx[0]])
        else:
            Gc_raw = G_raw[idx]
            nc, elabs = connected_components((Gc_raw @ Gc_raw.T) >= 0.80, directed=False)
            for ec in range(nc):
                sub = idx[elabs == ec]
                v = G2_hyb[sub].mean(axis=0)
                v /= max(np.linalg.norm(v), 1e-12)
                enc_vecs.append(v); enc_c.append(c); enc_s.append(int(np.sign(g_side[sub].sum())))
    Ge = np.vstack(enc_vecs)
    enc_c = np.array(enc_c)
    enc_s = np.array(enc_s)

    all_s = np.concatenate([q_side, enc_s])
    pen_all = pen * ((all_s[:, None] * all_s[None, :]) == -1)
    X_node = np.vstack([Q2_hyb, Ge])
    S_all = (X_node @ X_node.T) - pen_all
    S_tmp = S_all.copy()
    np.fill_diagonal(S_tmp, -1.0)
    t2_i = np.argpartition(S_tmp, -2, axis=1)[:, -2:]
    t2_v = np.power(np.maximum(np.take_along_axis(S_tmp, t2_i, axis=1), 0.0), 3.0)[:, :, None]
    X_qe = X_node + 0.35 * (t2_v * X_node[t2_i]).sum(axis=1)
    X_qe /= np.maximum(np.linalg.norm(X_qe, axis=1, keepdims=True), 1e-12)
    S_qe_all = (X_qe @ X_qe.T) - pen_all

    dm_p, ds_p, db_p, dsh_p, sc_bh = solve_diffusion_sparse(
        S_all, N_q, enc_c, enc_s, q_side, pen, M, k_nbr=10, alpha=0.82, gamma=3.5, asym_w=0.20, deg_damp=0.0
    )
    dm_d, ds_d, db_d, dsh_d, sc_bh_d = solve_diffusion_sparse(
        S_qe_all, N_q, enc_c, enc_s, q_side, pen, M, k_nbr=14, alpha=0.82, gamma=3.5, asym_w=0.20, deg_damp=0.015
    )

    S_ek = fast_k_reciprocal_rerank(Q2_hyb @ Ge.T, Ge @ Ge.T, Q2_hyb @ Q2_hyb.T, k1=10, k2=2, lambda_value=0.55)
    S_ek -= pen * ((q_side[:, None] * enc_s[None, :]) == -1)
    sc_ek = np.column_stack([S_ek[:, enc_c == c].max(axis=1) for c in range(M)])

    S_fk2 = fast_k_reciprocal_rerank(Q2_hyb @ G2_hyb.T, b2["S_gg_h"], Q2_hyb @ Q2_hyb.T, k1=8, k2=2, lambda_value=0.55) - pen * opp_broad
    sc_fk2 = np.column_stack([S_fk2[:, idx].max(axis=1) for idx in class_to_idx])

    adj_qq = ((Q_raw @ Q_raw.T) >= 0.82) & ((q_side[:, None] * q_side[None, :]) != -1)
    _, qq_clust = connected_components(adj_qq, directed=False)

    comp_diff = (
        0.18 * _glob_znorm(b2["sc_hyb"]) + 0.12 * _glob_znorm(sc_imp_deb) + 0.10 * _glob_znorm(b2["sc_vcent"])
        + 0.10 * _glob_znorm(b2["sc_csls"]) + 0.16 * _glob_znorm(ds_p) + 0.14 * _glob_znorm(db_d)
        + 0.10 * _glob_znorm(sc_bh) + 0.08 * _glob_znorm(sc_ek) + 0.06 * _glob_znorm(sc_fk2)
    )
    diff_view = {
        "unique_ids": unique_ids, "id_counts": id_counts, "qq_clust": qq_clust,
        **{k: b2[k] for k in ["sc_raw", "sc_l02", "sc_l30", "sc_pca", "sc_hyb", "sc_hyb_mean", "sc_hyb_top2",
                               "sc_aqe", "sc_dba", "sc_cent", "sc_vcent", "sc_ridge", "sc_csls",
                               "has_same_view", "opp_ratio", "strong_hit_frac", "r_Q"]},
        "sc_imp_deb": sc_imp_deb, "sc_fkrec": sc_fk2, "sc_ekrec": sc_ek,
        "dm_p": dm_p, "ds_p": ds_p, "db_p": db_p, "dsh_p": dsh_p, "sc_bh": sc_bh,
        "dm_d": dm_d, "ds_d": ds_d, "db_d": db_d, "dsh_d": dsh_d, "sc_bh_d": sc_bh_d, "comp": comp_diff,
        "tz_hyb": _trim_znorm(b2["sc_hyb"]), "tz_bh": _trim_znorm(sc_bh), "tz_deb": _trim_znorm(sc_imp_deb),
        "tz_fk": _trim_znorm(sc_fk2), "tz_ek": _trim_znorm(sc_ek), "tz_csls": _trim_znorm(b2["sc_csls"]),
        "tz_vcent": _trim_znorm(b2["sc_vcent"]), "tz_ridge": _trim_znorm(b2["sc_ridge"]), "tz_comp": _trim_znorm(comp_diff),
    }
    if sp == "hyena":
        _, ds_s, db_s, _, _ = solve_diffusion_sparse(
            S_all, N_q, enc_c, enc_s, q_side, pen, M, k_nbr=12, alpha=0.82, gamma=3.5, use_softmax=True, tau=0.075
        )
        diff_view["ds_s"] = ds_s
        diff_view["db_s"] = db_s

    krec_view["qq_clust"] = qq_clust
    return krec_view, diff_view


COLS_A_KREC = [
    "raw", "l02", "l30", "pca", "hyb", "hyb_mean", "hyb_top2", "aqe", "dba",
    "cent", "vcent", "ridge", "csls", "comp",
    "z_hyb", "z_l30", "z_pca", "z_csls", "z_cent", "z_vcent", "z_ridge", "z_comp",
    "gap_top1_hyb", "gap_next_hyb", "gap_top1_csls", "gap_top1_comp", "rank",
    "hyb_minus_cent", "hyb_minus_vcent", "log_gal_cnt", "has_same_view", "opp_ratio", "strong_hit_frac",
    "top1_z_hyb", "top1_top2_hyb", "top1_z_csls", "top1_top2_csls",
    "fkrec", "z_fkrec", "gap_top1_fkrec", "top1_top2_fkrec",
]

COLS_B_KREC = [
    "hyb_1", "hyb_2", "hyb_3", "hyb_5", "hyb_gap12", "hyb_gap15", "hyb_ent",
    "l02_1", "l30_1", "pca_1", "raw_1", "aqe_1", "dba_1",
    "z_hyb_1", "z_l30_1", "z_pca_1", "z_csls_1", "z_cent_1", "z_vcent_1", "z_ridge_1", "z_comp_1",
    "z_hyb_gap12", "z_csls_gap12", "csls_1", "csls_gap12", "cent_1", "vcent_1", "ridge_1",
    "hyb_cent_diff", "hyb_vcent_diff", "has_same_v_1", "opp_ratio_1", "strong_hit_1", "r_Q",
    "cand_prob_1", "cand_prob_2", "cand_prob_gap12", "cand_prob_sum3", "cand_prob_sum5",
    "fk_1", "fk_2", "fk_gap12", "fk_ent", "z_fk_1",
]


def _top_stats(A, tau=0.05):
    p5 = np.sort(A, axis=1)[:, -5:][:, ::-1]
    ex = np.exp((p5 - p5[:, :1]) / tau)
    ent = -np.sum((ex / ex.sum(axis=1, keepdims=True)) * np.log((ex / ex.sum(axis=1, keepdims=True)) + 1e-12), axis=1)
    return p5, ent


def build_stage_a_krec(v, que_labels=None, K=12):
    comp, z_comp = v["comp"], v["z_comp"]
    N_q, M = comp.shape
    K = min(K, M)
    topK = np.argpartition(comp, -K, axis=1)[:, -K:]
    cands = np.take_along_axis(topK, np.argsort(-np.take_along_axis(comp, topK, axis=1), axis=1), axis=1)
    ri = np.arange(N_q)[:, None]

    v_hyb, v_csls, v_comp = v["sc_hyb"][ri, cands], v["sc_csls"][ri, cands], comp[ri, cands]
    t1_h, t2_h = v["sc_hyb"].max(axis=1), np.partition(v["sc_hyb"], -2, axis=1)[:, -2]
    t1_c, t2_c = v["sc_csls"].max(axis=1), np.partition(v["sc_csls"], -2, axis=1)[:, -2]
    t1_f, t2_f = v["sc_fkrec"].max(axis=1), np.partition(v["sc_fkrec"], -2, axis=1)[:, -2]
    v_fk = v["sc_fkrec"][ri, cands]

    next_hyb = np.empty_like(v_hyb)
    next_hyb[:, :-1] = v_hyb[:, :-1] - v_hyb[:, 1:]
    next_hyb[:, -1] = 0.0
    ranks = np.broadcast_to(np.arange(K, dtype=np.float32)[None, :], (N_q, K))

    mat = np.stack([
        v["sc_raw"][ri, cands], v["sc_l02"][ri, cands], v["sc_l30"][ri, cands],
        v["sc_pca"][ri, cands], v_hyb, v["sc_hyb_mean"][ri, cands], v["sc_hyb_top2"][ri, cands],
        v["sc_aqe"][ri, cands], v["sc_dba"][ri, cands],
        v["sc_cent"][ri, cands], v["sc_vcent"][ri, cands], v["sc_ridge"][ri, cands], v_csls, v_comp,
        v["z_hyb"][ri, cands], v["z_l30"][ri, cands], v["z_pca"][ri, cands],
        v["z_csls"][ri, cands], v["z_cent"][ri, cands], v["z_vcent"][ri, cands],
        v["z_ridge"][ri, cands], z_comp[ri, cands],
        t1_h[:, None] - v_hyb, next_hyb, t1_c[:, None] - v_csls, v_comp[:, :1] - v_comp, ranks,
        v_hyb - v["sc_cent"][ri, cands], v_hyb - v["sc_vcent"][ri, cands],
        np.log1p(v["id_counts"][cands]).astype(np.float32),
        v["has_same_view"][ri, cands], v["opp_ratio"][ri, cands], v["strong_hit_frac"][ri, cands],
        np.broadcast_to(v["z_hyb"].max(axis=1)[:, None], (N_q, K)), np.broadcast_to((t1_h - t2_h)[:, None], (N_q, K)),
        np.broadcast_to(v["z_csls"].max(axis=1)[:, None], (N_q, K)), np.broadcast_to((t1_c - t2_c)[:, None], (N_q, K)),
        v_fk, v["z_fkrec"][ri, cands], t1_f[:, None] - v_fk, np.broadcast_to((t1_f - t2_f)[:, None], (N_q, K)),
    ], axis=-1)
    X = pd.DataFrame(mat.reshape(N_q * K, -1), columns=COLS_A_KREC)
    y = (v["unique_ids"][cands] == que_labels[:, None]).astype(int).reshape(-1) if que_labels is not None else None
    return X, y, np.full(N_q, K, dtype=int), cands


def build_stage_b_krec(v, cand_probs, cands):
    N_q = cands.shape[0]
    p_hyb, ent_hyb = _top_stats(v["sc_hyb"], tau=0.05)
    p_z_hyb, _ = _top_stats(v["z_hyb"], tau=1.0)
    p_z_csls, _ = _top_stats(v["z_csls"], tau=1.0)
    p_csls, _ = _top_stats(v["sc_csls"], tau=0.05)
    p5_fk, ent_fk = _top_stats(v["sc_fkrec"], tau=0.05)

    order_p = np.argsort(-cand_probs, axis=1)
    sp_p = np.take_along_axis(cand_probs, order_p, axis=1)
    best = cands[np.arange(N_q), order_p[:, 0]]
    ri = np.arange(N_q)

    hyb_1, cent_1, vcent_1 = v["sc_hyb"][ri, best], v["sc_cent"][ri, best], v["sc_vcent"][ri, best]
    mat = np.column_stack([
        p_hyb[:, 0], p_hyb[:, 1], p_hyb[:, 2], p_hyb[:, 4], p_hyb[:, 0] - p_hyb[:, 1], p_hyb[:, 0] - p_hyb[:, 4], ent_hyb,
        v["sc_l02"][ri, best], v["sc_l30"][ri, best], v["sc_pca"][ri, best],
        v["sc_raw"][ri, best], v["sc_aqe"][ri, best], v["sc_dba"][ri, best],
        v["z_hyb"][ri, best], v["z_l30"][ri, best], v["z_pca"][ri, best],
        v["z_csls"][ri, best], v["z_cent"][ri, best], v["z_vcent"][ri, best],
        v["z_ridge"][ri, best], v["z_comp"][ri, best],
        p_z_hyb[:, 0] - p_z_hyb[:, 1], p_z_csls[:, 0] - p_z_csls[:, 1],
        p_csls[:, 0], p_csls[:, 0] - p_csls[:, 1], cent_1, vcent_1, v["sc_ridge"][ri, best],
        hyb_1 - cent_1, hyb_1 - vcent_1,
        v["has_same_view"][ri, best], v["opp_ratio"][ri, best], v["strong_hit_frac"][ri, best], v["r_Q"],
        sp_p[:, 0], sp_p[:, 1], sp_p[:, 0] - sp_p[:, 1], sp_p[:, :3].sum(axis=1), sp_p[:, :5].sum(axis=1),
        p5_fk[:, 0], p5_fk[:, 1], p5_fk[:, 0] - p5_fk[:, 1], ent_fk, v["z_fkrec"][ri, best],
    ])
    return pd.DataFrame(mat, columns=COLS_B_KREC)


def build_stage_a_diff(v, que_labels=None, K=12):
    comp = v["comp"]
    N_q, M = comp.shape
    K = min(K, M)
    topK = np.argpartition(comp, -K, axis=1)[:, -K:]
    cands = np.take_along_axis(topK, np.argsort(-np.take_along_axis(comp, topK, axis=1), axis=1), axis=1)
    ri = np.arange(N_q)[:, None]

    def g(k):
        return v[k][ri, cands]

    t1_h, t2_h = v["sc_hyb"].max(axis=1), np.partition(v["sc_hyb"], -2, axis=1)[:, -2]
    t1_b, t2_b = v["sc_bh"].max(axis=1), np.partition(v["sc_bh"], -2, axis=1)[:, -2]
    t1_d, t2_d = v["ds_p"].max(axis=1), np.partition(v["ds_p"], -2, axis=1)[:, -2]
    t1_e, t2_e = v["sc_ekrec"].max(axis=1), np.partition(v["sc_ekrec"], -2, axis=1)[:, -2]
    ranks = np.broadcast_to(np.arange(K, dtype=np.float32)[None, :], (N_q, K))

    feats = [
        g("sc_raw"), g("sc_l02"), g("sc_l30"), g("sc_pca"), g("sc_hyb"), g("sc_imp_deb"), g("sc_hyb_mean"), g("sc_hyb_top2"),
        g("sc_aqe"), g("sc_dba"), g("sc_cent"), g("sc_vcent"), g("sc_ridge"), g("sc_csls"), g("sc_fkrec"), g("sc_ekrec"),
        g("dm_p"), g("ds_p"), g("db_p"), g("dsh_p"), g("sc_bh"), g("dm_d"), g("ds_d"), g("db_d"), g("sc_bh_d"),
    ]
    if "ds_s" in v:
        feats.extend([g("ds_s"), g("db_s")])
    feats.extend([
        g("comp"), g("tz_hyb"), g("tz_bh"), g("tz_deb"), g("tz_fk"), g("tz_ek"), g("tz_csls"), g("tz_vcent"), g("tz_ridge"), g("tz_comp"),
        t1_h[:, None] - g("sc_hyb"), t1_b[:, None] - g("sc_bh"), t1_d[:, None] - g("ds_p"), t1_e[:, None] - g("sc_ekrec"), g("comp")[:, :1] - g("comp"),
        ranks, g("sc_hyb") - g("sc_cent"), g("sc_hyb") - g("sc_vcent"),
        np.log1p(v["id_counts"][cands]).astype(np.float32), g("has_same_view"), g("opp_ratio"), g("strong_hit_frac"),
        np.broadcast_to(v["tz_hyb"].max(axis=1)[:, None], (N_q, K)), np.broadcast_to((t1_h - t2_h)[:, None], (N_q, K)),
        np.broadcast_to(v["tz_bh"].max(axis=1)[:, None], (N_q, K)), np.broadcast_to((t1_b - t2_b)[:, None], (N_q, K)),
        np.broadcast_to((t1_d - t2_d)[:, None], (N_q, K)), np.broadcast_to((t1_e - t2_e)[:, None], (N_q, K)),
    ])
    mat = np.stack(feats, axis=-1)
    X = pd.DataFrame(mat.reshape(N_q * K, -1), columns=[f"d_{i}" for i in range(mat.shape[-1])])
    y = (v["unique_ids"][cands] == que_labels[:, None]).astype(int).reshape(-1) if que_labels is not None else None
    return X, y, np.full(N_q, K, dtype=int), cands


def build_stage_b_diff(v, cand_probs, cands):
    N_q = cands.shape[0]
    p_hyb, ent_hyb = _top_stats(v["sc_hyb"], 0.05)
    p_deb, _ = _top_stats(v["sc_imp_deb"], 0.05)
    p_bh, ent_bh = _top_stats(v["sc_bh"], 0.05)
    p_dsp, ent_dsp = _top_stats(v["ds_p"], 0.25)
    p_ek, ent_ek = _top_stats(v["sc_ekrec"], 0.05)
    p_fk, ent_fk = _top_stats(v["sc_fkrec"], 0.05)
    p_tzh, _ = _top_stats(v["tz_hyb"], 1.0)
    p_tzb, _ = _top_stats(v["tz_bh"], 1.0)
    p_tze, _ = _top_stats(v["tz_ek"], 1.0)

    ord_p = np.argsort(-cand_probs, axis=1)
    sp_p = np.take_along_axis(cand_probs, ord_p, axis=1)
    best = cands[np.arange(N_q), ord_p[:, 0]]
    ri = np.arange(N_q)

    cols = [
        p_hyb[:, 0], p_hyb[:, 1], p_hyb[:, 2], p_hyb[:, 4], p_hyb[:, 0] - p_hyb[:, 1], p_hyb[:, 0] - p_hyb[:, 4], ent_hyb,
        p_deb[:, 0], p_deb[:, 0] - p_deb[:, 1],
        p_bh[:, 0], p_bh[:, 1], p_bh[:, 0] - p_bh[:, 1], ent_bh,
        p_dsp[:, 0], p_dsp[:, 1], p_dsp[:, 0] - p_dsp[:, 1], ent_dsp,
    ]
    if "ds_s" in v:
        p_dss, _ = _top_stats(v["ds_s"], 0.25)
        cols.extend([p_dss[:, 0], p_dss[:, 0] - p_dss[:, 1]])
    cols.extend([
        p_ek[:, 0], p_ek[:, 0] - p_ek[:, 1], ent_ek,
        p_fk[:, 0], p_fk[:, 0] - p_fk[:, 1], ent_fk,
        v["sc_csls"][ri, best], v["sc_l02"][ri, best], v["sc_l30"][ri, best], v["sc_pca"][ri, best], v["sc_raw"][ri, best],
        v["sc_aqe"][ri, best], v["sc_dba"][ri, best], v["sc_cent"][ri, best], v["sc_vcent"][ri, best], v["sc_ridge"][ri, best], v["r_Q"],
        v["ds_p"][ri, best], v["dm_p"][ri, best], v["db_p"][ri, best], v["dsh_p"][ri, best],
        v["ds_d"][ri, best], v["db_d"][ri, best], v["dsh_d"][ri, best], v["sc_bh"][ri, best], v["sc_bh_d"][ri, best],
    ])
    if "ds_s" in v:
        cols.extend([v["ds_s"][ri, best], v["db_s"][ri, best]])
    cols.extend([
        v["tz_hyb"][ri, best], v["tz_bh"][ri, best], v["tz_deb"][ri, best], v["tz_fk"][ri, best], v["tz_ek"][ri, best], v["tz_comp"][ri, best],
        p_tzh[:, 0] - p_tzh[:, 1], p_tzb[:, 0] - p_tzb[:, 1], p_tze[:, 0] - p_tze[:, 1],
        v["sc_hyb"][ri, best] - v["sc_cent"][ri, best], v["sc_hyb"][ri, best] - v["sc_vcent"][ri, best],
        v["has_same_view"][ri, best], v["opp_ratio"][ri, best], v["strong_hit_frac"][ri, best],
        sp_p[:, 0], sp_p[:, 1], sp_p[:, 0] - sp_p[:, 1], sp_p[:, :3].sum(axis=1), sp_p[:, :5].sum(axis=1),
    ])
    mat = np.column_stack(cols)
    return pd.DataFrame(mat, columns=[f"bd_{i}" for i in range(mat.shape[1])])


def train_stage_a(X, y, groups, seed=42):
    m1 = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.035, max_depth=4, num_leaves=15, min_child_samples=35, subsample=0.8, colsample_bytree=0.75, reg_lambda=5.0, random_state=seed, verbose=-1)
    m1.fit(X, y)
    m2 = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.04, max_depth=4, min_samples_leaf=35, l2_regularization=6.0, random_state=seed)
    m2.fit(X, y)
    m3 = lgb.LGBMRanker(objective="lambdarank", n_estimators=160, learning_rate=0.035, max_depth=4, num_leaves=12, min_child_samples=35, subsample=0.8, colsample_bytree=0.75, reg_lambda=5.0, lambdarank_truncation_level=5, random_state=seed, verbose=-1)
    m3.fit(X, y, group=groups)
    return (m1, m2, m3)


def predict_stage_a(models, X, K):
    m1, m2, m3 = models
    N_q = len(X) // K
    p_clf = 0.55 * m1.predict_proba(X)[:, 1].reshape(N_q, K) + 0.45 * m2.predict_proba(X)[:, 1].reshape(N_q, K)
    rk = m3.predict(X).reshape(N_q, K)
    rk_p = np.exp(rk - rk.max(axis=1, keepdims=True))
    rk_p /= rk_p.sum(axis=1, keepdims=True)
    return 0.65 * p_clf + 0.35 * (rk_p * np.maximum(p_clf.sum(axis=1, keepdims=True), 0.2))


def train_stage_b(X, y, seed=42):
    m1 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.035, max_depth=4, num_leaves=12, min_child_samples=25, subsample=0.8, colsample_bytree=0.75, reg_lambda=5.0, random_state=seed, verbose=-1)
    m1.fit(X, y)
    m2 = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=6.0, random_state=seed)
    m2.fit(X, y)
    m3 = GradientBoostingClassifier(n_estimators=140, learning_rate=0.035, max_depth=3, subsample=0.8, min_samples_leaf=20, random_state=seed)
    m3.fit(X, y)
    return (m1, m2, m3)


def predict_stage_b(models, X):
    m1, m2, m3 = models
    return 0.40 * m1.predict_proba(X)[:, 1] + 0.30 * m2.predict_proba(X)[:, 1] + 0.30 * m3.predict_proba(X)[:, 1]


def burst_pool_outputs(qq_clust, cands, cand_sc, p_nov, w_self=0.60):
    """Pool query-burst probabilities across test images belonging to the same burst encounter."""
    sc_out, pn_out = cand_sc.copy(), p_nov.copy()
    _, counts = np.unique(qq_clust, return_counts=True)
    for cid in np.where(counts > 1)[0]:
        members = np.where(qq_clust == cid)[0]
        pn_out[members] = w_self * pn_out[members] + (1.0 - w_self) * pn_out[members].mean()
        agg = {}
        for mi in members:
            for k in range(cands.shape[1]):
                u = cands[mi, k]
                agg[u] = agg.get(u, 0.0) + cand_sc[mi, k] / len(members)
        for mi in members:
            for k in range(cands.shape[1]):
                sc_out[mi, k] = w_self * cand_sc[mi, k] + (1.0 - w_self) * agg.get(cands[mi, k], 0.0)
    return sc_out, pn_out


def blend_bayes_decode(uid, ca1, sc1, pn1, ca2, sc2, pn2, w1=0.65, bias=1.0, temp=1.0):
    """Bayes Expected MAP@5 list decoder combining Manifold Diffusion (w1) and k-Reciprocal (1 - w1) branches."""
    sp1 = np.power(np.maximum(sc1, 1e-6), temp)
    sp1 /= np.maximum(sp1.sum(axis=1, keepdims=True), 1e-8)
    sp2 = np.power(np.maximum(sc2, 1e-6), temp)
    sp2 /= np.maximum(sp2.sum(axis=1, keepdims=True), 1e-8)

    p1 = np.clip(pn1 * bias, 0.001, 0.999)
    p2 = np.clip(pn2 * bias, 0.001, 0.999)
    w2 = 1.0 - w1
    pb = w1 * p1 + w2 * p2
    pk1 = (1.0 - p1) * 0.96
    pk2 = (1.0 - p2) * 0.96

    preds = []
    for i in range(len(ca1)):
        pool = {}
        for k in range(ca1.shape[1]):
            u = str(uid[ca1[i, k]])
            pool[u] = pool.get(u, 0.0) + w1 * pk1[i] * sp1[i, k]
        for k in range(ca2.shape[1]):
            u = str(uid[ca2[i, k]])
            pool[u] = pool.get(u, 0.0) + w2 * pk2[i] * sp2[i, k]
        items = [("new_individual", float(pb[i]))] + list(pool.items())
        items.sort(key=lambda x: -x[1])
        preds.append([it[0] for it in items[:5]])
    return preds


def fit_branch_oof(fold_views, fold_labels, test_view, build_a_fn, build_b_fn, K=12):
    """Fit 5-fold out-of-fold Stage A + Stage B GBDT ensembles and predict test set."""
    n_folds = len(fold_views)
    fds = []
    for vi, li in zip(fold_views, fold_labels):
        Xa, ya, ga, ca = build_a_fn(vi, li, K=K)
        fds.append((vi, Xa, ya, ga, ca, (li == "new_individual").astype(int), li))

    oof_Xb, oof_yb, oof_sc, oof_scq, oof_pn, oof_ca, oof_uid, oof_l = [], [], [], [], [], [], [], []
    for i in range(n_folds):
        sa = train_stage_a(
            pd.concat([fds[j][1] for j in range(n_folds) if j != i], ignore_index=True),
            np.concatenate([fds[j][2] for j in range(n_folds) if j != i]),
            np.concatenate([fds[j][3] for j in range(n_folds) if j != i]),
            seed=42 + i,
        )
        sc_val = predict_stage_a(sa, fds[i][1], K)
        oof_Xb.append(build_b_fn(fds[i][0], sc_val, fds[i][4]))
        oof_yb.append(fds[i][5])
        oof_sc.append(sc_val)
        oof_ca.append(fds[i][4])
        oof_uid.append(fds[i][0]["unique_ids"])
        oof_l.append(fds[i][6])

    Xa_full = pd.concat([f[1] for f in fds], ignore_index=True)
    ya_full = np.concatenate([f[2] for f in fds])
    ga_full = np.concatenate([f[3] for f in fds])
    stage_a_ens = [train_stage_a(Xa_full, ya_full, ga_full, seed=s) for s in [42, 100, 2024]]

    Xb_full = pd.concat(oof_Xb, ignore_index=True)
    yb_full = np.concatenate(oof_yb)
    for i in range(n_folds):
        sb = train_stage_b(
            pd.concat([oof_Xb[j] for j in range(n_folds) if j != i], ignore_index=True),
            np.concatenate([oof_yb[j] for j in range(n_folds) if j != i]),
            seed=100 + i,
        )
        p_raw = predict_stage_b(sb, oof_Xb[i])
        sc_q, pn_q = burst_pool_outputs(fold_views[i]["qq_clust"], oof_ca[i], oof_sc[i], p_raw, w_self=0.60)
        oof_scq.append(sc_q)
        oof_pn.append(pn_q)

    stage_b_ens = [train_stage_b(Xb_full, yb_full, seed=s) for s in [42, 100, 2024]]
    Xa_te, _, _, ca_te = build_a_fn(test_view, K=K)
    cand_te = np.mean([predict_stage_a(sa, Xa_te, K) for sa in stage_a_ens], axis=0)
    Xb_te = build_b_fn(test_view, cand_te, ca_te)
    pn_te = np.mean([predict_stage_b(sb, Xb_te) for sb in stage_b_ens], axis=0)
    cand_te_q, pn_te_q = burst_pool_outputs(test_view["qq_clust"], ca_te, cand_te, pn_te, w_self=0.60)

    return {
        "oof_uid": oof_uid, "oof_l": oof_l, "oof_yb": oof_yb,
        "oof_ca": oof_ca, "oof_sc": oof_scq, "oof_pn": oof_pn,
        "te_ca": ca_te, "te_sc": cand_te_q, "te_pn": pn_te_q,
        "Xb_full": Xb_full, "Xb_te": Xb_te,
        "auc": roc_auc_score(yb_full, np.concatenate(oof_pn)),
    }


m3_cv = {}
m3_preds = {}
SIFT_DIR = find_sift_directory(DATA_DIR)
if SIFT_DIR is None:
    print("  Supplied SIFT descriptors were not found. Running the global-only Model 3.")
K = 12
sp_configs = [
    ("hyena",   0.25, 0.38, 0.65, [0.90, 0.95, 1.00, 1.05, 1.12], [1.00, 1.10]),
    ("leopard", 0.26, 0.38, 0.50, [0.85, 0.90, 0.95, 1.00],       [1.00, 1.20]),
]

for sp, hr, er, w_diff, bias_grid, temp_grid in sp_configs:
    sp_tr = train_df[train_df["species"] == sp].reset_index(drop=True)
    sp_te = test_df[test_df["species"] == sp].reset_index(drop=True)
    sp_tr["enc"] = assign_encounters(sp_tr)

    krec_folds, diff_folds, fold_labels, sift_folds = [], [], [], []
    for s in [11, 22, 33, 44, 55]:
        gi, qi, li = make_open_set_split(sp_tr, holdout_id_ratio=hr, query_enc_ratio=er, max_per_id=10, seed=s)
        kv, dv = compute_dual_branch_views(gi, qi, sp)
        krec_folds.append(kv)
        diff_folds.append(dv)
        fold_labels.append(li)
        if sp == "leopard" and SIFT_DIR is not None and s in (11, 22):
            sift_folds.append((gi, qi, li))

    krec_test, diff_test = compute_dual_branch_views(sp_tr, sp_te, sp)
    uids = diff_test["unique_ids"]

    res_k = fit_branch_oof(krec_folds, fold_labels, krec_test, build_stage_a_krec, build_stage_b_krec, K=K)
    res_d = fit_branch_oof(diff_folds, fold_labels, diff_test, build_stage_a_diff, build_stage_b_diff, K=K)

    # Joint out-of-fold MAP@5 calibration across blended probabilities
    best_b, best_t, best_oof_map5 = 1.0, 1.0, 0.0
    n_f = len(fold_labels)
    for t in temp_grid:
        for b in bias_grid:
            sc_f = [
                map5_score(
                    blend_bayes_decode(
                        res_d["oof_uid"][i],
                        res_d["oof_ca"][i], res_d["oof_sc"][i], res_d["oof_pn"][i],
                        res_k["oof_ca"][i], res_k["oof_sc"][i], res_k["oof_pn"][i],
                        w1=w_diff, bias=b, temp=t,
                    ),
                    fold_labels[i],
                )
                for i in range(n_f)
            ]
            m_sc = float(np.mean(sc_f))
            if m_sc > best_oof_map5:
                best_oof_map5, best_b, best_t = m_sc, b, t

    oof_map_d = np.mean([
        map5_score(blend_bayes_decode(res_d["oof_uid"][i], res_d["oof_ca"][i], res_d["oof_sc"][i], res_d["oof_pn"][i],
                                      res_d["oof_ca"][i], res_d["oof_sc"][i], res_d["oof_pn"][i], w1=1.0, bias=best_b, temp=best_t), fold_labels[i])
        for i in range(n_f)
    ])
    oof_map_k = np.mean([
        map5_score(blend_bayes_decode(res_k["oof_uid"][i], res_k["oof_ca"][i], res_k["oof_sc"][i], res_k["oof_pn"][i],
                                      res_k["oof_ca"][i], res_k["oof_sc"][i], res_k["oof_pn"][i], w1=1.0, bias=best_b, temp=best_t), fold_labels[i])
        for i in range(n_f)
    ])
    oof_auc_b = roc_auc_score(
        np.concatenate(fold_labels) == "new_individual",
        w_diff * np.concatenate(res_d["oof_pn"]) + (1.0 - w_diff) * np.concatenate(res_k["oof_pn"]),
    )
    m3_cv[sp] = best_oof_map5

    te_lists = blend_bayes_decode(
        uids,
        res_d["te_ca"], res_d["te_sc"], res_d["te_pn"],
        res_k["te_ca"], res_k["te_sc"], res_k["te_pn"],
        w1=w_diff, bias=best_b, temp=best_t,
    )
    stock_r1_new_pct = 100.0 * np.mean([p[0] == "new_individual" for p in te_lists])
    if sift_folds:
        print("  [Model 3 - LEOPARD] Verifying shortlisted candidates with supplied SIFT descriptors...", flush=True)
        sift_training = [
            (*fold, res_k["oof_uid"][i], (res_k["oof_ca"][i], res_d["oof_ca"][i]))
            for i, fold in enumerate(sift_folds)
        ]
        te_lists = rerank_leopards(
            features, sp_tr, sp_te, uids, res_k, res_d, best_b, best_t, te_lists,
            sift_training, SIFT_DIR, flank_side,
        )
    for qi_idx, qid in enumerate(sp_te["id"].values):
        m3_preds[qid] = " ".join(te_lists[qi_idx])

    print(
        f"  [Model 3 - {sp.upper()}] kRec OOF={oof_map_k:.4f} (AUC={res_k['auc']:.4f}) | "
        f"Diffusion OOF={oof_map_d:.4f} (AUC={res_d['auc']:.4f}) | "
        f"Blend OOF={best_oof_map5:.4f} (AUC={oof_auc_b:.4f}, b={best_b:.2f}, t={best_t:.2f}, Stock Test R1 new={stock_r1_new_pct:.1f}%)"
    )


# ---------------------------------------------------------------------------
# 6. Export Final Submission & Summary Table
# ---------------------------------------------------------------------------
print("\n[5/5] Validating and exporting final Model 3 submissions...")
sub_m3 = pd.DataFrame([{"id": qid, "predictions": m3_preds[qid]} for qid in test_df["id"].values])
validate_submission(sub_m3, "submission_model3.csv")
validate_submission(sub_m3, "submission.csv")

n_h = (test_df["species"] == "hyena").sum()
n_l = (test_df["species"] == "leopard").sum()
w_avg = lambda d: (d["hyena"] * n_h + d["leopard"] * n_l) / (n_h + n_l)

summary = pd.DataFrame([
    {"Model": "Model 1 (Baseline Cosine k-NN)", "Hyena CV": f"{m1_cv['hyena']:.4f}", "Leopard CV": f"{m1_cv['leopard']:.4f}", "Weighted CV": f"{w_avg(m1_cv):.4f}", "Kaggle Public": "—"},
    {"Model": "Model 2 (Flank + Thresholds)",   "Hyena CV": f"{m2_cv['hyena']:.4f}", "Leopard CV": f"{m2_cv['leopard']:.4f}", "Weighted CV": f"{w_avg(m2_cv):.4f}", "Kaggle Public": "0.75821"},
    {"Model": "Model 3 (Dual Diffusion+kRec)" + (" + SIFT" if SIFT_DIR else ""), "Hyena CV": f"{m3_cv['hyena']:.4f}", "Leopard CV": f"{m3_cv['leopard']:.4f}", "Weighted CV": f"{w_avg(m3_cv):.4f}", "Kaggle Public": "0.84209" if SIFT_DIR else "0.83547"},
])
print("\n" + "=" * 72)
print(" MODEL PROGRESSION SUMMARY (MAP@5)")
print("=" * 72)
print(summary.to_string(index=False))
if SIFT_DIR:
    print("Model 3 CV columns show the global backbone. The SIFT refinement gained +0.00574 weighted MAP@5 in the separate nested audit.")
print(f"\nCompleted full pipeline in {time.time() - t0:.1f}s.")
