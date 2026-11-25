"""
Wildlife Re-Identification (Problem Statement 3)
Reproducible End-to-End Pipeline & Submission Generator

Builds and evaluates three progressive classical ML / GBDT model stages:
  - Model 1: L2-Normalized Cosine k-NN Baseline (naive Rank-5 new_individual)
  - Model 2: Viewpoint-Aware Cosine k-NN + Calibrated Thresholding (Kaggle LB: 0.75821)
  - Model 3: Multi-Space Whitened Retrieval + Two-Stage GBDT Ensemble + Bayes MAP@5 (Kaggle LB: 0.8177)

Strict Rules Compliance:
  Uses only precomputed MiewID features & tabular metadata with numpy, pandas, scipy,
  scikit-learn, and lightgbm. Zero deep learning frameworks or external datasets.
"""

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# Auto-detect dataset layout (supports both data/ and data/projectwild3/reid_student_data_main/)
PROJECT_ROOT = Path(__file__).resolve().parent
if (PROJECT_ROOT / "data" / "train.csv").exists():
    DATA_DIR = PROJECT_ROOT / "data"
elif (PROJECT_ROOT / "data" / "projectwild3" / "reid_student_data_main" / "train.csv").exists():
    DATA_DIR = PROJECT_ROOT / "data" / "projectwild3" / "reid_student_data_main"
else:
    raise FileNotFoundError(
        "Dataset not found. Expected train.csv inside 'data/' or 'data/projectwild3/reid_student_data_main/'."
    )

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
    out_path = PROJECT_ROOT / filename
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
# 5. Model 3: Multi-Space Retrieval + Two-Stage GBDT Ensemble + Bayes MAP@5
# ---------------------------------------------------------------------------
print("\n[4/5] Running Model 3 (Multi-Space Two-Stage GBDT + Bayes Expected MAP@5)...")

def whiten_multi(G, Q, gal_ids=None, regs=(0.0002, 0.0006, 0.003), supervised=True):
    """Center and whiten features strictly using gallery covariance (LDA within-class or PCA)."""
    mu = G.mean(axis=0, keepdims=True)
    Gc, Qc = G - mu, Q - mu
    if supervised:
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

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    Gp, Qp = Gc @ vecs, Qc @ vecs
    out = []
    for r in regs:
        sc = 1.0 / np.sqrt(vals + r)
        Gw = (Gp * sc) @ vecs.T
        Qw = (Qp * sc) @ vecs.T
        Gw /= np.maximum(np.linalg.norm(Gw, axis=1, keepdims=True), 1e-12)
        Qw /= np.maximum(np.linalg.norm(Qw, axis=1, keepdims=True), 1e-12)
        out.append((Gw, Qw))
    return out


def fast_k_reciprocal_rerank(S_qg, S_gg, S_qq, k1=8, k2=2, lambda_value=0.55):
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


def _glob_znorm(A):
    return (A - A.mean()) / max(A.std(), 1e-6)


def compute_multi_space_views(gal_df, que_df, sp):
    """Build all complementary similarity views once per split (shared by inductive and k-recip branches)."""
    G_raw = features[gal_df["feat_idx"].values]
    Q_raw = features[que_df["feat_idx"].values]
    gal_ids = gal_df["identity"].values
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

    (G_l02, Q_l02), (G_l05, Q_l05), (G_l30, Q_l30) = whiten_multi(
        G_raw, Q_raw, gal_ids, regs=(0.0002, 0.0006, 0.003), supervised=True
    )
    [(G_pca, Q_pca)] = whiten_multi(G_raw, Q_raw, regs=(0.003,), supervised=False)

    G_hyb = 0.40 * G_l05 + 0.25 * G_l02 + 0.20 * G_l30 + 0.15 * G_pca
    G_hyb /= np.maximum(np.linalg.norm(G_hyb, axis=1, keepdims=True), 1e-12)
    Q_hyb = 0.40 * Q_l05 + 0.25 * Q_l02 + 0.20 * Q_l30 + 0.15 * Q_pca
    Q_hyb /= np.maximum(np.linalg.norm(Q_hyb, axis=1, keepdims=True), 1e-12)

    S_raw = (Q_raw @ G_raw.T) - pen * opp_broad + 0.010 * same_view
    S_l02 = (Q_l02 @ G_l02.T) - pen * opp_broad + 0.015 * same_view
    S_l30 = (Q_l30 @ G_l30.T) - pen * opp_broad + 0.015 * same_view
    S_pca = (Q_pca @ G_pca.T) - pen * opp_broad + 0.010 * same_view
    S_hyb = (Q_hyb @ G_hyb.T) - pen * opp_broad + 0.015 * same_view

    # Inductive Alpha-QE + Same-Side Gallery DBA
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

    # Global & Viewpoint-split Centroids (left / right / other)
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

    # Balanced Kernel Ridge Classifier (Primal solve when N_g > D)
    Y = np.zeros((len(G_hyb), M), dtype=np.float32)
    Y[np.arange(len(G_hyb)), id_inv] = 1.0 / np.sqrt(id_counts[id_inv])
    if len(G_hyb) > G_hyb.shape[1]:
        W_ridge = np.linalg.solve((G_hyb.T @ G_hyb) + 6.0 * np.eye(G_hyb.shape[1], dtype=np.float32), G_hyb.T @ Y)
        sc_ridge = Q_hyb @ W_ridge
    else:
        Ab = np.linalg.solve((G_hyb @ G_hyb.T) + 6.0 * np.eye(len(G_hyb), dtype=np.float32), Y)
        sc_ridge = (Q_hyb @ G_hyb.T) @ Ab

    # CSLS Hubness Suppression
    S_gg_h = G_hyb @ G_hyb.T
    r_G = np.sort(S_gg_h, axis=1)[:, -16:-1].mean(axis=1)
    S_csls_img = 2.0 * S_hyb - 0.55 * r_G[None, :]
    r_Q = np.sort(sc_hyb, axis=1)[:, -15:].mean(axis=1)
    sc_csls = np.column_stack([S_csls_img[:, idx].max(axis=1) for idx in class_to_idx]) - 0.55 * r_Q[:, None]

    # k-Reciprocal Jaccard similarity
    S_fk = fast_k_reciprocal_rerank(Q_hyb @ G_hyb.T, S_gg_h, Q_hyb @ Q_hyb.T, k1=8, k2=2, lambda_value=0.55) - pen * opp_broad
    sc_fk = np.column_stack([S_fk[:, idx].max(axis=1) for idx in class_to_idx])

    comp_ind = (
        0.35 * _glob_znorm(sc_hyb)
        + 0.15 * _glob_znorm(sc_cent)
        + 0.20 * _glob_znorm(sc_vcent)
        + 0.20 * _glob_znorm(sc_csls)
        + 0.10 * _glob_znorm(sc_ridge)
        + 0.15 * _glob_znorm(sc_dba)
    )
    comp_krec = 0.78 * comp_ind + 0.28 * _glob_znorm(sc_fk)

    return {
        "unique_ids": unique_ids, "id_counts": id_counts,
        "sc_raw": sc_raw, "sc_l02": sc_l02, "sc_l30": sc_l30, "sc_pca": sc_pca,
        "sc_hyb": sc_hyb, "sc_hyb_mean": sc_hyb_mean, "sc_hyb_top2": sc_hyb_top2,
        "sc_aqe": sc_aqe, "sc_dba": sc_dba, "sc_cent": sc_cent, "sc_vcent": sc_vcent,
        "sc_ridge": sc_ridge, "sc_csls": sc_csls, "sc_fkrec": sc_fk,
        "comp_ind": comp_ind, "comp_krec": comp_krec,
        "z_hyb": _row_znorm(sc_hyb), "z_l30": _row_znorm(sc_l30), "z_pca": _row_znorm(sc_pca),
        "z_csls": _row_znorm(sc_csls), "z_cent": _row_znorm(sc_cent), "z_vcent": _row_znorm(sc_vcent),
        "z_ridge": _row_znorm(sc_ridge), "z_fkrec": _row_znorm(sc_fk),
        "z_comp_ind": _row_znorm(comp_ind), "z_comp_krec": _row_znorm(comp_krec),
        "has_same_view": has_same_view, "opp_ratio": opp_ratio, "strong_hit_frac": strong_hit_frac, "r_Q": r_Q,
    }


COLS_A_IND = [
    "raw", "l02", "l30", "pca", "hyb", "hyb_mean", "hyb_top2", "aqe", "dba",
    "cent", "vcent", "ridge", "csls", "comp",
    "z_hyb", "z_l30", "z_pca", "z_csls", "z_cent", "z_vcent", "z_ridge", "z_comp",
    "gap_top1_hyb", "gap_next_hyb", "gap_top1_csls", "gap_top1_comp", "rank",
    "hyb_minus_cent", "hyb_minus_vcent", "log_gal_cnt", "has_same_view", "opp_ratio", "strong_hit_frac",
    "top1_z_hyb", "top1_top2_hyb", "top1_z_csls", "top1_top2_csls",
]

COLS_B_IND = [
    "hyb_1", "hyb_2", "hyb_3", "hyb_5", "hyb_gap12", "hyb_gap15", "hyb_ent",
    "l02_1", "l30_1", "pca_1", "raw_1", "aqe_1", "dba_1",
    "z_hyb_1", "z_l30_1", "z_pca_1", "z_csls_1", "z_cent_1", "z_vcent_1", "z_ridge_1", "z_comp_1",
    "z_hyb_gap12", "z_csls_gap12", "csls_1", "csls_gap12", "cent_1", "vcent_1", "ridge_1",
    "hyb_cent_diff", "hyb_vcent_diff", "has_same_v_1", "opp_ratio_1", "strong_hit_1", "r_Q",
    "cand_prob_1", "cand_prob_2", "cand_prob_gap12", "cand_prob_sum3", "cand_prob_sum5",
]


def _top_stats(A, tau=0.05):
    p5 = np.sort(A, axis=1)[:, -5:][:, ::-1]
    ex = np.exp((p5 - p5[:, :1]) / tau)
    ent = -np.sum((ex / ex.sum(axis=1, keepdims=True)) * np.log((ex / ex.sum(axis=1, keepdims=True)) + 1e-12), axis=1)
    return p5, ent


def build_stage_a(views, que_labels=None, K=12, with_krec=False):
    comp = views["comp_krec"] if with_krec else views["comp_ind"]
    z_comp = views["z_comp_krec"] if with_krec else views["z_comp_ind"]
    N_q, M = comp.shape
    K = min(K, M)

    topK_idx = np.argpartition(comp, -K, axis=1)[:, -K:]
    cands = np.take_along_axis(topK_idx, np.argsort(-np.take_along_axis(comp, topK_idx, axis=1), axis=1), axis=1)
    ri = np.arange(N_q)[:, None]

    v_hyb, v_csls, v_comp = views["sc_hyb"][ri, cands], views["sc_csls"][ri, cands], comp[ri, cands]
    t1_hyb, t2_hyb = views["sc_hyb"].max(axis=1), np.partition(views["sc_hyb"], -2, axis=1)[:, -2]
    t1_csls, t2_csls = views["sc_csls"].max(axis=1), np.partition(views["sc_csls"], -2, axis=1)[:, -2]
    t1_z_hyb, t1_z_csls = views["z_hyb"].max(axis=1), views["z_csls"].max(axis=1)

    next_hyb = np.empty_like(v_hyb)
    next_hyb[:, :-1] = v_hyb[:, :-1] - v_hyb[:, 1:]
    next_hyb[:, -1] = 0.0
    ranks = np.broadcast_to(np.arange(K, dtype=np.float32)[None, :], (N_q, K))

    mat = np.stack([
        views["sc_raw"][ri, cands], views["sc_l02"][ri, cands], views["sc_l30"][ri, cands],
        views["sc_pca"][ri, cands], v_hyb, views["sc_hyb_mean"][ri, cands], views["sc_hyb_top2"][ri, cands],
        views["sc_aqe"][ri, cands], views["sc_dba"][ri, cands],
        views["sc_cent"][ri, cands], views["sc_vcent"][ri, cands], views["sc_ridge"][ri, cands], v_csls, v_comp,
        views["z_hyb"][ri, cands], views["z_l30"][ri, cands], views["z_pca"][ri, cands],
        views["z_csls"][ri, cands], views["z_cent"][ri, cands], views["z_vcent"][ri, cands],
        views["z_ridge"][ri, cands], z_comp[ri, cands],
        t1_hyb[:, None] - v_hyb, next_hyb, t1_csls[:, None] - v_csls, v_comp[:, :1] - v_comp, ranks,
        v_hyb - views["sc_cent"][ri, cands], v_hyb - views["sc_vcent"][ri, cands],
        np.log1p(views["id_counts"][cands]).astype(np.float32),
        views["has_same_view"][ri, cands], views["opp_ratio"][ri, cands], views["strong_hit_frac"][ri, cands],
        np.broadcast_to(t1_z_hyb[:, None], (N_q, K)), np.broadcast_to((t1_hyb - t2_hyb)[:, None], (N_q, K)),
        np.broadcast_to(t1_z_csls[:, None], (N_q, K)), np.broadcast_to((t1_csls - t2_csls)[:, None], (N_q, K)),
    ], axis=-1)

    X = pd.DataFrame(mat.reshape(N_q * K, -1), columns=COLS_A_IND)
    if with_krec:
        v_fk = views["sc_fkrec"][ri, cands]
        t1_fk, t2_fk = views["sc_fkrec"].max(axis=1), np.partition(views["sc_fkrec"], -2, axis=1)[:, -2]
        X["fkrec"] = v_fk.reshape(-1)
        X["z_fkrec"] = views["z_fkrec"][ri, cands].reshape(-1)
        X["gap_top1_fkrec"] = (t1_fk[:, None] - v_fk).reshape(-1)
        X["top1_top2_fkrec"] = np.broadcast_to((t1_fk - t2_fk)[:, None], (N_q, K)).reshape(-1)

    y = (views["unique_ids"][cands] == que_labels[:, None]).astype(int).reshape(-1) if que_labels is not None else None
    return X, y, np.full(N_q, K, dtype=int), cands


def build_stage_b(views, cand_probs, cands, with_krec=False):
    z_comp = views["z_comp_krec"] if with_krec else views["z_comp_ind"]
    N_q = cands.shape[0]
    p_hyb, ent_hyb = _top_stats(views["sc_hyb"], tau=0.05)
    p_z_hyb, _ = _top_stats(views["z_hyb"], tau=1.0)
    p_z_csls, _ = _top_stats(views["z_csls"], tau=1.0)
    p_csls, _ = _top_stats(views["sc_csls"], tau=0.05)

    order_p = np.argsort(-cand_probs, axis=1)
    sorted_p = np.take_along_axis(cand_probs, order_p, axis=1)
    best_cand = cands[np.arange(N_q), order_p[:, 0]]
    ri = np.arange(N_q)

    hyb_1, cent_1, vcent_1 = views["sc_hyb"][ri, best_cand], views["sc_cent"][ri, best_cand], views["sc_vcent"][ri, best_cand]
    mat = np.column_stack([
        p_hyb[:, 0], p_hyb[:, 1], p_hyb[:, 2], p_hyb[:, 4], p_hyb[:, 0] - p_hyb[:, 1], p_hyb[:, 0] - p_hyb[:, 4], ent_hyb,
        views["sc_l02"][ri, best_cand], views["sc_l30"][ri, best_cand], views["sc_pca"][ri, best_cand],
        views["sc_raw"][ri, best_cand], views["sc_aqe"][ri, best_cand], views["sc_dba"][ri, best_cand],
        views["z_hyb"][ri, best_cand], views["z_l30"][ri, best_cand], views["z_pca"][ri, best_cand],
        views["z_csls"][ri, best_cand], views["z_cent"][ri, best_cand], views["z_vcent"][ri, best_cand],
        views["z_ridge"][ri, best_cand], z_comp[ri, best_cand],
        p_z_hyb[:, 0] - p_z_hyb[:, 1], p_z_csls[:, 0] - p_z_csls[:, 1],
        p_csls[:, 0], p_csls[:, 0] - p_csls[:, 1], cent_1, vcent_1, views["sc_ridge"][ri, best_cand],
        hyb_1 - cent_1, hyb_1 - vcent_1,
        views["has_same_view"][ri, best_cand], views["opp_ratio"][ri, best_cand], views["strong_hit_frac"][ri, best_cand], views["r_Q"],
        sorted_p[:, 0], sorted_p[:, 1], sorted_p[:, 0] - sorted_p[:, 1], sorted_p[:, :3].sum(axis=1), sorted_p[:, :5].sum(axis=1),
    ])
    X = pd.DataFrame(mat, columns=COLS_B_IND)
    if with_krec:
        p5_fk, ent_fk = _top_stats(views["sc_fkrec"], tau=0.05)
        X["fk_1"] = p5_fk[:, 0]
        X["fk_2"] = p5_fk[:, 1]
        X["fk_gap12"] = p5_fk[:, 0] - p5_fk[:, 1]
        X["fk_ent"] = ent_fk
        X["z_fk_1"] = views["z_fkrec"][ri, best_cand]
    return X


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


def bayes_decode(uid, cands, cand_sc, p_nov, bias=1.0):
    s_pow = np.maximum(cand_sc, 1e-6)
    p_cond = s_pow / np.maximum(s_pow.sum(axis=1, keepdims=True), 1e-8)
    pn = np.clip(p_nov * bias, 0.001, 0.999)
    pk = (1.0 - pn) * 0.96
    preds = []
    for i in range(len(cands)):
        items = [("new_individual", pn[i])] + [(uid[cands[i, k]], pk[i] * p_cond[i, k]) for k in range(cands.shape[1])]
        items.sort(key=lambda x: -x[1])
        preds.append([it[0] for it in items[:5]])
    return preds


def fit_branch_and_predict(fold_views, fold_labels, v_test, K=12, with_krec=False):
    n_folds = len(fold_views)
    fds = []
    for vi, li in zip(fold_views, fold_labels):
        Xa, ya, ga, ca = build_stage_a(vi, li, K=K, with_krec=with_krec)
        fds.append((vi, Xa, ya, ga, ca, (li == "new_individual").astype(int), li))

    oof_Xb, oof_yb, oof_sc, oof_ca, oof_uid, oof_l = [], [], [], [], [], []
    for i in range(n_folds):
        sa = train_stage_a(
            pd.concat([fds[j][1] for j in range(n_folds) if j != i], ignore_index=True),
            np.concatenate([fds[j][2] for j in range(n_folds) if j != i]),
            np.concatenate([fds[j][3] for j in range(n_folds) if j != i]),
            seed=42 + i,
        )
        sc_val = predict_stage_a(sa, fds[i][1], K)
        oof_Xb.append(build_stage_b(fds[i][0], sc_val, fds[i][4], with_krec=with_krec))
        oof_yb.append(fds[i][5]); oof_sc.append(sc_val); oof_ca.append(fds[i][4])
        oof_uid.append(fds[i][0]["unique_ids"]); oof_l.append(fds[i][6])

    Xa_full = pd.concat([f[1] for f in fds], ignore_index=True)
    ya_full = np.concatenate([f[2] for f in fds])
    ga_full = np.concatenate([f[3] for f in fds])
    stage_a_ens = [train_stage_a(Xa_full, ya_full, ga_full, seed=s) for s in [42, 100, 2024]]

    Xb_full = pd.concat(oof_Xb, ignore_index=True)
    yb_full = np.concatenate(oof_yb)
    offsets = np.cumsum([0] + [len(x) for x in oof_Xb])
    oof_p_nov = np.zeros(len(X_b := Xb_full))
    for i in range(n_folds):
        sb = train_stage_b(
            pd.concat([oof_Xb[j] for j in range(n_folds) if j != i], ignore_index=True),
            np.concatenate([oof_yb[j] for j in range(n_folds) if j != i]),
            seed=100 + i,
        )
        oof_p_nov[offsets[i]:offsets[i + 1]] = predict_stage_b(sb, oof_Xb[i])

    bias_grid = [0.85, 0.95, 1.00, 1.05, 1.12, 1.20, 1.28, 1.35] if not with_krec else [0.85, 0.95, 1.00, 1.05, 1.12, 1.20, 1.28]
    best_bias, best_map5 = 1.0, 0.0
    for b in bias_grid:
        sc_f = [map5_score(bayes_decode(oof_uid[i], oof_ca[i], oof_sc[i], oof_p_nov[offsets[i]:offsets[i + 1]], bias=b), oof_l[i]) for i in range(n_folds)]
        if np.mean(sc_f) > best_map5:
            best_map5, best_bias = np.mean(sc_f), b

    stage_b_ens = [train_stage_b(X_b, yb_full, seed=s) for s in [42, 100, 2024]]
    Xa_te, _, _, ca_te = build_stage_a(v_test, K=K, with_krec=with_krec)
    cand_te = np.mean([predict_stage_a(sa, Xa_te, K) for sa in stage_a_ens], axis=0)
    Xb_te = build_stage_b(v_test, cand_te, ca_te, with_krec=with_krec)
    p_nov_te = np.mean([predict_stage_b(sb, Xb_te) for sb in stage_b_ens], axis=0)
    auc = roc_auc_score(yb_full, oof_p_nov)
    return ca_te, cand_te, p_nov_te, best_bias, best_map5, auc


m3_cv = {}
m3_preds = {}
K = 12
sp_configs = [("hyena", 0.25, 0.38), ("leopard", 0.26, 0.38)]

for sp, hr, er in sp_configs:
    sp_tr = train_df[train_df["species"] == sp].reset_index(drop=True)
    sp_te = test_df[test_df["species"] == sp].reset_index(drop=True)
    sp_tr["enc"] = assign_encounters(sp_tr)

    fold_views, fold_labels = [], []
    for s in [11, 22, 33, 44, 55]:
        gi, qi, li = make_open_set_split(sp_tr, holdout_id_ratio=hr, query_enc_ratio=er, max_per_id=10, seed=s)
        fold_views.append(compute_multi_space_views(gi, qi, sp))
        fold_labels.append(li)

    v_test = compute_multi_space_views(sp_tr, sp_te, sp)
    uids = v_test["unique_ids"]

    ca_i, sc_i, pn_i, bi_i, map_i, auc_i = fit_branch_and_predict(fold_views, fold_labels, v_test, K=K, with_krec=False)
    ca_k, sc_k, pn_k, bk_k, map_k, auc_k = fit_branch_and_predict(fold_views, fold_labels, v_test, K=K, with_krec=True)
    m3_cv[sp] = 0.55 * map_i + 0.45 * map_k
    print(f"  [Model 3 - {sp.upper()}] Inductive OOF={map_i:.4f} (AUC={auc_i:.4f}) | Hybrid+kRec OOF={map_k:.4f} (AUC={auc_k:.4f})")

    sc_i_n = np.maximum(sc_i, 1e-6)
    sc_i_n /= np.maximum(sc_i_n.sum(axis=1, keepdims=True), 1e-8)
    sc_k_n = np.maximum(sc_k, 1e-6)
    sc_k_n /= np.maximum(sc_k_n.sum(axis=1, keepdims=True), 1e-8)
    pn_i_adj = np.clip(pn_i * bi_i, 0.001, 0.999)
    pn_k_adj = np.clip(pn_k * bk_k, 0.001, 0.999)
    pn_blend = 0.55 * pn_i_adj + 0.45 * pn_k_adj

    for qi_idx, qid in enumerate(sp_te["id"].values):
        pool = {}
        pk_i = (1.0 - pn_i_adj[qi_idx]) * 0.96
        pk_k = (1.0 - pn_k_adj[qi_idx]) * 0.96
        for r in range(K):
            ci_name = str(uids[ca_i[qi_idx, r]])
            ck_name = str(uids[ca_k[qi_idx, r]])
            pool[ci_name] = pool.get(ci_name, 0.0) + 0.55 * (pk_i * sc_i_n[qi_idx, r])
            pool[ck_name] = pool.get(ck_name, 0.0) + 0.45 * (pk_k * sc_k_n[qi_idx, r])
        items = [("new_individual", float(pn_blend[qi_idx]))] + list(pool.items())
        items.sort(key=lambda x: -x[1])
        m3_preds[qid] = " ".join([it[0] for it in items[:5]])


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
    {"Model": "Model 3 (Multi-Space GBDT)",     "Hyena CV": f"{m3_cv['hyena']:.4f}", "Leopard CV": f"{m3_cv['leopard']:.4f}", "Weighted CV": f"{w_avg(m3_cv):.4f}", "Kaggle Public": "0.8177"},
])
print("\n" + "=" * 72)
print(" MODEL PROGRESSION SUMMARY (MAP@5)")
print("=" * 72)
print(summary.to_string(index=False))
print(f"\nCompleted full pipeline in {time.time() - t0:.1f}s.")
