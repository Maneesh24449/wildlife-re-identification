"""
Fast, vectorized, gallery-only local descriptor encoders for the final Model 3:
  1. FisherVectorEncoder (PCA-32 + Diagonal GMM K=32 + SSR + L2 norm)
  2. LNBNN (Local Naive Bayes Nearest Neighbor via chunked float32 matmul)

Rules compliance:
  - Uses only supplied SIFT descriptors (no raw images, no OpenCV image I/O, no DL).
  - All PCA / GMM / index fits use gallery data ONLY within each fold.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def load_all_rootsift(image_ids, sift_dir, max_desc=500, seed=1987):
    """
    Load and L2-normalize RootSIFT descriptors for a list of image_ids.
    Note: RootSIFT is L1-normalized then sqrt'd, which automatically makes
    every descriptor have unit L2 norm! (sum(sqrt(d/sum(d))^2) = sum(d/sum(d)) = 1).
    Returns dict: image_id -> (N_i, 128) float32 array (or None if missing/empty).
    """
    rng = np.random.RandomState(seed)
    out = {}
    for img_id in image_ids:
        part = "train" if str(img_id).startswith("tr_") else "test"
        p = sift_dir / f"sift_features_{part}.npz" / f"{img_id}_desc.npy"
        if not p.exists():
            out[img_id] = None
            continue
        desc = np.load(p, allow_pickle=False)
        if desc.ndim != 2 or desc.shape[1] != 128 or len(desc) == 0:
            out[img_id] = None
            continue
        desc = desc.astype(np.float32)
        s = desc.sum(axis=1, keepdims=True)
        keep = s[:, 0] > 0
        if not keep.any():
            out[img_id] = None
            continue
        desc = desc[keep]
        s = s[keep]
        if max_desc is not None and len(desc) > max_desc:
            idx = rng.choice(len(desc), max_desc, replace=False)
            desc = desc[idx]
            s = s[idx]
        # RootSIFT = sqrt(d / ||d||_1) -> unit L2 norm!
        rs = np.sqrt(np.maximum(desc / s, 0.0)).astype(np.float32)
        out[img_id] = rs
    return out


class FastFisherVectorEncoder:
    """
    Vectorized Fisher Vector encoder using PCA + diagonal GMM.
    """
    def __init__(self, n_components=32, pca_dim=32, seed=1987):
        self.K = n_components
        self.pca_dim = pca_dim
        self.seed = seed
        self.pca = None
        self.gmm = None

    def fit(self, gal_df, desc_cache, n_sample_enc=2, desc_per_img=48):
        """
        Fit PCA and diagonal GMM strictly on gallery images.
        Samples at most n_sample_enc encounters per identity, 1 image per encounter,
        and desc_per_img descriptors per image (as specified in MASTER_PLAN.md).
        """
        rng = np.random.RandomState(self.seed)
        samples = []
        for uid, grp in gal_df.groupby("identity"):
            if "enc" in grp.columns:
                encs = grp["enc"].unique()
                sel_encs = rng.choice(encs, min(n_sample_enc, len(encs)), replace=False)
                chosen_ids = [grp[grp["enc"] == ec]["id"].iloc[0] for ec in sel_encs]
            else:
                chosen_ids = grp["id"].iloc[:n_sample_enc].tolist()
            for img_id in chosen_ids:
                d = desc_cache.get(img_id)
                if d is None or len(d) == 0:
                    continue
                if len(d) > desc_per_img:
                    idx = rng.choice(len(d), desc_per_img, replace=False)
                    d = d[idx]
                samples.append(d)

        if len(samples) < 5:
            return self

        X = np.vstack(samples).astype(np.float64)
        dim = min(self.pca_dim, X.shape[1], X.shape[0] - 1)
        self.pca = PCA(n_components=dim, whiten=False, random_state=self.seed)
        X_proj = self.pca.fit_transform(X)

        self.gmm = GaussianMixture(
            n_components=self.K,
            covariance_type="diag",
            max_iter=150,
            tol=1e-3,
            reg_covar=1e-4,
            random_state=self.seed,
        )
        self.gmm.fit(X_proj)
        return self

    def encode_batch(self, image_ids, desc_cache):
        """
        Vectorized Fisher Vector encoding for a list of image_ids.
        Returns (N_images, 2 * K * dim) float32 L2-normalized matrix.
        """
        dim = self.pca.n_components_ if self.pca is not None else self.pca_dim
        fv_dim = 2 * self.K * dim
        out = np.zeros((len(image_ids), fv_dim), dtype=np.float32)
        if self.pca is None or self.gmm is None:
            return out

        mu = self.gmm.means_                          # (K, D)
        var = np.maximum(self.gmm.covariances_, 1e-10) # (K, D)
        inv_sigma = 1.0 / np.sqrt(var)                # (K, D)
        pi = np.maximum(self.gmm.weights_, 1e-10)     # (K,)
        inv_sqrt_pi = 1.0 / np.sqrt(pi)               # (K,)
        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * pi)        # (K,)

        for i, img_id in enumerate(image_ids):
            d = desc_cache.get(img_id)
            if d is None or len(d) == 0:
                continue
            proj = self.pca.transform(d.astype(np.float64))  # (N, D)
            gamma = self.gmm.predict_proba(proj)             # (N, K)
            N = float(len(proj))

            # Vectorized 1st and 2nd order statistics:
            # S0[k] = sum_n gamma[n, k]
            # S1[k, d] = sum_n gamma[n, k] * proj[n, d]
            # S2[k, d] = sum_n gamma[n, k] * proj[n, d]^2
            S0 = gamma.sum(axis=0)                           # (K,)
            S1 = gamma.T @ proj                              # (K, D)
            S2 = gamma.T @ (proj * proj)                     # (K, D)

            fv1 = ((S1 - S0[:, None] * mu) * inv_sigma) * inv_sqrt_pi[:, None]
            fv2 = ((S2 - 2.0 * mu * S1 + S0[:, None] * (mu * mu - var)) / var) * inv_sqrt_2pi[:, None]

            fv = np.concatenate([fv1.ravel(), fv2.ravel()]) / N
            # Signed square root + L2 normalization
            fv = np.sign(fv) * np.sqrt(np.abs(fv))
            norm = np.linalg.norm(fv)
            if norm > 1e-8:
                fv /= norm
            out[i] = fv.astype(np.float32)

        return out


def compute_fv_identity_scores(gal_df, que_df, desc_cache, n_components=32, pca_dim=32, seed=1987):
    """
    Fit FV on gallery only, encode gallery & query, return:
      sc_fv_max: (N_q, M) max cosine per identity
      sc_fv_mean: (N_q, M) mean cosine per identity
      unique_ids: (M,) sorted identity strings
    """
    enc = FastFisherVectorEncoder(n_components=n_components, pca_dim=pca_dim, seed=seed)
    enc.fit(gal_df, desc_cache)
    G_fv = enc.encode_batch(gal_df["id"].values, desc_cache)
    Q_fv = enc.encode_batch(que_df["id"].values, desc_cache)

    gal_ids = gal_df["identity"].to_numpy(dtype=str)
    unique_ids, id_inv = np.unique(gal_ids, return_inverse=True)
    M = len(unique_ids)
    S = Q_fv @ G_fv.T  # (N_q, N_gal)

    # Zero out invalid (zero-norm) images so they don't pollute max/mean
    g_valid = np.linalg.norm(G_fv, axis=1) > 0
    q_valid = np.linalg.norm(Q_fv, axis=1) > 0

    sc_max = np.zeros((len(que_df), M), dtype=np.float32)
    sc_mean = np.zeros((len(que_df), M), dtype=np.float32)
    for c in range(M):
        mask = (id_inv == c) & g_valid
        if mask.any():
            sub = S[:, mask]
            sc_max[:, c] = sub.max(axis=1)
            sc_mean[:, c] = sub.mean(axis=1)
    sc_max[~q_valid, :] = 0.0
    sc_mean[~q_valid, :] = 0.0
    return sc_max, sc_mean, unique_ids


def compute_lnbnn_identity_scores(gal_df, que_df, desc_cache, k_bg=10, max_desc_gal=64,
                                  max_desc_que=64, pca_dim=32, chunk_size=4096, seed=1987):
    """
    Fast exact LNBNN (HotSpotter-style) using gallery-fitted PCA + chunked float32
    dot products on unit-L2-norm RootSIFT descriptors: ||a - b||^2 = 2 - 2*(a . b).
    """
    rng = np.random.RandomState(seed)
    gal_ids = gal_df["identity"].to_numpy(dtype=str)
    unique_ids, id_inv = np.unique(gal_ids, return_inverse=True)
    M = len(unique_ids)

    g_list, g_lbl = [], []
    for idx_g, img_id in enumerate(gal_df["id"].values):
        d = desc_cache.get(img_id)
        if d is None or len(d) == 0:
            continue
        if len(d) > max_desc_gal:
            sel = rng.choice(len(d), max_desc_gal, replace=False)
            d = d[sel]
        g_list.append(d)
        g_lbl.append(np.full(len(d), id_inv[idx_g], dtype=np.int32))

    if not g_list:
        return np.zeros((len(que_df), M), dtype=np.float32), unique_ids

    G_raw = np.vstack(g_list).astype(np.float32)  # (N_g_desc, 128)
    G_lbl = np.concatenate(g_lbl)                 # (N_g_desc,)

    # Optional gallery-only PCA projection for 3x speedup & denoising
    pca_model = None
    if pca_dim is not None and pca_dim < G_raw.shape[1]:
        n_samp = min(50000, len(G_raw))
        samp_idx = rng.choice(len(G_raw), n_samp, replace=False)
        pca_model = PCA(n_components=pca_dim, random_state=seed)
        pca_model.fit(G_raw[samp_idx])
        G_all = pca_model.transform(G_raw).astype(np.float32)
        gn = np.linalg.norm(G_all, axis=1, keepdims=True)
        gn[gn < 1e-8] = 1.0
        G_all /= gn
    else:
        G_all = G_raw

    N_g = len(G_all)
    k_search = min(k_bg + 1, N_g)
    id_img_counts = np.maximum(np.bincount(id_inv, minlength=M).astype(np.float32), 1.0)
    sc_lnbnn = np.zeros((len(que_df), M), dtype=np.float32)

    # Stack all query descriptors for fast BLAS level-3 GEMM
    q_list, q_img_idx = [], []
    for qi, img_id in enumerate(que_df["id"].values):
        d = desc_cache.get(img_id)
        if d is None or len(d) == 0:
            continue
        if len(d) > max_desc_que:
            sel = rng.choice(len(d), max_desc_que, replace=False)
            d = d[sel]
        q_list.append(d)
        q_img_idx.append(np.full(len(d), qi, dtype=np.int32))

    if not q_list:
        return sc_lnbnn, unique_ids

    Q_all = np.vstack(q_list).astype(np.float32)
    Q_img = np.concatenate(q_img_idx)
    if pca_model is not None:
        Q_all = pca_model.transform(Q_all).astype(np.float32)
        qn = np.linalg.norm(Q_all, axis=1, keepdims=True)
        qn[qn < 1e-8] = 1.0
        Q_all /= qn

    N_q_desc = len(Q_all)
    q_batch = 256
    best_sims = np.full((N_q_desc, k_search), -2.0, dtype=np.float32)
    best_lbls = np.full((N_q_desc, k_search), -1, dtype=np.int32)

    for q_start in range(0, N_q_desc, q_batch):
        q_end = min(q_start + q_batch, N_q_desc)
        qb = Q_all[q_start:q_end]
        b_sims = np.full((q_end - q_start, k_search), -2.0, dtype=np.float32)
        b_lbls = np.full((q_end - q_start, k_search), -1, dtype=np.int32)

        for g_start in range(0, N_g, chunk_size):
            g_end = min(g_start + chunk_size, N_g)
            sims = qb @ G_all[g_start:g_end].T
            k_loc = min(k_search, g_end - g_start)
            loc_idx = np.argpartition(sims, -k_loc, axis=1)[:, -k_loc:]
            loc_sim = np.take_along_axis(sims, loc_idx, axis=1)
            loc_lbl = G_lbl[g_start + loc_idx]

            comb_sim = np.hstack([b_sims, loc_sim])
            comb_lbl = np.hstack([b_lbls, loc_lbl])
            top_k = np.argpartition(comb_sim, -k_search, axis=1)[:, -k_search:]
            b_sims = np.take_along_axis(comb_sim, top_k, axis=1)
            b_lbls = np.take_along_axis(comb_lbl, top_k, axis=1)

        order = np.argsort(-b_sims, axis=1)
        best_sims[q_start:q_end] = np.take_along_axis(b_sims, order, axis=1)
        best_lbls[q_start:q_end] = np.take_along_axis(b_lbls, order, axis=1)

    # Accumulate LNBNN votes across all query descriptors
    bg_sim = best_sims[:, -1]
    votes = np.maximum(0.0, 2.0 * (best_sims[:, :-1] - bg_sim[:, None]))

    # For each query descriptor, only count the best match per identity
    seen_pairs = set()
    # Vectorized per-k accumulation using flat index = Q_img * M + c
    flat_seen = np.zeros(len(que_df) * M, dtype=bool)
    # Wait: seen must be per-query-descriptor, not per-query-image!
    # A query image has multiple descriptors that can vote for the same identity!
    seen_qd = np.zeros((N_q_desc, M), dtype=bool)
    for k_idx in range(best_lbls.shape[1] - 1):
        lbls_k = best_lbls[:, k_idx]
        v_k = votes[:, k_idx]
        valid_k = (lbls_k >= 0) & (v_k > 0)
        if not valid_k.any():
            continue
        qd_rows = np.flatnonzero(valid_k)
        c_cols = lbls_k[valid_k]
        not_seen = ~seen_qd[qd_rows, c_cols]
        if not_seen.any():
            qr = qd_rows[not_seen]
            cc = c_cols[not_seen]
            seen_qd[qr, cc] = True
            np.add.at(sc_lnbnn, (Q_img[qr], cc), v_k[valid_k][not_seen])

    sc_lnbnn /= np.sqrt(id_img_counts)[None, :]

    # Row-wise L2 normalization for clean scale
    norms = np.linalg.norm(sc_lnbnn, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    sc_lnbnn /= norms
    return sc_lnbnn, unique_ids
