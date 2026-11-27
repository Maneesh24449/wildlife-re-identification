"""Final Model 3 candidate scoring with gallery-trained Fisher vectors and LNBNN."""

from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from local_encoders import compute_fv_identity_scores, compute_lnbnn_identity_scores, load_all_rootsift
from sift_rerank import (GLOBAL_COLUMNS, LOCAL_COLUMNS, NOVEL_LABEL, SiftVerifier,
                        candidate_evidence, query_evidence)


BUDGET = 4
DIMENSIONS = 192
SHRINKAGE = 0.18


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def row_z(x):
    return (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)


def encounter_vectors(x, gallery):
    groups = gallery.groupby("enc", sort=True).indices.values()
    reps, labels = [], []
    identities = gallery.identity.to_numpy(dtype=str)
    for rows in groups:
        label = identities[rows[0]]
        if not np.all(identities[rows] == label):
            raise ValueError("Mixed identities in a gallery encounter")
        reps.append(unit(x[rows].mean(axis=0)))
        labels.append(label)
    return np.asarray(reps), np.asarray(labels)


def grouped_max(x, assignment, n):
    return np.column_stack([x[:, assignment == k].max(axis=1) for k in range(n)])


def shrink(cov):
    return (1-SHRINKAGE) * cov + SHRINKAGE * (np.trace(cov) / len(cov)) * np.eye(len(cov))


class GaussianIdentity:
    def fit(self, encounters, classes, n_ids):
        self.pca = PCA(n_components=min(DIMENSIONS, len(encounters)-1), svd_solver="randomized", random_state=22)
        x = self.pca.fit(encounters).transform(encounters).astype(np.float64)
        counts = np.bincount(classes, minlength=n_ids)
        means = np.vstack([x[classes == i].mean(axis=0) for i in range(n_ids)])
        self.center = means.mean(axis=0)
        off = x - means[classes]
        expanded = counts[classes] > 1
        if not expanded.any():
            raise ValueError("The model requires repeated gallery identities")
        weighted = off[expanded] / np.sqrt(counts[classes[expanded], None]-1)
        within = shrink((weighted.T @ weighted) / (counts > 1).sum())
        demeaned = means - self.center
        between = (demeaned.T @ demeaned) / len(means) - np.mean(1/counts) * within
        eig, rotation = eigh(shrink(between), within, check_finite=True)
        self.between = np.maximum(eig, 0.025).astype(np.float32)
        self.rotation = rotation.astype(np.float32)
        self.center = self.center.astype(np.float32)
        self.proto = demeaned.astype(np.float32) @ self.rotation
        self.enc = (x.astype(np.float32) - self.center) @ self.rotation
        self.counts = counts
        return self

    def score(self, queries):
        latent = (self.pca.transform(queries) - self.center) @ self.rotation
        reference = (np.square(latent) / (1 + self.between)).sum(axis=1) + np.log1p(self.between).sum()

        def log_ratio(proto, n):
            posterior = self.between / (1 + n * self.between)
            mu = proto * (n * posterior)
            inv = 1 / (1 + posterior)
            score = (np.square(latent) @ inv.T - 2 * (latent @ (mu * inv).T)
                     + (np.square(mu) * inv + np.log1p(posterior)).sum(axis=1)[None, :])
            return 0.5 * (reference[:, None] - score)

        return log_ratio(self.proto, self.counts[:, None].astype(np.float32)), log_ratio(self.enc, np.ones((1, 1), np.float32))


@dataclass
class GalleryScores:
    raw: np.ndarray
    compatible: np.ndarray
    centered: np.ndarray
    likelihood: np.ndarray
    nearest: np.ndarray
    combined: np.ndarray
    images: np.ndarray
    encounters: np.ndarray


def score_gallery(stock, gallery, query, ids):
    features = stock["features"]
    x, q = (features[frame.feat_idx] for frame in (gallery, query))
    enc, enc_labels = encounter_vectors(x, gallery)
    n = len(ids)
    classes = np.searchsorted(ids, gallery.identity.to_numpy(dtype=str))
    enc_classes = np.searchsorted(ids, enc_labels)
    if not np.array_equal(np.unique(gallery.identity.to_numpy(dtype=str)), ids):
        raise ValueError("Gallery score identity order differs from the control")
    qside, gside = (stock["flank_side"](f.viewpoint) for f in (query, gallery))
    eligible = qside[:, None] * gside[None, :] != -1
    sim = q @ x.T
    raw = grouped_max(sim, classes, n)
    compatible = grouped_max(np.where(eligible, sim, -1.0), classes, n)
    center = np.vstack([enc[enc_classes == i].mean(axis=0) for i in range(n)]).mean(axis=0)
    centered = grouped_max(unit(q-center) @ unit(enc-center).T, enc_classes, n)
    likelihood, pair = GaussianIdentity().fit(enc, enc_classes, n).score(q)
    nearest = grouped_max(pair, enc_classes, n)
    combined = 0.50 * row_z(likelihood) + 0.25 * row_z(nearest) + 0.25 * row_z(centered)
    return GalleryScores(raw, compatible, centered, likelihood, nearest, combined,
                         np.bincount(classes, minlength=n), np.bincount(enc_classes, minlength=n))


def proposal_indices(values, stock_ranks, budget=BUDGET):
    eligible = values.copy()
    for ranks in stock_ranks:
        np.put_along_axis(eligible, ranks, -np.inf, axis=1)
    return np.argsort(-eligible, axis=1, kind="stable")[:, :budget]


def build_table_b2(stock, gallery, query, ids, ranks, evidence, extra_geometry,
                   sc_fv_max, sc_fv_mean, sc_lnbnn, add_fv=4, add_ln=4, verify_local=0):
    s = score_gallery(stock, gallery, query, ids)
    extra = proposal_indices(s.combined, ranks)
    plda = row_z(s.likelihood)
    near = row_z(s.nearest)
    context = []
    for scores in (s.raw, s.likelihood, s.nearest, s.centered, s.combined):
        best = np.sort(scores, axis=1)[:, -5:]
        context.extend([best[:, -1], best[:, -1] - best[:, -2], best[:, -1] - best[:, 0]])
    context.extend(query_evidence(evidence).to_numpy().T)
    context.extend([query.viewpoint.isna().to_numpy(), stock["flank_side"](query.viewpoint) == 0])

    if extra_geometry is not None:
        extra_table = extra_geometry(stock, gallery, query, ids, extra[:, :2])
        strong = extra_table[(extra_table.log_inliers >= np.log1p(4)) & (extra_table.inlier_fraction >= 0.35)
                             & (extra_table.spread >= 0.01)]
        per_query = strong.groupby("query")
        context.append(per_query.log_inliers.max().reindex(range(len(query)), fill_value=0).to_numpy())
        context.append(per_query.size().reindex(range(len(query)), fill_value=0).to_numpy())
        evidence = pd.concat([evidence, extra_table], ignore_index=True)

    queries = np.vstack(context).T
    fv_z = (sc_fv_max - sc_fv_max.mean(axis=1, keepdims=True)) / np.maximum(sc_fv_max.std(axis=1, keepdims=True), 1e-6)
    ln_z = (sc_lnbnn - sc_lnbnn.mean(axis=1, keepdims=True)) / np.maximum(sc_lnbnn.std(axis=1, keepdims=True), 1e-6)

    fv_ord = np.argsort(-sc_fv_max, axis=1)
    ln_ord = np.argsort(-sc_lnbnn, axis=1)

    # Collect proposals outside the stock and Gaussian shortlists.
    local_verify_list = []

    orders = []
    for i, branches in enumerate(zip(*ranks)):
        order = {}
        for b, indices in enumerate(branches):
            for rank, k in enumerate(indices):
                order.setdefault(int(k), [0.0, 0.0])[b] = 1 / (rank + 1)
        for k in extra[i]:
            order.setdefault(int(k), [0.0, 0.0])

        fv_added, ln_added = [], []
        for c in fv_ord[i]:
            if int(c) not in order and len(fv_added) < add_fv:
                order[int(c)] = [0.0, 0.0]
                fv_added.append(int(c))
        for c in ln_ord[i]:
            if int(c) not in order and len(ln_added) < add_ln:
                order[int(c)] = [0.0, 0.0]
                ln_added.append(int(c))
        orders.append(order)
        if verify_local > 0:
            v_row = (fv_added[:verify_local] + ln_added[:verify_local])
            while len(v_row) < 2 * verify_local:
                v_row.append(fv_added[0] if fv_added else int(extra[i][0]))
            local_verify_list.append(v_row[:2 * verify_local])

    if extra_geometry is not None and verify_local > 0 and local_verify_list:
        loc_arr = np.asarray(local_verify_list, dtype=int)
        # Call ProposalVerification directly on local proposals (up to 2*verify_local)
        loc_table = extra_geometry(stock, gallery, query, ids, loc_arr)
        evidence = pd.concat([evidence, loc_table], ignore_index=True)

    verifications = {(r["query"], r["candidate"]): [r[column] for column in LOCAL_COLUMNS]
                     for r in evidence.to_dict("records")}

    ranked_candidates, rows, groups = [], [], []
    for i, order in enumerate(orders):
        k = np.array(list(order))
        basic = np.array(list(order.values()))
        verified = np.array([verifications.get((i, j), [np.nan] * len(LOCAL_COLUMNS)) for j in k])
        search = np.column_stack((
            s.raw[i, k], s.raw[i, k] - s.raw[i].max(), s.compatible[i, k], s.centered[i, k],
            s.likelihood[i, k], s.nearest[i, k], plda[i, k], near[i, k], s.combined[i, k],
            s.combined[i, k] - s.combined[i].max(), np.log1p(s.images[k]), np.log1p(s.encounters[k]),
        ))
        local_f = np.column_stack((
            sc_fv_max[i, k], sc_fv_mean[i, k], fv_z[i, k], sc_fv_max[i, k] - sc_fv_max[i].max(),
            sc_lnbnn[i, k], ln_z[i, k], sc_lnbnn[i, k] - sc_lnbnn[i].max(),
            sc_fv_max[i, k] * sc_lnbnn[i, k],
        ))
        rows.append(np.column_stack((basic, search, verified, local_f, np.broadcast_to(queries[i], (len(k), queries.shape[1])))))
        ranked_candidates.append(ids[k])
        groups.append(len(k))
    return ranked_candidates, np.concatenate(rows), np.asarray(groups), queries


def branch_candidates(views, size=12):
    result = []
    for view in views:
        scores = view["comp"]
        indices = np.argpartition(scores, -size, axis=1)[:, -size:]
        order = np.argsort(-np.take_along_axis(scores, indices, axis=1), axis=1)
        result.append(np.take_along_axis(indices, order, axis=1))
    return tuple(result)


def _saved_evidence(table):
    """Use the same float representation as the original CSV evidence tables."""
    return pd.read_csv(StringIO(table.to_csv(index=False)))


def _proposal_verifier(verifier):
    def verify(stock, gallery, query, ids, proposed):
        g_side, q_side = (stock["flank_side"](frame.viewpoint) for frame in (gallery, query))
        classes = gallery.identity.to_numpy(dtype=str)
        groups = [np.flatnonzero(classes == label) for label in ids]
        features = stock["features"]
        cosine = features[query.feat_idx] @ features[gallery.feat_idx].T
        gallery_ids = gallery.id.to_numpy()
        rows = []
        for i, query_id in enumerate(query.id):
            for candidate in proposed[i]:
                pool = groups[candidate]
                pool = pool[g_side[pool] * q_side[i] != -1]
                if len(pool):
                    chosen = pool[cosine[i, pool].argmax()]
                    valid, matches, inliers, spread = verifier.score(query_id, gallery_ids[chosen])
                    rows.append((i, candidate, int(valid), np.log1p(inliers), inliers/max(matches, 1), spread, 0))
                else:
                    rows.append((i, candidate, 0, 0.0, 0.0, 0.0, 0))
        return _saved_evidence(pd.DataFrame(rows, columns=["query", "candidate", *LOCAL_COLUMNS]))

    return verify


def stock_pools(ids, krec, diff, bias, power, blend):
    for i in range(len(diff["te_ca"])):
        scores = {NOVEL_LABEL: 0.0}
        for branch, weight in ((diff, blend), (krec, 1-blend)):
            novelty = float(np.clip(branch["te_pn"][i] * bias, 0.001, 0.999))
            scores[NOVEL_LABEL] += weight * novelty
            candidate_scores = np.power(np.maximum(branch["te_sc"][i], 1e-6), power)
            candidate_scores /= max(candidate_scores.sum(), 1e-8)
            for candidate, value in zip(branch["te_ca"][i], candidate_scores):
                label = str(ids[candidate])
                scores[label] = scores.get(label, 0.0) + weight * (1-novelty) * 0.96 * value
        yield scores


def leopard_control(ids, stock, inner_evidence, query_evidence_rows):
    tables, features, outcomes = [], [], []
    for evidence, ranks, labels, inner_ids in inner_evidence:
        tables.append(evidence.assign(match=inner_ids[evidence.candidate] == labels[evidence["query"]]))
        covered = np.logical_or.reduce([(inner_ids[branch] == labels[:, None]).any(axis=1) for branch in ranks])
        outcomes.append(np.where(labels == NOVEL_LABEL, "novel", np.where(covered, "covered", "missed")))
        features.append(query_evidence(evidence))
    table = pd.concat(tables, ignore_index=True)
    background = LogisticRegression(C=0.35, max_iter=300).fit(table[GLOBAL_COLUMNS], table.match)
    full = LogisticRegression(C=0.35, max_iter=300).fit(table[GLOBAL_COLUMNS + LOCAL_COLUMNS], table.match)
    novelty = LogisticRegression(C=0.15, max_iter=300).fit(pd.concat(features), np.concatenate(outcomes))
    evidence = query_evidence_rows
    strong = evidence[(evidence.available == 1) & (evidence.log_inliers >= np.log1p(4))
                      & (evidence.inlier_fraction >= 0.35) & (evidence.spread >= 0.01)].copy()
    boosts = {}
    if len(strong):
        lift = full.decision_function(strong[GLOBAL_COLUMNS + LOCAL_COLUMNS])
        lift -= background.decision_function(strong[GLOBAL_COLUMNS])
        strong["boost"] = np.exp(np.clip(lift, 0, np.log(8)))
        boosts = dict(list(strong.groupby("query")))
    probabilities = dict(zip(novelty.classes_, novelty.predict_proba(query_evidence(evidence)).T))
    result = []
    for i, values in enumerate(stock):
        pool = values.copy()
        unseen = pool.pop(NOVEL_LABEL)
        initial_mass = sum(pool.values())
        if i in boosts:
            for row in boosts[i].itertuples(index=False):
                pool[str(ids[row.candidate])] *= row.boost
        scale = initial_mass / sum(pool.values())
        known = {label: float(value * scale) for label, value in pool.items()}
        total = sum(known.values())
        covered = 0.65 * total + 0.35 * probabilities["covered"][i]
        result.append({NOVEL_LABEL: 0.65 * unseen + 0.35 * probabilities["novel"][i],
                       **{label: value * covered / total for label, value in known.items()}})
    return result


def rank_tables(inner, test_table, controls):
    """Fit the candidate and novelty heads on the encounter-disjoint inner rows."""
    x, y, contexts, outcomes = [], [], [], []
    for (options, table, _, context), truth in inner:
        x.append(table)
        y.append(np.concatenate([row == label for row, label in zip(options, truth)]))
        contexts.append(context)
        outcomes.extend("novel" if label == NOVEL_LABEL else "covered" if label in row else "missed"
                        for row, label in zip(options, truth))
    classifier = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.07, max_leaf_nodes=9,
                                                 min_samples_leaf=45, l2_regularization=3,
                                                 early_stopping=False, random_state=1987)
    classifier.fit(np.concatenate(x), np.concatenate(y))
    novelty = make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=400))
    novelty.fit(np.concatenate(contexts), outcomes)
    options, values, lengths, queries = test_table
    matches = np.split(np.maximum(classifier.predict_proba(values)[:, 1], 1e-8), np.cumsum(lengths)[:-1])
    p = dict(zip(novelty.classes_, novelty.predict_proba(queries).T))
    predictions = []
    for control, labels, match, novel, covered in zip(controls, options, matches, p["novel"], p["covered"]):
        known = match / match.sum() * covered
        scores = {str(label): 0.65 * control.get(label, 0.0) + 0.35 * value for label, value in zip(labels, known)}
        scores[NOVEL_LABEL] = 0.65 * control[NOVEL_LABEL] + 0.35 * novel
        predictions.append(sorted(scores, key=scores.get, reverse=True)[:5])
    return predictions


def rank_final(features, flank_side, species, gallery, query, ids, krec, diff, bias, power, blend,
               stock_lists, folds, sift_directory):
    stock = {"features": features, "flank_side": flank_side}
    leopard = species == "leopard"
    verifier = SiftVerifier(sift_directory)
    geometry = _proposal_verifier(verifier) if leopard else None
    descriptors = load_all_rootsift(list(gallery.id) + list(query.id), sift_directory, max_desc=128, seed=1987)

    def table(g, q, labels, ranks):
        evidence = _saved_evidence(candidate_evidence(features, g, q, labels, ranks, verifier, flank_side))
        fv_max, fv_mean, fv_ids = compute_fv_identity_scores(g, q, descriptors, n_components=32, pca_dim=32, seed=1987)
        lnbnn, ln_ids = compute_lnbnn_identity_scores(g, q, descriptors, k_bg=10, max_desc_gal=64,
                                                       max_desc_que=64, pca_dim=32, seed=1987)
        if not np.array_equal(labels, fv_ids) or not np.array_equal(labels, ln_ids):
            raise ValueError("Local descriptor scores use a different gallery identity order")
        return (build_table_b2(stock, g, q, labels, ranks, evidence, geometry, fv_max, fv_mean, lnbnn,
                               add_fv=4, add_ln=4, verify_local=int(leopard)), evidence)

    train, evidence_train = [], []
    for g, q, truth, inner_ids, ranks in folds:
        data, evidence = table(g, q, inner_ids, ranks)
        train.append((data, truth))
        evidence_train.append((evidence, ranks, truth, inner_ids))
    destination, evidence = table(gallery, query, ids, (krec["te_ca"], diff["te_ca"]))
    controls = list(stock_pools(ids, krec, diff, bias, power, blend))
    if any(sorted(row, key=row.get, reverse=True)[:5] != list(expected)
           for row, expected in zip(controls, stock_lists)):
        raise ValueError("Reconstructed stock scores differ from Model 3")
    if leopard:
        controls = leopard_control(ids, controls, evidence_train, evidence)
    return rank_tables(train, destination, controls)
