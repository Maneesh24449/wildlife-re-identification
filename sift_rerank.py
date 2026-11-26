"""Leopard candidate rescoring with the supplied local descriptors."""

from functools import lru_cache
import hashlib

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression


NOVEL_LABEL = "new_individual"
VERIFY_RANKS = (0, 1, 4, 5, 6, 2)
# Keep the fitted feature schema used for the 0.84209 submission.
GLOBAL_COLUMNS = ["agreement", "reciprocal_sum", "raw_top", "local_rank", "local_margin"]
LOCAL_COLUMNS = ["available", "log_inliers", "inlier_fraction", "spread", "second_support"]


def find_sift_directory(data_dir):
    """Find the paired, extracted descriptor stores for a dataset layout."""
    for directory in (data_dir, data_dir.parent):
        train = directory / "sift_features_train.npz"
        test = directory / "sift_features_test.npz"
        if train.is_dir() and test.is_dir():
            return directory
    if any((directory / name).exists() for directory in (data_dir, data_dir.parent)
           for name in ("sift_features_train.npz", "sift_features_test.npz")):
        raise FileNotFoundError("Extract both supplied SIFT stores as directories beside the dataset.")
    return None


class SiftVerifier:
    def __init__(self, directory):
        self.directory = directory
        self.descriptor = lru_cache(maxsize=512)(self._descriptor)
        self.matches = {}

    def _descriptor(self, image_id):
        partition = "train" if image_id.startswith("tr_") else "test"
        store = self.directory / f"sift_features_{partition}.npz"
        xy = np.load(store / f"{image_id}_xy.npy", allow_pickle=False)
        vectors = np.load(store / f"{image_id}_desc.npy", allow_pickle=False).astype(np.float32)
        if xy.shape != (len(vectors), 2) or vectors.shape[1:] != (128,):
            raise ValueError(f"Malformed supplied SIFT arrays for {image_id}")
        keep = vectors.sum(axis=1) > 0
        vectors = vectors[keep]
        return xy[keep], np.sqrt(vectors / vectors.sum(axis=1, keepdims=True))

    def score(self, query_id, gallery_id):
        key = query_id, gallery_id
        if key not in self.matches:
            seed = int.from_bytes(hashlib.sha256(f"{query_id}|{gallery_id}".encode()).digest()[:8], "little")
            self.matches[key] = self._verify(query_id, gallery_id, np.random.default_rng(seed))
        return self.matches[key]

    def _verify(self, query_id, gallery_id, rng):
        xy_a, a = self.descriptor(query_id)
        xy_b, b = self.descriptor(gallery_id)
        if min(len(a), len(b)) < 2:
            return False, 0, 0, 0.0
        distances = cdist(a, b, metric="sqeuclidean")

        def nearest(matrix):
            best = matrix.argmin(axis=1)
            two = np.partition(matrix, 1, axis=1)[:, :2]
            return best, two[:, 0] < (0.8 ** 2) * two[:, 1]

        ab, a_pass = nearest(distances)
        ba, b_pass = nearest(distances.T)
        i = np.flatnonzero(a_pass & b_pass[ab] & (ba[ab] == np.arange(len(a))))
        if len(i) < 4:
            return True, len(i), 0, 0.0
        j = ab[i]
        # Fit orientation-preserving similarity transforms from two matches.
        z, w = (xy[:, 0] + 1j * xy[:, 1] for xy in (xy_a[i], xy_b[j]))
        left = rng.integers(len(i), size=128)
        right = (left + rng.integers(1, len(i), size=128)) % len(i)
        dz, dw = z[right] - z[left], w[right] - w[left]
        transform = dw / np.where(np.abs(dz) > 1e-6, dz, 1)
        scale = np.abs(transform)
        keep = (np.abs(dz) > 1e-6) & (scale >= 0.4) & (scale <= 2.5)
        if not keep.any():
            return True, len(i), 0, 0.0
        transform, scale = transform[keep], scale[keep]
        offset = w[left[keep]] - transform * z[left[keep]]
        error = np.abs(transform[:, None] * z + offset[:, None] - w)
        inliers = error <= 7.0 * np.minimum(1.0, scale[:, None])
        accepted = inliers[inliers.sum(axis=1).argmax()]
        count = int(accepted.sum())
        spread = 0.0
        if count >= 4:
            area = np.prod(np.maximum(np.ptp(xy_a, axis=0), 1.0))
            spread = float(np.prod(np.ptp(xy_a[i[accepted]], axis=0)) / area)
        return True, len(i), count, spread


def verify_identity(verifier, query_id, gallery_ids, encounters, rows):
    used, scores = set(), []
    for row in rows:
        encounter = encounters[row]
        if encounter in used:
            continue
        used.add(encounter)
        available, matches, inliers, spread = verifier.score(query_id, gallery_ids[row])
        if available:
            scores.append((inliers, inliers / max(1, matches), spread))
        if len(used) == 2:
            break
    if not scores:
        return 0, 0.0, 0.0, 0.0, 0
    inliers, fraction, spread = max(scores)
    second = sum(score[0] >= 4 for score in scores) - int(inliers >= 4)
    return 1, np.log1p(inliers), fraction, spread, second


def candidate_evidence(features, gallery, query, identities, branches, verifier, flank_side):
    gallery_ids = gallery.id.to_numpy()
    encounters = gallery.enc.to_numpy()
    labels = gallery.identity.to_numpy()
    g_side, q_side = (flank_side(frame.viewpoint) for frame in (gallery, query))
    groups = [np.flatnonzero(labels == identity) for identity in identities]
    cosine = features[query.feat_idx] @ features[gallery.feat_idx].T
    records = []
    for i, query_id in enumerate(query.id):
        vote, members = {}, {}
        for candidates in branches:
            for r, candidate in enumerate(candidates[i]):
                vote[candidate] = vote.get(candidate, 0.0) + 1 / (r + 2)
                members[candidate] = members.get(candidate, 0) + 1
        ranked = sorted(vote, key=vote.get, reverse=True)
        for rank in VERIFY_RANKS:
            if rank >= len(ranked):
                continue
            candidate = ranked[rank]
            rows = groups[candidate]
            raw_top = float(cosine[i, rows].max())
            compatible = rows[g_side[rows] * q_side[i] != -1]
            compatible = compatible[np.argsort(-cosine[i, compatible], kind="stable")]
            local = verify_identity(verifier, query_id, gallery_ids, encounters, compatible)
            records.append((i, candidate, members[candidate], vote[candidate], raw_top, -1, 0.0, *local))
    return pd.DataFrame(records, columns=["query", "candidate", *GLOBAL_COLUMNS, *LOCAL_COLUMNS])


def query_evidence(table):
    local = table.assign(valid_spread=table.spread * table.inlier_fraction)
    return local.groupby("query", sort=True).agg(
        highest_raw=("raw_top", "max"), strongest_agreement=("reciprocal_sum", "max"),
        strongest_local=("log_inliers", "max"), local_spread=("valid_spread", "max"),
        verified_candidates=("available", "sum"),
    )


def fit_sift_correction(features, folds, verifier, flank_side):
    tables, novelty_features, outcomes = [], [], []
    for gallery, query, labels, identities, candidates in folds:
        evidence = candidate_evidence(features, gallery, query, identities, candidates, verifier, flank_side)
        tables.append(evidence.assign(match=identities[evidence.candidate] == labels[evidence["query"]]))
        covered = np.logical_or.reduce([(identities[branch] == labels[:, None]).any(axis=1) for branch in candidates])
        outcomes.append(np.where(labels == NOVEL_LABEL, "novel", np.where(covered, "covered", "missed")))
        novelty_features.append(query_evidence(evidence))
    training = pd.concat(tables, ignore_index=True)
    reference = LogisticRegression(C=0.35, max_iter=300).fit(training[GLOBAL_COLUMNS], training.match)
    corrected = LogisticRegression(C=0.35, max_iter=300).fit(training[GLOBAL_COLUMNS + LOCAL_COLUMNS], training.match)
    coverage = LogisticRegression(C=0.15, max_iter=300).fit(pd.concat(novelty_features), np.concatenate(outcomes))
    return reference, corrected, coverage


def rerank_leopards(features, gallery, query, identities, krec, diff, bias, power, stock_lists,
                    folds, directory, flank_side):
    """Re-rank the full known candidate union and keep missed-known mass unassigned."""
    verifier = SiftVerifier(directory)
    reference, corrected, coverage = fit_sift_correction(features, folds, verifier, flank_side)
    evidence = candidate_evidence(features, gallery, query, identities,
                                  (krec["te_ca"], diff["te_ca"]), verifier, flank_side)
    verified = evidence[(evidence.available == 1) & (evidence.log_inliers >= np.log1p(4))
                        & (evidence.inlier_fraction >= 0.35) & (evidence.spread >= 0.01)].copy()
    boosts = {}
    if len(verified):
        lift = corrected.decision_function(verified[GLOBAL_COLUMNS + LOCAL_COLUMNS])
        lift -= reference.decision_function(verified[GLOBAL_COLUMNS])
        verified["boost"] = np.exp(np.clip(lift, 0, np.log(8)))
        boosts = dict(list(verified.groupby("query")))
    probabilities = dict(zip(coverage.classes_, coverage.predict_proba(query_evidence(evidence)).T))
    predictions = []
    for i, original in enumerate(stock_lists):
        scores = {NOVEL_LABEL: 0.0}
        for branch in (diff, krec):
            novelty = float(np.clip(branch["te_pn"][i] * bias, 0.001, 0.999))
            scores[NOVEL_LABEL] += 0.5 * novelty
            ranks = np.power(np.maximum(branch["te_sc"][i], 1e-6), power)
            ranks /= max(ranks.sum(), 1e-8)
            for candidate, value in zip(branch["te_ca"][i], ranks):
                label = str(identities[candidate])
                scores[label] = scores.get(label, 0.0) + 0.5 * (1 - novelty) * 0.96 * value
        if sorted(scores, key=scores.get, reverse=True)[:5] != original:
            raise ValueError(f"Leopard score reconstruction differs from Model 3 for {query.id.iloc[i]}")
        novelty = scores.pop(NOVEL_LABEL)
        original_total = sum(scores.values())
        if i in boosts:
            for row in boosts[i].itertuples(index=False):
                scores[str(identities[row.candidate])] *= row.boost
        normalization = original_total / sum(scores.values())
        known = {label: float(value * normalization) for label, value in scores.items()}
        total = sum(known.values())
        covered = 0.65 * total + 0.35 * probabilities["covered"][i]
        updated = {NOVEL_LABEL: 0.65 * novelty + 0.35 * probabilities["novel"][i],
                   **{label: value * covered / total for label, value in known.items()}}
        predictions.append(sorted(updated, key=updated.get, reverse=True)[:5])
    return predictions
