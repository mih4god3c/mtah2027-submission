"""C6: adapt the C3 frame model to the match it runs on (PLAN.md C6, backlog E9).

The frame model (solution/framemodel.py) is trained on other matches and is team-swap
symmetric, so it cannot learn how *these* two teams look in and out of possession. C6 fits a
second, match-specific model on frames of the match itself whose team is taken as known, and
reads its output instead of (or next to) the base model's:

    adapted log-odds = g([base log-odds, features]), g fitted on the match's picked frames.

g is not symmetrised: telling Barcelona's shape from Athletic's is the point. Two learners
(`KINDS`): "logit", an L2 logistic regression on standardised features, and "gbm", a small
HistGradientBoostingClassifier, both on [base log-odds, features]; and "pooled", the frame
model itself refitted on its training matches (framemodel, swap-augmented) plus the match's
picked frames (not swapped), the two weighted equally in total, read without symmetrising.
All are deterministic (no randomness is used; random_state is fixed anyway).

Which frames are picked, and with which label, is the caller's choice:
  - Step 0, the ceiling (tools/selftrain_check.py --ceiling): the TRUE label on frames a
    pseudo-labeller could plausibly pick (`pickable`: ball seen, in play, the touch engine's
    holder unchanged for at least `min_hold_s`). Proxy only; it reads match.labels.
  - Step 1: an algorithmic pseudo-label from the match's own frames. Smart Tagging is never
    used to pick (it is the only independent check on BAR-ATH).

Cross-fitting. Frames are cut into `block_s` blocks, block b in fold b mod `folds`; each fold
is predicted by a g fitted on the picked frames of the other folds. A frame inside a ball gap
is then never predicted from labels of the same block, so g cannot just copy the possession
around the gap: it has to have learnt something about the teams.
"""

import zlib
from functools import lru_cache

import numpy as np

from solution import framemodel
from solution.match import pm

PARAMS = {
    "kind": "gbm",          # learner g: "gbm" or "logit"
    "min_hold_s": 1.0,      # pickable: holder unchanged for at least this long
    "block_s": 60.0,        # cross-fitting block length
    "folds": 5,
    "step": 5,              # train on every STEP-th picked frame
    "min_train": 200,       # a fold with fewer picked training frames keeps the base model
    # gbm
    "max_iter": 100,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2": 1.0,
    # logit
    "C": 1.0,
}
KINDS = ("gbm", "logit", "pooled")


def holder_runs(holder, fps, min_s):
    """(T,) bool: frames inside a run of one holding team (>= 0) lasting at least min_s."""
    out = np.zeros(len(holder), dtype=bool)
    edges = np.flatnonzero(np.diff(holder)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(holder)]))
    minimum = int(round(min_s * fps))
    for a, b in zip(starts, stops):
        if holder[a] >= 0 and b - a >= minimum:
            out[a:b] = True
    return out


def pickable(match, holder, min_hold_s=PARAMS["min_hold_s"]):
    """(T,) bool: ball seen, in play, and the touch engine's holder unchanged >= min_hold_s."""
    return (match.ball_seen & match.in_play_mask() &
            holder_runs(holder, match.fps, min_hold_s))


def fold_ids(n, fps, block_s=PARAMS["block_s"], folds=PARAMS["folds"]):
    return (np.arange(n) // int(round(block_s * fps))) % folds


def _design(base, X):
    return np.column_stack((base, X)).astype(np.float64)


@lru_cache(maxsize=2)
def _pooled_rows(exclude):
    """(X, y) of the frame model's training set without `exclude`, swapped copies included."""
    from solution.sources import idsse
    p = framemodel.PARAMS
    rows, targets, kinds = [], [], None
    for m in idsse.DEV_IDS:
        if "idsse_" + m == exclude:
            continue
        match = framemodel._training_match(m, p["labels"])
        X, _, kinds = framemodel.features(match)
        alive = np.flatnonzero(np.isin(match.labels, pm.POSSESSION_LABELS) &
                               match.in_play_mask())[::p["step"]]
        rows.append(X[alive])
        targets.append((match.labels[alive] == pm.H).astype(np.int8))
    X, y = np.vstack(rows), np.concatenate(targets)
    return np.vstack((X, framemodel.swap(X, kinds))), np.concatenate((y, 1 - y))


def _fit_predict_pooled(exclude, X, y, X_out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = framemodel.PARAMS
    X0, y0 = _pooled_rows(exclude)
    weight = np.concatenate((np.ones(len(y0)), np.full(len(y), len(y0) / float(len(y)))))
    model = HistGradientBoostingClassifier(
        max_iter=p["max_iter"], learning_rate=p["learning_rate"],
        max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p["min_samples_leaf"],
        l2_regularization=p["l2"], early_stopping=False, random_state=0)
    model.fit(np.vstack((X0, X)), np.concatenate((y0, y)), sample_weight=weight)
    return model.predict_proba(X_out)[:, 1]


def _fit_predict(Z, y, Z_out, params):
    if params["kind"] == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=params["max_iter"], learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"], l2_regularization=params["l2"],
            early_stopping=False, random_state=0)
        model.fit(Z, y)
        return model.predict_proba(Z_out)[:, 1]
    if params["kind"] == "logit":
        from sklearn.linear_model import LogisticRegression
        mean = np.nanmean(Z, axis=0)
        sd = np.nanstd(Z, axis=0)
        sd[~(sd > 0)] = 1.0
        mean[~np.isfinite(mean)] = 0.0

        def scale(A):
            out = (A - mean) / sd
            return np.where(np.isfinite(out), out, 0.0)
        model = LogisticRegression(C=params["C"], max_iter=1000)
        model.fit(scale(Z), y)
        return model.predict_proba(scale(Z_out))[:, 1]
    raise ValueError("unknown kind {!r}".format(params["kind"]))


def adapt(match, base, X, picked, target, params=PARAMS):
    """Cross-fitted adapted log-odds, (T,), NaN outside the halves.

    base: (T,) the frame model's log-odds; X: (T, F) framemodel.features; picked: (T,) bool
    training frames; target: (T,) 1 where the picked frame's team is home, 0 away (read only
    on picked frames). A fold with too few training frames, or only one team among them,
    keeps the base log-odds.
    """
    clip = framemodel.PARAMS["clip"]
    n = match.n_frames
    out = base.astype(np.float64).copy()
    Z = _design(base, X)
    inside = match.in_play_mask() & np.isfinite(base)
    fold = fold_ids(n, match.fps, params["block_s"], params["folds"])
    for k in range(params["folds"]):
        train = np.flatnonzero(picked & inside & (fold != k))[::params["step"]]
        y = target[train].astype(np.int8)
        rows = np.flatnonzero(inside & (fold == k))
        if len(train) < params["min_train"] or len(np.unique(y)) < 2 or not len(rows):
            continue
        if params["kind"] == "pooled":
            exclude = match.name if match.name.startswith("idsse_") else ""
            p = _fit_predict_pooled(exclude, X[train], y, X[rows])
        else:
            p = _fit_predict(Z[train], y, Z[rows], params)
        p = np.clip(p, clip, 1.0 - clip)
        out[rows] = np.log(p) - np.log1p(-p)
    out[~match.in_play_mask()] = np.nan
    return out


_MEMO = {}


def _key(match, tag):
    digest = zlib.crc32(np.ascontiguousarray(match.ball_x).tobytes())
    digest = zlib.crc32(np.ascontiguousarray(match.player_x).tobytes(), digest)
    return (match.name, match.n_frames, digest, tag)


def base_inputs(match):
    """(base log-odds, features) for a match, memoised on its contents (the proxy league asks
    for the same degraded match once per method)."""
    key = _key(match, "base")
    if key not in _MEMO:
        if len(_MEMO) > 64:
            _MEMO.clear()
        X, _, _ = framemodel.features(match)
        _MEMO[key] = (framemodel.log_odds(match), X)
    return _MEMO[key]


def oracle_log_odds(match, select="hold", params=PARAMS):
    """Step 0 (proxy only): adapted log-odds trained on the TRUE labels of this match.

    select "hold": `pickable` frames (ball seen, holder stable); "seen": every frame with the
    ball seen. Only frames whose true label is alive train g.
    """
    from solution import touch
    if match.labels is None:
        raise ValueError("the C6 oracle needs labels; it is a proxy-league ceiling only")
    key = _key(match, ("oracle", select, zlib.crc32(match.labels.tobytes()),
                       tuple(sorted(params.items()))))
    if key in _MEMO:
        return _MEMO[key]
    base, X = base_inputs(match)
    if select == "hold":
        picked = pickable(match, touch.detect(match).holder, params["min_hold_s"])
    elif select == "seen":
        picked = match.ball_seen & match.in_play_mask()
    else:
        raise ValueError("unknown select {!r}".format(select))
    labels = match.labels
    picked = picked & np.isin(labels, pm.POSSESSION_LABELS)
    out = adapt(match, base, X, picked, (labels == pm.H).astype(np.int8), params)
    _MEMO[key] = out
    return out


def where_seen(match, inside, outside):
    """`inside` on frames with the ball seen, `outside` elsewhere."""
    return np.where(match.ball_seen, inside, outside)
