"""E15, first version: infer the ball inside ball gaps, then read possession off it.

Idea (denis, 2026-11-07): the proxy league hides the IDSSE ball the way BAR-ATH loses it. The
methods so far learn to cope without the ball there (speed, team shape). Here we learn to put it
back: a regressor trained on the hidden-ball IDSSE dev matches against their true ball, then
applied to the frames where the ball is missing, BAR-ATH included.

Two stages, both fitted on the IDSSE dev matches only (never the hold-out, D3), leave the
scored match out, and simulator v1 with framemodel.TRAIN_SEED:

1. **Imputer** (`fit_imputer`, `impute`). Per frame inside a gap, the ball's offset from the
   straight line between the last and the next seen ball of the same half (or from the players'
   centroid where one end is missing), from: those two ball positions, the time into and left
   in the gap, where the players are (centroid, spread, each team's centroid, the player with
   the most others within DENSE_M, the player nearest the straight-line point, the fastest
   player) and how they move (mean velocity). Two HistGradientBoostingRegressors, x and y.
   Players only beyond the two seen endpoints, never the hidden ball.
2. **Team reader** (`fit_reader`, `log_odds`). z = (distance from the imputed ball to the
   nearest away player) - (to the nearest home player), clipped to +-Z_MAX m: positive when a
   home player is closer. A binned table P(home | z, alive), made swap-symmetric (z -> -z with
   the other team) and add-one smoothed, gives per-frame log P(home) / P(away). The table is fitted
   on CROSS-FITTED imputations (each training match imputed by an imputer that did not see it),
   so its confidence is the imputer's out-of-sample confidence, not its training fit.

The decoder reads the result through semi_markov's `frame_fn` on gap frames, next to the C3
frame model on seen frames (methods.ball_frame_fn): same term, same weight, dead ball takes the
better team's likelihood, so this only chooses between the teams.

`mode="oracle"` replaces stage 1 with the TRUE ball plus Gaussian noise of `noise_m` per axis,
read `lag_s` late (IDSSE only): the Step 0 ceiling of how good the imputer has to be.
"""

from functools import lru_cache

import numpy as np

from solution.match import AWAY, HOME, pm

PARAMS = {
    "labels": "keep2",      # IDSSE label variant the reader is fitted on (as framemodel, D21)
    "step": 5,              # train on every STEP-th gap frame
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "l2": 1.0,
    "clip": 0.02,           # reader probabilities clipped to [clip, 1 - clip]
}
DENSE_M = 10.0              # radius for "the player with the most others near him"
Z_MAX = 30.0                # m; z beyond this is clipped
Z_EDGES = np.arange(-Z_MAX, Z_MAX + 0.01, 2.0)
VEL_LAG = 5                 # frames either side for player velocity (0.2 s)


# -- features ----------------------------------------------------------------------------------

def _gap_endpoints(seen, halves, n):
    """For every frame: index of the last seen frame before it and the next seen frame after
    it, within the same half (-1 where none)."""
    prev = np.full(n, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    for start, stop in halves:
        idx = np.arange(start, stop)
        s = seen[start:stop]
        last = np.where(s, idx, -1)
        prev[start:stop] = np.maximum.accumulate(last)
        first = np.where(s, idx, np.iinfo(np.int64).max)
        nx = np.minimum.accumulate(first[::-1])[::-1]
        nxt[start:stop] = np.where(nx == np.iinfo(np.int64).max, -1, nx)
    return prev, nxt


def _nanmean(a, axis=1):
    finite = np.isfinite(a)
    count = finite.sum(axis=axis)
    total = np.where(finite, a, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _nanstd(a, mean, axis=1):
    finite = np.isfinite(a)
    count = finite.sum(axis=axis)
    dev = np.where(finite, (a - mean[:, None]) ** 2, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 1, np.sqrt(dev / np.maximum(count, 1)), np.nan)


def _pick(x, y, index):
    """Positions of player `index` per frame (-1 -> NaN)."""
    rows = np.arange(len(index))
    safe = np.maximum(index, 0)
    px, py = x[rows, safe], y[rows, safe]
    return np.where(index >= 0, px, np.nan), np.where(index >= 0, py, np.nan)


def features(match, frames):
    """(len(frames), F) feature rows and (len(frames), 2) base point, for `frames` (indices)."""
    n = match.n_frames
    seen = match.ball_seen
    prev, nxt = _gap_endpoints(seen, match.halves, n)
    p, q = prev[frames], nxt[frames]
    bx, by = match.ball_x.astype(float), match.ball_y.astype(float)
    px = np.where(p >= 0, bx[np.maximum(p, 0)], np.nan)
    py = np.where(p >= 0, by[np.maximum(p, 0)], np.nan)
    qx = np.where(q >= 0, bx[np.maximum(q, 0)], np.nan)
    qy = np.where(q >= 0, by[np.maximum(q, 0)], np.nan)
    fps = float(match.fps)
    since = np.where(p >= 0, (frames - p) / fps, np.nan)
    until = np.where(q >= 0, (q - frames) / fps, np.nan)
    length = since + until
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = since / length
    lx = np.where(np.isfinite(frac), px + frac * (qx - px), np.where(np.isfinite(px), px, qx))
    ly = np.where(np.isfinite(frac), py + frac * (qy - py), np.where(np.isfinite(py), py, qy))

    X = match.player_x[frames].astype(float)
    Y = match.player_y[frames].astype(float)
    cx, cy = _nanmean(X), _nanmean(Y)
    sx, sy = _nanstd(X, cx), _nanstd(Y, cy)
    home = match.team == HOME
    hx, hy = _nanmean(X[:, home]), _nanmean(Y[:, home])
    ax, ay = _nanmean(X[:, ~home]), _nanmean(Y[:, ~home])

    lo = np.clip(frames - VEL_LAG, 0, n - 1)
    hi = np.clip(frames + VEL_LAG, 0, n - 1)
    span = np.maximum(hi - lo, 1)[:, None] / fps
    vxp = (match.player_x[hi].astype(float) - match.player_x[lo]) / span
    vyp = (match.player_y[hi].astype(float) - match.player_y[lo]) / span
    vx, vy = _nanmean(vxp), _nanmean(vyp)
    speed = np.hypot(vxp, vyp)
    fastest = np.where(np.isfinite(speed).any(axis=1),
                       np.argmax(np.where(np.isfinite(speed), speed, -1.0), axis=1), -1)
    fx, fy = _pick(X, Y, fastest)

    # densest player: most others within DENSE_M, in chunks to bound memory
    dense = np.full(len(frames), -1, dtype=np.int64)
    chunk = 4000
    for a in range(0, len(frames), chunk):
        xs, ys = X[a:a + chunk], Y[a:a + chunk]
        d2 = (xs[:, :, None] - xs[:, None, :]) ** 2 + (ys[:, :, None] - ys[:, None, :]) ** 2
        near = (d2 < DENSE_M ** 2).sum(axis=2).astype(float)
        near[~np.isfinite(xs)] = -1.0
        ok = np.isfinite(xs).any(axis=1)
        dense[a:a + chunk] = np.where(ok, np.argmax(near, axis=1), -1)
    dx, dy = _pick(X, Y, dense)

    d2l = (X - lx[:, None]) ** 2 + (Y - ly[:, None]) ** 2
    finite = np.isfinite(d2l)
    nearest = np.where(finite.any(axis=1), np.argmin(np.where(finite, d2l, np.inf), axis=1), -1)
    nx_, ny_ = _pick(X, Y, nearest)
    nd = np.hypot(nx_ - lx, ny_ - ly)

    base_x = np.where(np.isfinite(lx), lx, cx)
    base_y = np.where(np.isfinite(ly), ly, cy)
    rows = np.column_stack([
        px, py, qx, qy, since, until, length, frac, lx, ly,
        cx, cy, sx, sy, hx, hy, ax, ay, vx, vy, fx, fy, dx, dy, nx_, ny_, nd,
        dx - base_x, dy - base_y, nx_ - base_x, ny_ - base_y, cx - base_x, cy - base_y,
    ])
    return rows, np.column_stack([base_x, base_y])


# -- stage 1: the imputer ----------------------------------------------------------------------

def _gap_frames(match):
    return np.flatnonzero(~match.ball_seen & match.in_play_mask())


def fit_imputer(pairs, params=PARAMS):
    """(regressor x, regressor y) from (degraded match, clean match) pairs: gap frames whose
    true ball is known, every STEP-th."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows, targets = [], []
    for degraded, clean in pairs:
        frames = _gap_frames(degraded)
        frames = frames[np.isfinite(clean.ball_x[frames]) & np.isfinite(clean.ball_y[frames])]
        frames = frames[::params["step"]]
        X, base = features(degraded, frames)
        ok = np.isfinite(base).all(axis=1)
        rows.append(X[ok])
        targets.append(np.column_stack([clean.ball_x[frames], clean.ball_y[frames]])[ok] - base[ok])
    X, T = np.vstack(rows), np.vstack(targets)
    models = []
    for k in range(2):
        model = HistGradientBoostingRegressor(
            max_iter=params["max_iter"], learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2"], early_stopping=False, random_state=0)
        model.fit(X, T[:, k])
        models.append(model)
    return tuple(models)


def impute(models, match):
    """(T,) ball x, y: the seen ball where seen, the imputed one in gaps inside the halves."""
    bx, by = match.ball_x.astype(float).copy(), match.ball_y.astype(float).copy()
    frames = _gap_frames(match)
    if len(frames):
        X, base = features(match, frames)
        ok = np.isfinite(base).all(axis=1)
        bx[frames[ok]] = base[ok, 0] + models[0].predict(X[ok])
        by[frames[ok]] = base[ok, 1] + models[1].predict(X[ok])
    return bx, by


def oracle_ball(match, clean, noise_m, lag_s, seed=0):
    """The true ball in the gaps, `lag_s` late, plus N(0, noise_m) per axis (Step 0)."""
    bx, by = match.ball_x.astype(float).copy(), match.ball_y.astype(float).copy()
    frames = _gap_frames(match)
    src = np.clip(frames - int(round(lag_s * match.fps)), 0, match.n_frames - 1)
    rng = np.random.RandomState(seed)
    bx[frames] = clean.ball_x[src] + rng.normal(0.0, noise_m, len(frames))
    by[frames] = clean.ball_y[src] + rng.normal(0.0, noise_m, len(frames))
    return bx, by


# -- stage 2: the team reader ------------------------------------------------------------------

def closeness(match, bx, by):
    """(T,) z = nearest away distance - nearest home distance to (bx, by), clipped; NaN where
    either team has nobody tracked or there is no ball."""
    def nearest(side):
        cols = match.team == side
        d = np.hypot(match.player_x[:, cols] - bx[:, None], match.player_y[:, cols] - by[:, None])
        d = np.where(np.isfinite(d), d, np.inf)
        out = d.min(axis=1) if d.shape[1] else np.full(len(bx), np.inf)
        return np.where(np.isfinite(out), out, np.nan)
    with np.errstate(invalid="ignore"):
        return np.clip(nearest(AWAY) - nearest(HOME), -Z_MAX, Z_MAX)


def _bins(z):
    return np.clip(np.searchsorted(Z_EDGES, z, side="right") - 1, 0, len(Z_EDGES) - 2)


def fit_reader(samples, params=PARAMS):
    """(n_bins,) log P(home) - log P(away) per z bin from (z, labels) of alive gap frames,
    made symmetric: a frame (z, H) also counts as (-z, A)."""
    nb = len(Z_EDGES) - 1
    counts = np.ones((2, nb))                      # add-one smoothing; rows: home, away
    for z, labels in samples:
        ok = np.isfinite(z) & np.isin(labels, (pm.H, pm.A))
        b, mirror = _bins(z[ok]), _bins(-z[ok])
        home = labels[ok] == pm.H
        np.add.at(counts[0], b[home], 1)
        np.add.at(counts[1], b[~home], 1)
        np.add.at(counts[1], mirror[home], 1)      # the swapped copy
        np.add.at(counts[0], mirror[~home], 1)
    p = np.clip(counts[0] / counts.sum(axis=0), params["clip"], 1.0 - params["clip"])
    return np.log(p) - np.log1p(-p)


def read(table, z, frames_mask):
    """Per-frame log-odds from the reader table where frames_mask and z is known, else NaN."""
    out = np.full(len(z), np.nan)
    use = frames_mask & np.isfinite(z)
    out[use] = table[_bins(z[use])]
    return out


# -- fitted on the IDSSE dev matches -----------------------------------------------------------

@lru_cache(maxsize=None)
def _pair(match_id, labels):
    from solution import framemodel
    from solution.sources import idsse
    return framemodel._training_match(match_id, labels), idsse.load(match_id, labels=labels)


@lru_cache(maxsize=None)
def _imputer(ids, items):
    params = dict(items)
    return fit_imputer([_pair(m, params["labels"]) for m in ids], params)


def _dev_ids(exclude):
    from solution.sources import idsse
    ids = tuple(m for m in idsse.DEV_IDS if "idsse_" + m != exclude)
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    return ids


def _ball_for(mode, ids, items, match, clean, noise_m, lag_s):
    if mode == "oracle":
        return oracle_ball(match, clean, noise_m, lag_s)
    return impute(_imputer(ids, items), match)


@lru_cache(maxsize=None)
def _reader(exclude, items, mode="model", noise_m=0.0, lag_s=0.0):
    """Reader table fitted on the dev matches minus `exclude`, each imputed by an imputer
    trained without it (cross-fitting) or, for the oracle, by the same noisy truth."""
    params = dict(items)
    ids = _dev_ids(exclude)
    samples = []
    for m in ids:
        degraded, clean = _pair(m, params["labels"])
        others = tuple(k for k in ids if k != m)
        bx, by = _ball_for(mode, others, items, degraded, clean, noise_m, lag_s)
        z = closeness(degraded, bx, by)
        gaps = ~degraded.ball_seen & degraded.in_play_mask()
        samples.append((np.where(gaps, z, np.nan), clean.labels))
    return fit_reader(samples, params)


def log_odds(match, params=PARAMS, mode="model", noise_m=0.0, lag_s=0.0):
    """Per-frame log P(home) / P(away) from the inferred ball on gap frames; NaN elsewhere."""
    items = tuple(sorted(params.items()))
    clean = None
    if mode == "oracle":
        from solution.sources import idsse
        if not match.name.startswith("idsse_"):
            raise ValueError("{}: the oracle ball exists on IDSSE matches only".format(match.name))
        refused = {"tempo", "speed_factor", "prekickoff"} & set(match.notes)
        if refused:
            raise ValueError("oracle ball does not support {}".format(sorted(refused)))
        clean = idsse.load(match.name[len("idsse_"):])
    ids = _dev_ids(match.name)
    bx, by = _ball_for(mode, ids, items, match, clean, noise_m, lag_s)
    z = closeness(match, bx, by)
    gaps = ~match.ball_seen & match.in_play_mask()
    return read(_reader(match.name, items, mode, noise_m, lag_s), z, gaps)


def ball_error(match, clean, params=PARAMS):
    """Diagnostics on a degraded IDSSE match: per gap frame distance (m) between the imputed
    and the true ball, and the same for the straight line between the gap's ends."""
    items = tuple(sorted(params.items()))
    bx, by = impute(_imputer(_dev_ids(match.name), items), match)
    frames = _gap_frames(match)
    frames = frames[np.isfinite(clean.ball_x[frames])]
    _, base = features(match, frames)
    err = np.hypot(bx[frames] - clean.ball_x[frames], by[frames] - clean.ball_y[frames])
    lin = np.hypot(base[:, 0] - clean.ball_x[frames], base[:, 1] - clean.ball_y[frames])
    return frames, err, lin
