"""#3 control model: is a touch of the touch engine a real change of control? (EXPERIMENTS section 21)

The touch engine (solution/touch.py) hands the ball to the other team at its first touch
(`touch_takes`, D10). Under controlled possession (keep2, D21) about 14% of its touches are
contradicted: the truth over the next second says the other team has the ball. Those touches
start most of the decoder's invented possessions where the ball is seen (D25). Removing every
contradicted touch from the holder sweep is worth +0.062 on the proxy, and a model that removes
70% of them while wrongly removing 10% of the agreed ones about +0.04 (section 21, Step 0).

This module scores each touch with P(contradicted) and drops the touches above a threshold
from the holder (touch.sweep_holder's keep mask). Features are relative to the touching team
("own") and the other ("opp"), so no team swap is needed, and read no absolute position (the
transfer risk of D22 / D26). Control episodes are left alone.

Training is leave-one-match-out on the IDSSE dev matches (never the hold-out, D3), degraded by
simulator v1 with framemodel.TRAIN_SEED plus D12's ball noise and jumps at both smoothings in
TRAIN_SMOOTH_S, on keep2 labels. The threshold is the PARAMS["false_drop"] upper quantile of the
training matches' own out-of-fold scores on agreed touches (inner leave-one-match-out), so
nothing is set on the evaluated match. On BAR-ATH all five dev matches train it.
HistGradientBoosting with a fixed random_state and no early stopping: deterministic.
"""

import warnings
from functools import lru_cache

import numpy as np

from solution import touch
from solution.framemodel import TRAIN_SEED
from solution.match import pm

PARAMS = {
    "labels": "keep2",
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2": 1.0,
    "false_drop": 0.10,      # share of agreed training touches the threshold drops
}
WINDOW_S = 1.0               # the truth over [f, f + WINDOW_S) gives a touch its verdict
TRAIN_SMOOTH_S = (0.26, 0.40)  # ball smoothings the training matches are degraded with
BALL_NOISE, BALL_JUMPS = 0.05, 3.4   # D12
NEAR_M = 3.0                 # players "near" the ball at the touch
CAP_S = 5.0                  # timing features are capped at this
SHARE_WINDOWS_S = ((-1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 4.0))
AGREED, CONTRADICTED, NEUTRAL = 1, -1, 0


def verdicts(truth, frames, teams, fps):
    """AGREED / CONTRADICTED / NEUTRAL per event, from the truth over [f, f + WINDOW_S)."""
    width = int(round(WINDOW_S * fps))
    out = np.full(len(frames), NEUTRAL, dtype=np.int8)
    for i, (f, k) in enumerate(zip(frames, teams)):
        if k not in (pm.H, pm.A):
            continue
        window = truth[f:f + width]
        if 2 * np.sum(window == k) > len(window):
            out[i] = AGREED
        elif 2 * np.sum(window == 1 - k) > len(window):
            out[i] = CONTRADICTED
    return out


def _window_sum(cumsum, lo, hi):
    """sum over [lo, hi) from a cumsum with a leading 0; lo, hi clipped arrays."""
    return cumsum[hi] - cumsum[lo]


def features(match, touches, params=touch.PARAMS):
    """(X (n_touches, F) float, names): the features of each touch, as the module describes."""
    fps, n = match.fps, match.n_frames
    usable, piece_start, piece_stop = touch.clean_ball(match, params)
    bx, by = touch.smoothed_ball(match, usable, piece_start, piece_stop, params)
    lag = touch._frames(params["lag_s"], fps)
    scale = fps / float(lag)
    vin = np.hypot(bx - touch._shift(bx, -lag), by - touch._shift(by, -lag)) * scale
    vout = np.hypot(touch._shift(bx, lag) - bx, touch._shift(by, lag) - by) * scale
    dvx = (touch._shift(bx, lag) - 2 * bx + touch._shift(bx, -lag)) * scale
    dvy = (touch._shift(by, lag) - 2 * by + touch._shift(by, -lag)) * scale
    dv = np.hypot(dvx, dvy)

    px, py = match.player_x.astype(np.float64), match.player_y.astype(np.float64)
    team_dist = np.full((n, 2), np.nan)
    for k in (0, 1):
        cols = match.team == k
        d = np.hypot(px[:, cols] - bx[:, None], py[:, cols] - by[:, None])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN rows: no ball
            team_dist[:, k] = np.nanmin(d, axis=1)
    ok = usable & np.isfinite(team_dist).all(axis=1)
    # Team 0 nearest counts 1, a tie 1/2: so swapping the teams swaps the shares exactly.
    with np.errstate(invalid="ignore"):
        near0 = np.where(ok, (team_dist[:, 0] < team_dist[:, 1]) +
                         0.5 * (team_dist[:, 0] == team_dist[:, 1]), 0.0)
    cum_ok = np.concatenate(([0.0], np.cumsum(ok)))
    cum_near0 = np.concatenate(([0.0], np.cumsum(near0)))
    diff10 = np.where(ok, team_dist[:, 1] - team_dist[:, 0], 0.0)    # opp - own for team 0
    cum_diff = np.concatenate(([0.0], np.cumsum(diff10)))
    cum_usable = np.concatenate(([0.0], np.cumsum(usable)))

    frames, teams = touches.touch_frame, touches.touch_team.astype(int)
    by_team = {k: frames[teams == k] for k in (0, 1)}
    control_start = {k: touches.control[touches.control[:, 2] == k, 0] for k in (0, 1)}

    def since(sorted_frames, f):
        i = np.searchsorted(sorted_frames, f) - 1
        return min((f - sorted_frames[i]) / fps, CAP_S) if i >= 0 else CAP_S

    def until(sorted_frames, f):
        i = np.searchsorted(sorted_frames, f, side="right")
        return min((sorted_frames[i] - f) / fps, CAP_S) if i < len(sorted_frames) else CAP_S

    names = ["dv", "speed_in", "speed_out", "own_m", "opp_m", "own_near", "opp_near",
             "held_before", "since_own", "since_opp", "until_own", "until_opp",
             "until_own_control", "until_opp_control"]
    for lo, hi in SHARE_WINDOWS_S:
        names += ["own_share_{:g}_{:g}".format(lo, hi), "dist_gap_{:g}_{:g}".format(lo, hi)]
    names += ["travel_1s", "usable_2s"]
    X = np.full((len(frames), len(names)), np.nan)
    one = int(round(1.0 * fps))
    for i, (f, k) in enumerate(zip(frames, teams)):
        if k not in (0, 1):
            continue
        o = 1 - k
        cols_k, cols_o = match.team == k, match.team == o
        d_own = np.hypot(px[f, cols_k] - bx[f], py[f, cols_k] - by[f])
        d_opp = np.hypot(px[f, cols_o] - bx[f], py[f, cols_o] - by[f])
        held = touches.holder[f - 1] if f > 0 else touch.NONE
        row = [dv[f], vin[f], vout[f], team_dist[f, k], team_dist[f, o],
               np.sum(d_own <= NEAR_M), np.sum(d_opp <= NEAR_M),
               1.0 if held == k else -1.0 if held == o else 0.0,
               since(by_team[k], f), since(by_team[o], f),
               until(by_team[k], f), until(by_team[o], f),
               until(control_start[k], f - 1), until(control_start[o], f - 1)]
        for lo_s, hi_s in SHARE_WINDOWS_S:
            lo = int(np.clip(f + round(lo_s * fps), 0, n))
            hi = int(np.clip(f + round(hi_s * fps), 0, n))
            count = _window_sum(cum_ok, lo, hi)
            if count > 0:
                near0 = _window_sum(cum_near0, lo, hi) / count
                gap10 = _window_sum(cum_diff, lo, hi) / count
                row += [near0 if k == 0 else 1.0 - near0, gap10 if k == 0 else -gap10]
            else:
                row += [np.nan, np.nan]
        g = min(f + one, n - 1)
        travel = np.hypot(bx[g] - bx[f], by[g] - by[f])
        hi = min(f + 2 * one, n)
        row += [travel, _window_sum(cum_usable, f, hi) / max(hi - f, 1)]
        X[i] = row
    return X, names


def fit(rows, params=PARAMS):
    """A classifier of P(contradicted) from [(X, verdicts)]; neutral touches are left out."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.vstack([x[v != NEUTRAL] for x, v in rows])
    y = np.concatenate([(v[v != NEUTRAL] == CONTRADICTED).astype(np.int8) for _, v in rows])
    model = HistGradientBoostingClassifier(
        max_iter=params["max_iter"], learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2"], early_stopping=False, random_state=0)
    model.fit(X, y)
    return model


@lru_cache(maxsize=None)
def _training_rows(match_id, smooth_s, labels):
    """(X, verdicts) of one degraded dev match's touches, at one ball smoothing."""
    from solution.simulate import add_ball_noise, degrade, train_version
    from solution.sources import idsse
    raw = idsse.load(match_id, labels=labels)
    match = add_ball_noise(degrade(raw, TRAIN_SEED, train_version()), BALL_NOISE, BALL_JUMPS, seed=TRAIN_SEED,
                           smooth_s=smooth_s)
    touches = touch.detect(match)
    X, _ = features(match, touches)
    return X, verdicts(raw.labels, touches.touch_frame, touches.touch_team, match.fps)


def _rows(ids, labels):
    return [_training_rows(m, s, labels) for m in ids for s in TRAIN_SMOOTH_S]


@lru_cache(maxsize=None)
def idsse_model(exclude="", items=()):
    """(model, threshold) trained on the dev matches minus `exclude` (a Match name). The
    threshold is the 1 - params["false_drop"] quantile of the agreed touches' scores, each
    scored by a model trained without its own match (inner leave-one-match-out)."""
    from solution.sources import idsse
    params = dict(items) if items else dict(PARAMS)
    ids = [m for m in idsse.DEV_IDS if "idsse_" + m != exclude]
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    agreed = []
    for held in ids:
        inner = fit(_rows([m for m in ids if m != held], params["labels"]), params)
        for X, v in _rows([held], params["labels"]):
            agreed.append(inner.predict_proba(X[v == AGREED])[:, 1])
    threshold = float(np.quantile(np.concatenate(agreed), 1.0 - params["false_drop"]))
    return fit(_rows(ids, params["labels"]), params), threshold


def scores(match, touches, params=PARAMS):
    """P(contradicted) per touch, from the model trained without this match."""
    model, _ = idsse_model(match.name, tuple(sorted(params.items())))
    X, _ = features(match, touches)
    return model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)


def keep_mask(match, touches, params=PARAMS):
    """True for the touches the holder keeps: P(contradicted) at most the threshold."""
    _, threshold = idsse_model(match.name, tuple(sorted(params.items())))
    return scores(match, touches, params) <= threshold
