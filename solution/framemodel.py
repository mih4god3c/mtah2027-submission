"""C3, first version: a per-frame team model from the players alone (PLAN.md C3).

Aimed at who has the ball where the ball is not tracked (about 40% of graded frames). C3 Step 0
(`tools/gap_check.py --ceiling`, EXPERIMENTS section 13) put the decoder's capacity with ideal
per-frame team evidence at spread's typical strength at +0.053 on controlled-possession truth
(keep2, D21), but at +0.035 when that evidence trails the truth by 1 s and +0.004 at 3 s. So
the model has to be timely, not just right on average.

The model gives P(home | alive) per frame. It does not judge dead ball: the decoder reads it
with dead ball taking the better team's likelihood (decoder.block_emissions, `frame_w`), as
D16 keeps for spread, so speed and C2b keep that question. Features read player positions only
(never the ball, D15), in home's attacking frame (gapteam.attacking_x), and come in three
kinds, so that swapping the teams is an exact operation on the feature vector (`swap`):
  - antisymmetric: home-minus-away or signed-towards-home values, negated by a swap;
  - pairs: one value per team, measured in that team's own frame, exchanged by a swap;
  - symmetric: unchanged.
Training adds every frame's swapped copy with the other label, and prediction averages
p(x) with 1 - p(swap(x)), so the model cannot favour a side or a colour.

Training is leave-one-match-out on the IDSSE dev matches (never the hold-out, D3), degraded by
simulator v1 with its own seed TRAIN_SEED (players are what the model reads, and the simulator
drops and ghosts them), on alive frames of the label variant asked for (keep2 by default,
D21), every STEP-th frame. On BAR-ATH all five dev matches train it.

The learner is scikit-learn's HistGradientBoostingClassifier (pinned in requirements-lock.txt):
the LightGBM-style histogram GBM PLAN.md C3 names, with NaN handled natively, no early stopping
and a fixed random_state, so the same inputs give the same output.
"""

import warnings
from functools import lru_cache

import numpy as np

from solution import gapteam, handover
from solution.match import AWAY, HOME, pm

PARAMS = {
    "labels": "keep2",      # IDSSE label variant the model is trained on (solution/relabel.py)
    "step": 5,              # train on every STEP-th alive frame (5 Hz, the decoder's block)
    "clip": 0.02,           # probabilities are clipped to [clip, 1 - clip] before the log
    "max_iter": 200,        # boosting rounds
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 200,
    "l2": 1.0,
}
TRAIN_SEED = 101            # simulator seed for the training matches; eval seeds are 0, 1, 2
CONTEXT_S = (3.0, 8.0)      # centred means of spread
SIDE_S = 3.0                # past-only and future-only means of spread and advance


def _window_mean(values, lo, hi, cut=True):
    """mean(values[t + lo:t + hi]), NaN skipped; NaN where the window is empty, or (with
    `cut`) where it runs past either end."""
    finite = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    count = np.concatenate(([0.0], np.cumsum(finite.astype(float))))
    n = len(values)
    t = np.arange(n)
    a, b = t + lo, t + hi
    ok = (a >= 0) & (b <= n)
    a, b = np.clip(a, 0, n), np.clip(b, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (total[b] - total[a]) / (count[b] - count[a])
    if cut:
        out[~ok] = np.nan
    return out


def _team_speed(match, lag=2):
    """(T, 2) mean speed (m/s) of each team's tracked players, from positions over +-lag."""
    x, y = match.player_x, match.player_y
    dx = np.full(x.shape, np.nan, dtype=np.float32)
    dy = np.full(y.shape, np.nan, dtype=np.float32)
    dx[lag:-lag] = x[2 * lag:] - x[:-2 * lag]
    dy[lag:-lag] = y[2 * lag:] - y[:-2 * lag]
    speed = np.hypot(dx, dy) * match.fps / (2.0 * lag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.column_stack([np.nanmean(speed[:, match.team == t], axis=1)
                                for t in (HOME, AWAY)])


def features(match):
    """(X (T, F) float32, names, kinds): kinds[f] is "anti", "pair" or "sym"; a pair is two
    adjacent columns, home then away. NaN where a value is unknown."""
    fps = match.fps
    length = match.pitch[0]
    x = gapteam.attacking_x(match)
    y = match.player_y.astype(np.float64)
    home, away = match.team == HOME, match.team == AWAY
    cols, names, kinds = [], [], []

    def add(name, value, kind="anti"):
        cols.append(np.asarray(value, dtype=np.float64))
        names.append(name)
        kinds.append(kind)

    spread = gapteam.spread(match)
    adv = gapteam.advance(match)                                  # 2 s
    add("spread", spread)
    add("gyration", gapteam.spread(match, dict(gapteam.PARAMS, spread="gyration")))
    add("line", gapteam.line(match))
    add("advance2", adv)
    add("advance1", handover.feature(match, "advance"))
    add("reversal", handover.feature(match, "reversal"))
    for name in handover.FEATURES:
        if name[0] == "d":
            add(name, handover.feature(match, name))
    for s in CONTEXT_S:
        w = int(round(s * fps))
        add("spread_mean{:g}".format(s), _window_mean(spread, -(w // 2), w - w // 2, cut=False))
    side = int(round(SIDE_S * fps))
    for name, value in (("spread", spread), ("advance2", adv)):
        add(name + "_past", _window_mean(value, -side, 0))
        add(name + "_future", _window_mean(value, 0, side))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        add("tilt", np.nanmean(x, axis=1) - length / 2.0)
        own = (x[:, home], length - x[:, away])                     # depth from own goal
        ys = (y[:, home], y[:, away])
        velocity = handover.team_velocity(match, 1.0)
        velocity = (velocity[:, 0], -velocity[:, 1])                # along own attack
        speed = _team_speed(match)
        pairs = {
            "depth": [np.nanmean(d, axis=1) for d in own],
            "front": [np.nanmax(d, axis=1) for d in own],
            "width": [np.nanmax(v, axis=1) - np.nanmin(v, axis=1) for v in ys],
            "velocity": list(velocity),
            "speed": [speed[:, 0], speed[:, 1]],
            "tracked": [np.isfinite(d).sum(axis=1).astype(float) for d in own],
        }
        speed_all = np.nanmean(np.column_stack(pairs["speed"]), axis=1)
    for name, (h, a) in pairs.items():
        add(name + "_home", h, "pair")
        add(name + "_away", a, "pair")
    add("speed_all", speed_all, "sym")
    return np.column_stack(cols).astype(np.float32), names, kinds


def swap(X, kinds):
    """The feature vectors with home and away exchanged."""
    out = X.copy()
    i = 0
    while i < len(kinds):
        if kinds[i] == "anti":
            out[:, i] = -X[:, i]
            i += 1
        elif kinds[i] == "pair":
            out[:, i], out[:, i + 1] = X[:, i + 1], X[:, i]
            i += 2
        else:
            i += 1
    return out


def fit(matches, params=PARAMS):
    """A classifier of P(home | alive) from labelled (degraded) matches."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows, targets = [], []
    kinds = None
    for match in matches:
        X, _, kinds = features(match)
        labels = match.labels
        alive = np.flatnonzero(np.isin(labels, pm.POSSESSION_LABELS) & match.in_play_mask())
        pick = alive[::params["step"]]
        rows.append(X[pick])
        targets.append((labels[pick] == pm.H).astype(np.int8))
    X = np.vstack(rows)
    y = np.concatenate(targets)
    X = np.vstack((X, swap(X, kinds)))
    y = np.concatenate((y, 1 - y))
    model = HistGradientBoostingClassifier(
        max_iter=params["max_iter"], learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2"], early_stopping=False, random_state=0)
    model.fit(X, y)
    return model


def home_probability(model, match, params=PARAMS):
    """Per frame P(home | alive), symmetrised over a team swap; NaN outside the halves."""
    X, _, kinds = features(match)
    p = 0.5 * (model.predict_proba(X)[:, 1] + 1.0 - model.predict_proba(swap(X, kinds))[:, 1])
    p = np.clip(p, params["clip"], 1.0 - params["clip"])
    p[~match.in_play_mask()] = np.nan
    return p


def log_odds(match, params=PARAMS):
    """Per frame log P(home) / P(away) from the model trained without this match."""
    items = tuple(sorted(params.items()))
    p = home_probability(idsse_model(match.name, items), match, params)
    return np.log(p) - np.log1p(-p)


@lru_cache(maxsize=None)
def _training_match(match_id, labels):
    from solution.simulate import degrade, train_version
    from solution.sources import idsse
    return degrade(idsse.load(match_id, labels=labels), TRAIN_SEED, train_version())


@lru_cache(maxsize=None)
def idsse_model(exclude="", items=()):
    """fit() on the degraded IDSSE dev matches minus `exclude` (a Match name); never the
    hold-out. `items` is sorted (name, value) pairs of the full params, hashable."""
    from solution.sources import idsse
    params = dict(items) if items else dict(PARAMS)
    ids = [m for m in idsse.DEV_IDS if "idsse_" + m != exclude]
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    return fit([_training_match(m, params["labels"]) for m in ids], params)
