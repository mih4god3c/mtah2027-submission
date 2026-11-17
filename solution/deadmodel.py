"""Visible-ball Step 3, Step 0 only: a learned per-frame P(dead) from the players (PLAN C2b).

The decoder's dead-ball evidence is the within-match rank of collective player speed (D8, D9).
D15 put the perfect-dead-ball ceiling at +0.089 under the D12 stress, mostly at the edges of
stoppages (start +0.037, end +0.018) and false stoppages (+0.020). On those frames the players
keep moving, so a single speed value cannot see them. This model asks whether the players' pattern
around a frame (how speed changes over the next and last few seconds, how evenly they move, where
they stand) can.

Rules it keeps, as C3's frame model does (D22):
  - players only, never the ball (D15: the proxy's still dead ball comes from DFL's feed, and the
    ball beyond a line is overstated, D7);
  - every speed is read as its rank within the match, never in m/s: BAR-ATH plays about 10%
    slower than the IDSSE dev matches (D9). The other features are positions, counts or
    scale-free ratios;
  - dead ball has no team, so every feature is unchanged by a team swap (tested);
  - leave-one-match-out on the IDSSE dev matches degraded with framemodel.TRAIN_SEED, never the
    hold-out (D3). Graded frames, every STEP-th, label D against H or A.
Not read by the decoder: tools/deadmodel_check.py scores it at frame level against the speed rank.
`version` "v2" is the same model without absolute positions (DROPPED; EXPERIMENTS section 16).
"""

import warnings
from functools import lru_cache

import numpy as np

from solution import deadball, decoder
from solution.framemodel import TRAIN_SEED, _window_mean
from solution.match import pm

PARAMS = {
    "step": 5,              # train on every STEP-th graded frame
    "clip": 0.02,           # probabilities are clipped to [clip, 1 - clip]
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 200,
    "l2": 1.0,
}
CONTEXT_S = (1.0, 3.0, 8.0)     # past-only and future-only means of the speed rank
SLOW_MS = 1.5                   # a player slower than this is "standing" (only its share's rank
                                # within the match is used, so the threshold's scale washes out)
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
# Feature sets. "v1" is the D25 model, every column of features(). "v2" (EXPERIMENTS section
# 16, set once before its first run) keeps only ranks, scale-free ratios and the beyond-line
# windows: it drops the columns read in metres from absolute positions (gyration, |tilt|,
# nearest player to a touchline and a goal line, box count) and the tracked count.
DROPPED = {"v1": (),
           "v2": ("gyration", "tilt_abs", "tracked", "touchline_min", "goalline_min", "in_box")}


def _rank(values, mask):
    """values as their quantile among the `mask` frames where they are finite (decoder's rank)."""
    out = np.full(len(values), np.nan)
    use = mask & np.isfinite(values)
    if use.any():
        ordered = np.sort(values[use])
        out[use] = np.searchsorted(ordered, values[use], side="right") / float(len(ordered))
    return out


def _player_speeds(match, lag=2):
    """(T, S) per-player speed from positions over +-lag frames (as simulate.mean_player_speed)."""
    x, y = match.player_x, match.player_y
    dx = np.full(x.shape, np.nan, dtype=np.float32)
    dy = np.full(y.shape, np.nan, dtype=np.float32)
    dx[lag:-lag] = x[2 * lag:] - x[:-2 * lag]
    dy[lag:-lag] = y[2 * lag:] - y[:-2 * lag]
    return np.hypot(dx, dy) * match.fps / (2.0 * lag)


FEATURE_NAMES = (["speed_rank"]
                 + ["rank_{}{:g}".format(kind, s) for s in CONTEXT_S
                    for kind in ("past", "future", "change")]
                 + ["speed_cv", "slow_share_rank", "speed_max_rank", "speed_p25_rank", "gyration",
                    "tilt_abs", "tracked", "touchline_min", "goalline_min", "in_box",
                    "beyond_past3", "beyond_future3", "cv_past3", "cv_future3"])


def features(match):
    """(X (T, F) float32, names). Every column is unchanged by swapping the teams."""
    fps = match.fps
    length, width = match.pitch
    graded = match.in_play_mask()
    cols, names = [], []

    def add(name, value):
        cols.append(np.asarray(value, dtype=np.float64))
        names.append(name)

    rank = decoder.speed_feature(match, 2, "rank")
    add("speed_rank", rank)
    for s in CONTEXT_S:
        w = int(round(s * fps))
        past, future = _window_mean(rank, -w, 0), _window_mean(rank, 0, w)
        add("rank_past{:g}".format(s), past)
        add("rank_future{:g}".format(s), future)
        add("rank_change{:g}".format(s), future - past)
    speeds = _player_speeds(match)
    x, y = match.player_x.astype(np.float64), match.player_y.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(speeds, axis=1)
        tracked = np.isfinite(speeds).sum(axis=1).astype(float)
        add("speed_cv", np.nanstd(speeds, axis=1) / mean)               # scale-free
        add("slow_share_rank", _rank(np.nansum(speeds < SLOW_MS, axis=1) / tracked, graded))
        add("speed_max_rank", _rank(np.nanmax(speeds, axis=1), graded))
        add("speed_p25_rank", _rank(np.nanpercentile(speeds, 25, axis=1), graded))
        cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
        add("gyration", np.sqrt(np.nanmean((x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2,
                                           axis=1)))
        add("tilt_abs", np.abs(cx - length / 2.0))
        add("tracked", tracked)
        add("touchline_min", np.nanmin(np.minimum(y, width - y), axis=1))
        add("goalline_min", np.nanmin(np.minimum(x, length - x), axis=1))
        in_box = ((np.minimum(x, length - x) < BOX_DEPTH) &
                  (np.abs(y - width / 2.0) < BOX_HALF_WIDTH))
        add("in_box", in_box.sum(axis=1).astype(float))
    beyond = deadball.beyond_line(match).astype(float)
    side = int(round(3.0 * fps))
    add("beyond_past3", _window_mean(beyond, -side, 0))
    add("beyond_future3", _window_mean(beyond, 0, side))
    cv = cols[names.index("speed_cv")]
    add("cv_past3", _window_mean(cv, -side, 0))
    add("cv_future3", _window_mean(cv, 0, side))
    assert names == FEATURE_NAMES, "FEATURE_NAMES is out of date"
    return np.column_stack(cols).astype(np.float32), names


def columns(version):
    """Indices of features()'s columns that model `version` reads."""
    names = FEATURE_NAMES
    return [i for i, name in enumerate(names) if name not in DROPPED[version]]


def select(X, version):
    return X if not DROPPED[version] else X[:, columns(version)]


def fit(matches, params=PARAMS, version="v1"):
    """A classifier of P(dead | graded frame) from labelled (degraded) matches."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows, targets = [], []
    for match in matches:
        X = select(features(match)[0], version)
        graded = np.flatnonzero(np.isin(match.labels, (pm.H, pm.A, pm.D)) & match.in_play_mask())
        pick = graded[::params["step"]]
        rows.append(X[pick])
        targets.append((match.labels[pick] == pm.D).astype(np.int8))
    model = HistGradientBoostingClassifier(
        max_iter=params["max_iter"], learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2"], early_stopping=False, random_state=0)
    targets = np.concatenate(targets)
    model.fit(np.vstack(rows), targets)
    model.dead_share_ = float(np.mean(targets))      # the training prior, for log_ratio
    return model


def dead_probability(match, params=PARAMS, version="v1"):
    """Per frame P(dead) from the model trained without this match; NaN outside the halves."""
    model = idsse_model(match.name, tuple(sorted(params.items())), version)
    p = np.clip(model.predict_proba(select(features(match)[0], version))[:, 1], params["clip"],
                1.0 - params["clip"])
    p[~match.in_play_mask()] = np.nan
    return p


def log_ratio(match, params=PARAMS, version="v1"):
    """Per frame log P(x | dead) - log P(x | alive): the model's log-odds minus its training
    prior's, so the decoder's duration priors are not counted twice. NaN outside the halves."""
    p = dead_probability(match, params, version)
    share = idsse_model(match.name, tuple(sorted(params.items())), version).dead_share_
    return np.log(p) - np.log1p(-p) - (np.log(share) - np.log1p(-share))


@lru_cache(maxsize=None)
def idsse_model(exclude="", items=(), version="v1"):
    """fit() on the degraded IDSSE dev matches minus `exclude`; never the hold-out. Dead ball is
    the same in every label variant (relabel keeps D), so the labels are DFL's."""
    from solution.simulate import degrade, train_version
    from solution.sources import idsse
    params = dict(items) if items else dict(PARAMS)
    ids = [m for m in idsse.DEV_IDS if "idsse_" + m != exclude]
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    sim = train_version()
    return fit([degrade(idsse.load(m, labels="dfl"), TRAIN_SEED, sim) for m in ids], params,
               version)
