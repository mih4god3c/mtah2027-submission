"""The keeper's ball against the goal kick: a learned per-frame state model in the keeper zone.

denis's video review (2026-11-11, solution/keeper.py): on BAR-ATH the keeper holding the ball is
decoded D, and a goal kick is first decoded as a team. Both happen around one goal with the
ball near the keeper or lost, so one model reads them together: in the "keeper zone" of a team
T, is the ball alive with T (the keeper's ball, or T playing out), alive with T's opponent, or
dead (a goal kick, a corner, the ball out behind the goal)?

The zone (positions and the ball only, no labels):
  - the last seen ball (carried forward inside the half, at most `age_s` old) lies within
    `zone_depth_m` of T's goal line (or beyond it) and `zone_half_m` of the pitch's long axis;
  - and the ball is unseen now, or seen within `near_m` of T's keeper (keeper.keepers: T's
    deepest tracked player), or seen inside T's goal area widened by `six_pad_m`;
  - and T's keeper is tracked. One team at most: the two areas are 70 m apart.
Features, all read from T's side, so a team swap maps a frame onto its mirror (no side or
colour can be learned):
  - the ball: seen, seconds since seen, the last seen ball's depth from T's goal line and
    lateral offset, its distance to T's keeper and to the nearest opponent, seen share over the
    past and next SIDE_S;
  - T's keeper: depth, lateral offset, speed (m/s), depth change over the past and next
    MOVE_S, share of the past MOVE_S beyond his goal line;
  - the opponents: how many in T's penalty area, mean depth, mean depth velocity (positive =
    leaving T's goal) over the past and next SIDE_S;
  - T's outfield: mean depth, depth velocity over the next SIDE_S, lateral spread;
  - everyone: the collective speed rank (decoder.speed_feature, D9) and its past and next
    SIDE_S means; seconds since the zone began.
Target, oriented: 0 alive with T, 1 alive with T's opponent, 2 dead (keep2 truth; D is the same
in every label variant). Output for the decoder (log_lik): per frame, log P(state | x) minus log
of the training share of the state in the zone, as (H, A, D) columns; NaN off the zone.

Training, fixed before the first run (2026-11-11): leave-one-match-out on the IDSSE dev matches
(never the hold-out, D3), each degraded by the training simulator with framemodel.TRAIN_SEED,
then the D12 ball stress, then keeper.hide_keeper_ball at keep `train_hide_keep` 0.5, so the
model sees the keeper's ball both seen and lost and cannot learn "lost near the keeper = out".
`train_lose_keep` < 1 adds keeper.lose_ball_near_goal (v7_keeper2: 0.5, with train_hide_keep 1,
fixed 2026-11-11 after the BAR-ATH gap check and before any league result): the ball also lost
out behind the goal until the keeper has it, as BAR-ATH's tracker seems to.
On BAR-ATH all five train it. HistGradientBoostingClassifier as in framemodel and deadmodel.
"""

import warnings
from functools import lru_cache

import numpy as np

from solution import decoder, keeper
from solution.framemodel import TRAIN_SEED, _window_mean
from solution.match import AWAY, HOME, pm

PARAMS = {
    "age_s": 20.0,
    "zone_depth_m": 20.0,
    "zone_half_m": 26.0,
    "near_m": 4.0,
    "six_pad_m": 2.0,
    "train_hide_keep": 0.5,
    "train_lose_keep": 1.0,     # keeper.lose_ball_near_goal in training (1: off); v7_keeper2 0.5
    "labels": "keep2",
    "step": 2,
    "clip": 0.02,
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "l2": 1.0,
}
SIDE_S = 3.0
MOVE_S = 5.0
SPEED_LAG = 2
TRAIN_STRESS = (0.05, 3.4, 0.26)      # D12, as tools/deadball_check.STRESS
FEATURE_NAMES = [
    "seen", "age_s", "ball_depth", "ball_lat", "ball_keeper_m", "ball_opp_m",
    "seen_past", "seen_future",
    "keeper_depth", "keeper_lat", "keeper_speed", "keeper_move_past", "keeper_move_future",
    "keeper_beyond_past",
    "opp_in_box", "opp_depth", "opp_vel_past", "opp_vel_future",
    "own_depth", "own_vel_future", "own_spread",
    "speed_rank", "speed_rank_past", "speed_rank_future", "zone_age_s",
]


def last_seen(match):
    """(x, y, age_s): the last seen ball carried forward inside each half; NaN before the
    half's first sighting."""
    n = match.n_frames
    lx, ly, age = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    seen = match.ball_seen
    for start, stop in match.halves:
        idx = np.where(seen[start:stop], np.arange(stop - start), -1)
        idx = np.maximum.accumulate(idx)
        ok = idx >= 0
        frames = start + idx[ok]
        lx[start:stop][ok] = match.ball_x[frames]
        ly[start:stop][ok] = match.ball_y[frames]
        age[start:stop][ok] = (np.arange(stop - start)[ok] - idx[ok]) / float(match.fps)
    return lx, ly, age


def _velocity(values, fps, lag=SPEED_LAG):
    """Centred difference per second along axis 0; NaN at the ends."""
    out = np.full(values.shape, np.nan)
    if len(values) > 2 * lag:
        out[lag:-lag] = (values[2 * lag:] - values[:-2 * lag]) * fps / (2.0 * lag)
    return out


def _shift(values, frames):
    """values[t + frames], NaN past either end."""
    out = np.full(len(values), np.nan)
    if frames > 0:
        out[:-frames] = values[frames:]
    elif frames < 0:
        out[-frames:] = values[:frames]
    else:
        out[:] = values
    return out


def zone(match, params=PARAMS):
    """(T,) int8: the team whose keeper zone the frame is in, -1 for none."""
    length, width = match.pitch
    lx, ly, age = last_seen(match)
    kx, ky = keeper.keeper_xy(match)
    seen = match.ball_seen
    out = np.full(match.n_frames, -1, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        lateral = np.abs(ly - width / 2.0)
        for team in (HOME, AWAY):
            d = keeper.depth(match, team, lx)
            near_goal = (d <= params["zone_depth_m"]) & (lateral <= params["zone_half_m"]) & \
                (age <= params["age_s"])
            keeper_near = np.hypot(match.ball_x - kx[:, team],
                                   match.ball_y - ky[:, team]) < params["near_m"]
            six = (d <= keeper.SIX_DEPTH + params["six_pad_m"]) & \
                (lateral <= keeper.SIX_HALF_WIDTH + params["six_pad_m"])
            gate = ~seen | keeper_near | (seen & six)
            on = near_goal & gate & np.isfinite(kx[:, team]) & match.in_play_mask()
            out[on & (out < 0)] = team
    return out


def _team_features(match, team, lx, ly, age, kx, ky, speed_rank):
    """(T, F) features read from `team`'s side, for every frame (the zone picks rows)."""
    fps, width = match.fps, match.pitch[1]
    other = AWAY if team == HOME else HOME
    x = match.player_x.astype(np.float64)
    y = match.player_y.astype(np.float64)
    own_cols, opp_cols = match.team == team, match.team == other
    seen = match.ball_seen.astype(float)
    side, move = int(round(SIDE_S * fps)), int(round(MOVE_S * fps))
    cols = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cols["seen"] = seen
        cols["age_s"] = age
        cols["ball_depth"] = keeper.depth(match, team, lx)
        cols["ball_lat"] = np.abs(ly - width / 2.0)
        cols["ball_keeper_m"] = np.hypot(lx - kx[:, team], ly - ky[:, team])
        ox, oy = x[:, opp_cols], y[:, opp_cols]
        cols["ball_opp_m"] = np.nanmin(np.hypot(ox - lx[:, None], oy - ly[:, None]), axis=1)
        cols["seen_past"] = _window_mean(seen, -side, 0)
        cols["seen_future"] = _window_mean(seen, 0, side)
        kd = keeper.depth(match, team, kx[:, team])
        cols["keeper_depth"] = kd
        cols["keeper_lat"] = np.abs(ky[:, team] - width / 2.0)
        cols["keeper_speed"] = np.hypot(_velocity(kx[:, team], fps), _velocity(ky[:, team], fps))
        cols["keeper_move_past"] = kd - _shift(kd, -move)
        cols["keeper_move_future"] = _shift(kd, move) - kd
        cols["keeper_beyond_past"] = _window_mean((kd < -0.5).astype(float), -move, 0)
        od = keeper.depth(match, team, ox)
        in_box = (od >= 0) & (od < keeper.BOX_DEPTH) & \
            (np.abs(oy - width / 2.0) < keeper.BOX_HALF_WIDTH)
        cols["opp_in_box"] = np.sum(in_box, axis=1).astype(float)
        cols["opp_depth"] = np.nanmean(od, axis=1)
        opp_vel = np.nanmean(_velocity(od, fps), axis=1)
        cols["opp_vel_past"] = _window_mean(opp_vel, -side, 0)
        cols["opp_vel_future"] = _window_mean(opp_vel, 0, side)
        # T's outfield: everyone but the keeper column of the frame
        own_d = keeper.depth(match, team, x[:, own_cols])
        own_y = y[:, own_cols]
        kcol = keeper.keepers(match)[:, team]
        own_idx = np.flatnonzero(own_cols)
        is_keeper = own_idx[None, :] == kcol[:, None]
        own_d = np.where(is_keeper, np.nan, own_d)
        own_y = np.where(is_keeper, np.nan, own_y)
        cols["own_depth"] = np.nanmean(own_d, axis=1)
        own_vel = np.nanmean(_velocity(keeper.depth(match, team, x[:, own_cols]), fps), axis=1)
        cols["own_vel_future"] = _window_mean(own_vel, 0, side)
        cols["own_spread"] = np.nanstd(own_y, axis=1)
        cols["speed_rank"] = speed_rank
        cols["speed_rank_past"] = _window_mean(speed_rank, -side, 0)
        cols["speed_rank_future"] = _window_mean(speed_rank, 0, side)
    cols["zone_age_s"] = np.zeros(match.n_frames)       # filled in features()
    return np.column_stack([cols[name] for name in FEATURE_NAMES[:-1]] + [cols["zone_age_s"]])


def features(match, params=PARAMS):
    """(X (T, F) float32, zone (T,) int8): each frame's features from its zone team's side;
    rows off the zone are NaN."""
    z = zone(match, params)
    lx, ly, age = last_seen(match)
    kx, ky = keeper.keeper_xy(match)
    rank = decoder.speed_feature(match, SPEED_LAG, "rank")
    X = np.full((match.n_frames, len(FEATURE_NAMES)), np.nan)
    for team in (HOME, AWAY):
        rows = z == team
        if rows.any():
            X[rows] = _team_features(match, team, lx, ly, age, kx, ky, rank)[rows]
    # seconds since the current zone run began
    zone_age = np.full(match.n_frames, np.nan)
    for a, b in keeper.runs_of(z >= 0):
        zone_age[a:b] = np.arange(b - a) / float(match.fps)
    X[:, FEATURE_NAMES.index("zone_age_s")] = zone_age
    return X.astype(np.float32), z


def oriented_target(labels, z):
    """0 alive with the zone team, 1 alive with its opponent, 2 dead; -1 elsewhere."""
    out = np.full(len(labels), -1, dtype=np.int8)
    for team, own, opp in ((HOME, pm.H, pm.A), (AWAY, pm.A, pm.H)):
        rows = z == team
        out[rows & (labels == own)] = 0
        out[rows & (labels == opp)] = 1
        out[rows & (labels == pm.D)] = 2
    return out


def fit(matches, params=PARAMS):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows, targets = [], []
    for match in matches:
        X, z = features(match, params)
        y = oriented_target(match.labels, z)
        pick = np.flatnonzero(y >= 0)[::params["step"]]
        rows.append(X[pick])
        targets.append(y[pick])
    model = HistGradientBoostingClassifier(
        max_iter=params["max_iter"], learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2"], early_stopping=False, random_state=0)
    targets = np.concatenate(targets)
    model.fit(np.vstack(rows), targets)
    model.class_share_ = np.bincount(targets, minlength=3) / float(len(targets))
    return model


@lru_cache(maxsize=None)
def _training_match(match_id, labels, hide_keep, lose_keep=1.0):
    from solution.simulate import add_ball_noise, degrade, train_version
    from solution.sources import idsse
    raw = idsse.load(match_id, labels=labels)
    match = degrade(raw, TRAIN_SEED, train_version())
    match = add_ball_noise(match, *TRAIN_STRESS[:2], seed=TRAIN_SEED, smooth_s=TRAIN_STRESS[2])
    match = keeper.hide_keeper_ball(match, raw, hide_keep, TRAIN_SEED)
    return keeper.lose_ball_near_goal(match, raw, lose_keep, TRAIN_SEED)


@lru_cache(maxsize=None)
def idsse_model(exclude="", items=()):
    """fit() on the IDSSE dev matches minus `exclude` (a Match name); never the hold-out."""
    from solution.sources import idsse
    params = dict(items) if items else dict(PARAMS)
    ids = [m for m in idsse.DEV_IDS if "idsse_" + m != exclude]
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    return fit([_training_match(m, params["labels"], params["train_hide_keep"],
                                params["train_lose_keep"]) for m in ids], params)


def probabilities(match, params=PARAMS):
    """((T, 3) P(own alive, opponent alive, dead) from the model trained without this match,
    NaN off the zone; zone (T,))."""
    model = idsse_model(match.name, tuple(sorted(params.items())))
    X, z = features(match, params)
    p = np.full((match.n_frames, 3), np.nan)
    rows = z >= 0
    if rows.any():
        q = np.clip(model.predict_proba(X[rows]), params["clip"], 1.0)
        p[rows] = q / q.sum(axis=1, keepdims=True)
    return p, z


def log_lik(match, params=PARAMS):
    """(T, 3) log P(x | H), log P(x | A), log P(x | D) up to a per-frame constant: the model's
    log posterior minus its training class share. NaN off the zone."""
    model = idsse_model(match.name, tuple(sorted(params.items())))
    p, z = probabilities(match, params)
    ll = np.log(p) - np.log(np.maximum(model.class_share_, 1e-6))
    out = np.full((match.n_frames, 3), np.nan)
    home, away = z == HOME, z == AWAY
    out[home, pm.H], out[home, pm.A], out[home, pm.D] = ll[home, 0], ll[home, 1], ll[home, 2]
    out[away, pm.A], out[away, pm.H], out[away, pm.D] = ll[away, 0], ll[away, 1], ll[away, 2]
    return out
