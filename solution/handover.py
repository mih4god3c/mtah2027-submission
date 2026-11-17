"""Hand-over moment features from the players: candidate cues for C2c (PLAN.md C2c, D17).

D17 closed hand-over timing inside ball gaps: the decoder would place short possessions given
ideal evidence (the capacity oracle, +0.050 to +0.072), but spread (gapteam.spread) is a
posture signal. It flips within +-1 s of only 5% of true in-gap hand-overs and lags them by a
median 3 s. This module holds physically defined per-frame features that might mark the moment
itself, for `tools/gap_check.py --events` to score before anything reaches the decoder.

Every feature reads player positions, plus where the ball was last seen for "leave" and
"crowd". Simulator v1 drops and ghosts players but keeps their positions, so the proxy
represents these fairly (D15, D16). Signed features point towards home, in home's attacking
frame (gapteam.attacking_x: home attacks +x'):
  - "spread": gapteam.spread, as a centred mean over `spread_s`. The baseline: posture;
  - "advance": the mean velocity along x' of all tracked players, over `advance_s` (centred).
    Everyone runs towards the goal the team in possession attacks;
  - "reversal": both teams' run direction. Per team, the mean velocity along x'; where both
    run the same way faster than `agree_ms`, the slower team's speed signed by that way, NaN
    elsewhere. Its sign flips when both teams turn round;
  - "leave": home minus away mean radial speed away from the last-seen ball, of the players
    within `near_m` of it, for `hold_s` after the ball was last seen (NaN later). The team
    that wins the ball moves away to make space, the team that loses it closes in;
  - "crowd": players of both teams within `crowd_m` of the last-seen ball, for `hold_s` after
    it was last seen. Unsigned: a duel gathers players, the hand-over disperses them;
  - "d<feature><w>": past-minus-future difference, mean over [t, t + w) minus mean over
    [t - w, t), of spread, advance (over `fast_s`) or gapteam.line, w in DIFF_S. A change
    detector centred on t: it peaks where the feature moves fastest.

Events (`events`): a "cross" feature fires where it crosses zero, towards home (H) or away
(A), NaN frames skipped. A "peak" feature fires at local maxima of |value| (signed; the
direction is the sign) or of value (unsigned; no direction) above the `peak_q` quantile of the
match's in-play frames, at most one per `suppress_s`. The quantile is label-free, so the same
rule runs on BAR-ATH. numpy only, no randomness.
"""

import warnings

import numpy as np

from solution import gapteam
from solution.match import AWAY, HOME, pm

PARAMS = {
    "spread_s": 1.0,        # centred mean of gapteam.spread
    "advance_s": 1.0,       # window of the players' velocity along x' ("advance", "reversal")
    "fast_s": 0.4,          # ... and for the velocity the "dadvance" difference reads
    "agree_ms": 0.3,        # both teams count as running one way above this speed
    "vel_s": 0.4,           # centred difference for player velocities ("leave")
    "near_m": 15.0,         # players this close to the last-seen ball count for "leave"
    "crowd_m": 8.0,         # ... and for "crowd"
    "hold_s": 3.0,          # the last-seen ball is used for this long after it was seen
    "leave_s": 0.5,         # centred mean of "leave"
    "peak_q": 0.9,          # peak events: above this quantile of |value| over in-play frames
    "suppress_s": 1.0,      # ... at most one per this window, the largest
}
DIFF_S = (0.5, 1.0, 2.0)
# name: (kind, signed). Filled in below for the past-minus-future differences.
FEATURES = {"spread": ("cross", True), "advance": ("cross", True),
            "reversal": ("cross", True), "leave": ("cross", True),
            "crowd": ("peak", False)}
for _base in ("spread", "advance", "line"):
    for _w in DIFF_S:
        FEATURES["d{}{:g}".format(_base, _w)] = ("peak", True)


def _frames(seconds, fps):
    return max(1, int(round(seconds * fps)))


def moving_mean(values, frames):
    """Centred moving mean that skips NaN; NaN where the window has nothing."""
    finite = np.isfinite(values)
    kernel = np.ones(frames)
    total = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    count = np.convolve(finite.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def past_minus_future(values, frames):
    """mean(values[t:t + w]) - mean(values[t - w:t]), NaN skipped; NaN where either is empty."""
    finite = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    count = np.concatenate(([0.0], np.cumsum(finite.astype(float))))
    n = len(values)
    t = np.arange(n)
    lo, hi = np.maximum(t - frames, 0), np.minimum(t + frames, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        past = (total[t] - total[lo]) / (count[t] - count[lo])
        future = (total[hi] - total[t]) / (count[hi] - count[t])
    out = future - past
    out[(t - frames < 0) | (t + frames > n)] = np.nan
    return out


def team_velocity(match, seconds):
    """(T, 2) mean velocity along x' (m/s) of each team's players tracked at both window ends."""
    half = max(1, int(round(seconds * match.fps / 2.0)))
    x = gapteam.attacking_x(match)
    out = np.full((match.n_frames, 2), np.nan)
    if len(x) > 2 * half:
        step = (x[2 * half:] - x[:-2 * half]) * match.fps / (2.0 * half)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for k, team in enumerate((HOME, AWAY)):
                out[half:-half, k] = np.nanmean(step[:, match.team == team], axis=1)
    return out


def reversal(match, params=PARAMS):
    """Both teams' run direction: the slower team's speed along x', signed, where they agree."""
    v = team_velocity(match, params["advance_s"])
    with np.errstate(invalid="ignore"):
        same = (np.sign(v[:, 0]) == np.sign(v[:, 1])) & (np.min(np.abs(v), axis=1) >
                                                          params["agree_ms"])
        return np.where(same, np.sign(v[:, 0]) * np.min(np.abs(v), axis=1), np.nan)


def last_seen_ball(match, params=PARAMS):
    """(bx, by): where the ball was last seen in the half, NaN beyond hold_s after that."""
    n = match.n_frames
    last = np.full(n, -1, dtype=np.int64)
    for start, stop in match.halves:              # the last sighting never crosses a half
        index = np.where(match.ball_seen[start:stop], np.arange(start, stop), -1)
        last[start:stop] = np.maximum.accumulate(index)
    ok = (last >= 0) & (np.arange(n) - last <= _frames(params["hold_s"], match.fps))
    bx = np.where(ok, match.ball_x[np.maximum(last, 0)], np.nan)
    by = np.where(ok, match.ball_y[np.maximum(last, 0)], np.nan)
    return bx.astype(np.float64), by.astype(np.float64)


def leave(match, params=PARAMS):
    """Home minus away mean speed away from the last-seen ball, players within near_m of it."""
    bx, by = last_seen_ball(match, params)
    px, py = match.player_x.astype(np.float64), match.player_y.astype(np.float64)
    half = max(1, int(round(params["vel_s"] * match.fps / 2.0)))
    vx, vy = np.full(px.shape, np.nan), np.full(py.shape, np.nan)
    if len(px) > 2 * half:
        scale = match.fps / (2.0 * half)
        vx[half:-half] = (px[2 * half:] - px[:-2 * half]) * scale
        vy[half:-half] = (py[2 * half:] - py[:-2 * half]) * scale
    with np.errstate(invalid="ignore", divide="ignore"):
        dx, dy = px - bx[:, None], py - by[:, None]
        r = np.hypot(dx, dy)
        radial = (vx * dx + vy * dy) / r
        use = np.isfinite(radial) & (r <= params["near_m"]) & (r > 0)
    means = []
    for team in (HOME, AWAY):
        mine = use & (match.team == team)[None, :]
        count = mine.sum(axis=1)
        total = np.where(mine, radial, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means.append(np.where(count > 0, total / count, np.nan))
    return moving_mean(means[0] - means[1], _frames(params["leave_s"], match.fps))


def crowd(match, params=PARAMS):
    """Past minus future count of players within crowd_m of the last-seen ball, over 1 s:
    positive where players leave the place the ball was last seen."""
    bx, by = last_seen_ball(match, params)
    with np.errstate(invalid="ignore"):
        near = np.hypot(match.player_x - bx[:, None], match.player_y - by[:, None]) <= \
            params["crowd_m"]
    count = np.where(np.isfinite(bx), near.sum(axis=1), np.nan).astype(np.float64)
    return -past_minus_future(count, _frames(1.0, match.fps))


def feature(match, name, params=PARAMS):
    """Per-frame value of one feature of FEATURES (signed towards home where signed)."""
    fps = match.fps
    if name == "spread":
        return moving_mean(gapteam.spread(match), _frames(params["spread_s"], fps))
    if name == "advance":
        return gapteam.advance(match, dict(gapteam.PARAMS, advance_s=params["advance_s"]))
    if name == "reversal":
        return reversal(match, params)
    if name == "leave":
        return leave(match, params)
    if name == "crowd":
        return crowd(match, params)
    for base in ("spread", "advance", "line"):
        if name.startswith("d" + base):
            w = float(name[len(base) + 1:])
            if base == "spread":
                raw = gapteam.spread(match)
            elif base == "advance":
                raw = gapteam.advance(match, dict(gapteam.PARAMS, advance_s=params["fast_s"]))
            else:
                raw = gapteam.line(match)
            return past_minus_future(raw, _frames(w, fps))
    raise ValueError("unknown feature {!r}".format(name))


def crossings(value, start, stop):
    """(frames, teams) where value crosses zero inside [start, stop), NaN frames skipped:
    H where it turns from <= 0 to > 0, A where it turns from >= 0 to < 0."""
    frames = np.arange(start, stop)
    keep = np.isfinite(value[start:stop])
    frames, v = frames[keep], value[start:stop][keep]
    up = (v[:-1] <= 0) & (v[1:] > 0)
    down = (v[:-1] >= 0) & (v[1:] < 0)
    at = frames[1:]
    out = np.concatenate((at[up], at[down]))
    teams = np.concatenate((np.full(up.sum(), pm.H), np.full(down.sum(), pm.A)))
    order = np.argsort(out, kind="stable")
    return out[order].astype(np.int64), teams[order].astype(np.int8)


def peaks(score, start, stop, threshold, reach):
    """Frames in [start, stop) where score is finite, >= threshold and the largest within
    +-reach frames (ties to the earliest)."""
    s = np.where(np.isfinite(score[start:stop]), score[start:stop], -np.inf)
    padded = np.concatenate((np.full(reach, -np.inf), s, np.full(reach, -np.inf)))
    window = np.lib.stride_tricks.sliding_window_view(padded, 2 * reach + 1)
    earlier = window[:, :reach].max(axis=1)
    later = window[:, reach + 1:].max(axis=1)
    return (start + np.flatnonzero((s > earlier) & (s >= later) & (s >= threshold))).astype(
        np.int64)


def events(match, name, value=None, params=PARAMS):
    """(frames, teams) of one feature's events in the halves; team -1 for unsigned features."""
    kind, signed = FEATURES[name]
    value = feature(match, name, params) if value is None else value
    frames, teams = [], []
    if kind == "cross":
        for start, stop in match.halves:
            f, k = crossings(value, start, stop)
            frames.append(f)
            teams.append(k)
    else:
        score = np.abs(value) if signed else value
        inside = match.in_play_mask() & np.isfinite(score)
        if not inside.any():
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8)
        threshold = np.quantile(score[inside], params["peak_q"])
        reach = _frames(params["suppress_s"], match.fps)
        for start, stop in match.halves:
            f = peaks(score, start, stop, threshold, reach)
            frames.append(f)
            teams.append(np.where(value[f] > 0, pm.H, pm.A).astype(np.int8) if signed
                         else np.full(len(f), -1, dtype=np.int8))
    return np.concatenate(frames), np.concatenate(teams)
