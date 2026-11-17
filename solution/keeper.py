"""Keeper geometry: each team's own goal and keeper, the keeper on the ball, and a ball-loss stress.

denis's video review (2026-11-11): on BAR-ATH the keeper holding the ball (alive, the keeper's
team) is decoded D, and a goal kick (dead from the ball going out) is first decoded as a team.
His reading: the tracker loses the ball in the keeper's hands, so the hold is a ball gap, and in
a gap the dead-ball model sees slow players walking back and says D. Simulator v1 drops the
ball on a fixed timing profile, blind to the play, so the proxy never loses it in the keeper's
hands more than anywhere else. `hide_keeper_ball` is the stress that does, as `thin_beyond` is
for the ball beyond a line.

Everything here reads positions only (no labels), so it runs on BAR-ATH as on the proxy:
  - `defends_zero(match)`: per frame and team, whether its own goal line is x = 0 (from
    gapteam.attack_sign, the halves' mean positions);
  - `depth(match, team, x)`: metres from that team's own goal line, negative beyond it;
  - `keepers(match)`: per frame and team, the column of its deepest tracked player, the keeper
    in practice whenever the ball is near its goal (-1 when the team has nobody tracked);
  - `on_ball(match)`: [(start, stop, team)] stretches where the seen ball is within `radius_m`
    of that team's keeper and the keeper stands inside his own penalty area, at least `min_s`
    long after closing gaps of `gap_s`. With hands or feet: without the ball's height the two
    are the same here.
numpy only.
"""

import warnings

import numpy as np

from solution.gapteam import attack_sign
from solution.match import AWAY, HOME

BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
SIX_DEPTH, SIX_HALF_WIDTH = 5.5, 9.16
ON_BALL = {"radius_m": 2.0, "min_s": 1.0, "gap_s": 0.4}


def defends_zero(match):
    """(T, 2) bool: [:, team] is True where that team's own goal line is x = 0."""
    sign = attack_sign(match)
    home = sign >= 0
    return np.column_stack([home, ~home])


def depth(match, team, x):
    """x (any shape with T rows) as metres from `team`'s own goal line."""
    zero = defends_zero(match)[:, team]
    zero = zero.reshape((-1,) + (1,) * (np.ndim(x) - 1))
    return np.where(zero, x, match.pitch[0] - x)


def keepers(match):
    """(T, 2) int: per frame, the column of each team's deepest tracked player, -1 if none."""
    out = np.full((match.n_frames, 2), -1, dtype=np.int64)
    for team in (HOME, AWAY):
        cols = np.flatnonzero(match.team == team)
        d = depth(match, team, match.player_x[:, cols].astype(np.float64))
        known = np.isfinite(d) & np.isfinite(match.player_y[:, cols])
        pick = np.argmin(np.where(known, d, np.inf), axis=1)
        out[:, team] = np.where(known.any(axis=1), cols[pick], -1)
    return out


def keeper_xy(match, keeper=None):
    """(x, y), each (T, 2) float: each team's keeper position, NaN where none is tracked."""
    keeper = keepers(match) if keeper is None else keeper
    rows = np.arange(match.n_frames)[:, None]
    safe = np.maximum(keeper, 0)
    x = np.where(keeper >= 0, match.player_x[rows, safe], np.nan).astype(np.float64)
    y = np.where(keeper >= 0, match.player_y[rows, safe], np.nan).astype(np.float64)
    return x, y


def in_own_box(match, team, x, y):
    """Bool, x and y's shape: inside `team`'s own penalty area (goal line to 16.5 m)."""
    d = depth(match, team, x)
    with np.errstate(invalid="ignore"):
        return (d >= 0) & (d < BOX_DEPTH) & (np.abs(y - match.pitch[1] / 2.0) < BOX_HALF_WIDTH)


def runs_of(mask):
    """[(start, stop)] of True runs."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def close_gaps(spans, gap):
    """Merge runs separated by at most `gap` frames."""
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def on_ball(match, radius_m=ON_BALL["radius_m"], min_s=ON_BALL["min_s"],
            gap_s=ON_BALL["gap_s"]):
    """[(start, stop, team)] the keeper on the seen ball inside his own penalty area."""
    kx, ky = keeper_xy(match)
    inside = match.in_play_mask()
    out = []
    for team in (HOME, AWAY):
        with np.errstate(invalid="ignore"):
            near = np.hypot(match.ball_x - kx[:, team], match.ball_y - ky[:, team]) < radius_m
        mask = near & in_own_box(match, team, kx[:, team], ky[:, team]) & inside
        spans = close_gaps(runs_of(mask), int(round(gap_s * match.fps)))
        out += [(a, b, team) for a, b in spans if b - a >= min_s * match.fps]
    return sorted(out)


def hide_keeper_ball(match, clean, keep, seed=0):
    """A copy of `match` (degraded) with the ball hidden over each stretch where `clean` has the
    keeper on the ball (on_ball), with probability 1 - `keep` per stretch.

    A stress, not part of any simulator version. `clean` is the same match before degrading (its
    clean ball says where the keeper has it). Apply after add_ball_noise and thin_beyond. Its own
    random stream, so other stresses draw the same numbers with it on or off."""
    out = match.copy()
    if keep >= 1:
        return out
    from solution.simulate import _rng
    rng = _rng(match, seed, 9)
    for a, b, _ in on_ball(clean):
        if rng.rand() >= keep:
            out.ball_x[a:b] = np.nan
            out.ball_y[a:b] = np.nan
    out.notes["hide_keeper"] = keep
    return out


def keeper_near_ball(match, radius_m=ON_BALL["radius_m"]):
    """(T,) bool: the seen ball within `radius_m` of a keeper standing in his own area."""
    kx, ky = keeper_xy(match)
    out = np.zeros(match.n_frames, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for team in (HOME, AWAY):
            near = np.hypot(match.ball_x - kx[:, team], match.ball_y - ky[:, team]) < radius_m
            out |= near & in_own_box(match, team, kx[:, team], ky[:, team])
    return out


def gaps_by_keeper(match, radius_m=ON_BALL["radius_m"]):
    """{"before": (lengths, flags), "after": (lengths, flags)}: every ball gap inside a half
    (with a sighting on both sides), its length in seconds, and whether the sighting just
    before it ("before") or just after it ("after") is a keeper on the ball in his own area.
    Label-free. A tracker that loses the ball in the keeper's hands makes "after" gaps (the
    ball reappears with the keeper, before his kick or throw) longer and more common."""
    near = keeper_near_ball(match, radius_m)
    seen = match.ball_seen
    lengths, before, after = [], [], []
    for start, stop in match.halves:
        for a, b in runs_of(~seen[start:stop]):
            a, b = a + start, b + start
            if a == start or b == stop:
                continue
            lengths.append((b - a) / float(match.fps))
            before.append(near[a - 1])
            after.append(near[b])
    lengths = np.array(lengths)
    return {"before": (lengths, np.array(before, dtype=bool)),
            "after": (lengths, np.array(after, dtype=bool))}


LOSS = {"tail_s": 1.0, "cap_s": 60.0}


def lose_ball_near_goal(match, clean, keep, seed=0, tail_s=LOSS["tail_s"], cap_s=LOSS["cap_s"],
                        keep_hold=None):
    """A copy of `match` (degraded) losing the ball around each goal the way BAR-ATH's tracker
    seems to (tools/keeper_check.py --bar-ath, 2026-11-11: inside the keeper zone BAR-ATH sees
    the ball on 0.30 of frames against the proxy's 0.42, and 12.9% of its gaps end with the ball
    at a keeper against the proxy's 2.6%, a median 16 s long). Read from `clean` (the same match
    before degrading), with no labels, each episode hidden with probability 1 - `keep`:
      - out over a goal line: from the first frame the clean ball is beyond a goal line until
        it is next within ON_BALL radius of a keeper in his own area (at most `cap_s`; nothing
        if it never is): the goal kick's ball reappears when the keeper has it;
      - the keeper's ball (on_ball): hidden but for its last `tail_s`, so it reappears with the
        keeper before his kick or throw.
    `keep_hold` (default: `keep`) sets the keeper's-ball part's probability on its own.
    A stress, not part of any simulator version; apply after add_ball_noise. Its own stream."""
    out = match.copy()
    keep_hold = keep if keep_hold is None else keep_hold
    if keep >= 1 and keep_hold >= 1:
        return out
    from solution.simulate import _rng
    rng = _rng(match, seed, 10)
    fps, length = clean.fps, clean.pitch[0]
    near = keeper_near_ball(clean)
    cap, tail = int(round(cap_s * fps)), int(round(tail_s * fps))
    hide = []
    with np.errstate(invalid="ignore"):
        beyond = (clean.ball_x < 0) | (clean.ball_x > length)
    for start, stop in clean.halves:
        for a, b in runs_of(beyond[start:stop]):
            a = a + start
            later = np.flatnonzero(near[a:min(stop, a + cap)])
            if len(later):
                hide.append((a, a + later[0], keep))
    for a, b, _ in on_ball(clean):
        if b - a > tail:
            hide.append((a, b - tail, keep_hold))
    for a, b, p in sorted(hide):
        if rng.rand() >= p:
            out.ball_x[a:b] = np.nan
            out.ball_y[a:b] = np.nan
    out.notes["lose_near_goal"] = (keep, keep_hold)
    return out
