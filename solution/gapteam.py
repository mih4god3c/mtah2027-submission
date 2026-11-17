"""E5: which team has the ball while the ball is not seen, from the players alone (PLAN.md E5).

`tools/gap_check.py` (Step 0) shows what ball gaps cost `touch_kickoff` under the D12 stress:
with the true team on every gap frame where truth and prediction are both alive, pooled
spellF1 rises by +0.095. Nearly all of it is in gaps that hold an open-play hand-over, where
the decoder gets the team right on 57% of frames, little better than a coin. Possessions
under 2 s that lie mostly inside a gap are found 0.2% of the time.

The cues read player positions only. The ball is not read inside the gap (DFL freezes it in
dead ball, D15), apart from where it was last seen. Simulator v1 drops and ghosts players but
keeps their positions, so the proxy shows these cues as IDSSE players produce them. Each cue is
a per-frame number signed towards the home team, in its attacking frame: x' runs from the home
goal line (0) to the away goal line (length), so home attacks +x':
  - "advance": the mean velocity along x' of the tracked players, over `advance_s` (centred).
    Everyone runs towards the goal the team in possession attacks;
  - "line": home's defensive-line height minus away's, each measured from its own goal line.
    A team's line is the mean of its second to fourth deepest tracked players (the deepest is
    the goalkeeper). The team in possession pushes its line up;
  - "spread": home's spread minus away's, over outfield players (the deepest left out): the
    width (y range) or the radius of gyration (`spread`). A team spreads out when it has the
    ball and closes up when it defends;
  - "pressure": home minus away players within `pressure_m` of the ball where it was last seen,
    for `pressure_hold_s` after that, NaN later. The team without the ball closes it down.
The attack direction per half comes from the teams' mean x over the half, never hard-coded
(PLAN.md C1).

The decoder reads them like the speed and dead-ball terms (decoder.block_emissions, team_w):
per cue, histograms on symmetric bins (CUE_BINS) are fitted on the clean IDSSE dev labels,
leave-one-match-out (D3): one of the cue signed towards the team in possession, over alive
frames, which gives P(c | H) and, mirrored, P(c | A); and one over dead-ball frames, made
symmetric. With `team_dead` "max" the dead-ball state takes the better of the two team
likelihoods instead, so the cue can only choose between the teams, never between alive and
dead (that is C2b's question).

numpy only, no randomness.
"""

import warnings
from functools import lru_cache

import numpy as np

from solution.match import AWAY, HOME, pm

PARAMS = {
    "cues": ("advance",),    # which cues the decoder reads, in CUES
    "advance_s": 2.0,        # window of the players' velocity along x' (centred)
    "spread": "width",       # "width": y range of the outfield players; "gyration": their radius
    "pressure_m": 5.0,       # players this close to the last-seen ball ...
    "pressure_hold_s": 2.0,  # ... for this long after it was last seen
    "min_players": 5,        # tracked players a team needs for line and spread
}
# (half range, bins): each cue is clipped to +-half range and cut into an odd number of equal
# bins, so the middle bin holds 0 and bin i of c is bin (n - 1 - i) of -c.
CUE_BINS = {"advance": (3.0, 15), "line": (50.0, 15), "spread": (30.0, 15),
            "pressure": (6.5, 13)}


def attack_sign(match):
    """Per frame, +1 where home attacks +x, -1 where it attacks -x, 0 outside the halves."""
    out = np.zeros(match.n_frames, dtype=np.int8)
    home, away = match.team == HOME, match.team == AWAY
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for start, stop in match.halves:
            x = match.player_x[start:stop]
            out[start:stop] = 1 if np.nanmean(x[:, home]) <= np.nanmean(x[:, away]) else -1
    return out


def attacking_x(match):
    """(T, S) player x in home's attacking frame: 0 at the home goal line."""
    sign = attack_sign(match)[:, None]
    return np.where(sign >= 0, match.player_x, match.pitch[0] - match.player_x)


def advance(match, params=PARAMS):
    """Mean velocity along x' (m/s) of the players tracked at both ends of the window."""
    half = max(1, int(round(params["advance_s"] * match.fps / 2.0)))
    x = attacking_x(match)
    out = np.full(match.n_frames, np.nan)
    if len(x) > 2 * half:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[half:-half] = np.nanmean(x[2 * half:] - x[:-2 * half], axis=1) * match.fps / (
                2.0 * half)
    return out


def _sorted_depth(depth, columns, minimum):
    """Per frame, the team's depths sorted ascending (NaN last), NaN rows under `minimum`."""
    values = np.sort(depth[:, columns], axis=1)
    values[np.sum(np.isfinite(values), axis=1) < minimum] = np.nan
    return values


def line(match, params=PARAMS):
    """Home's defensive-line height minus away's, metres from each team's own goal line."""
    x = attacking_x(match)
    out = []
    for team, depth in ((HOME, x), (AWAY, match.pitch[0] - x)):
        values = _sorted_depth(depth, match.team == team, params["min_players"])
        out.append(values[:, 1:4].mean(axis=1))
    return out[0] - out[1]


def spread(match, params=PARAMS):
    """Home's outfield spread minus away's, metres (width or radius of gyration)."""
    x = attacking_x(match)
    out = []
    for team, depth in ((HOME, x), (AWAY, match.pitch[0] - x)):
        columns = np.flatnonzero(match.team == team)
        d = depth[:, columns]
        y = match.player_y[:, columns].astype(np.float64)
        known = np.isfinite(d) & np.isfinite(y)
        # Leave out the deepest tracked player, the goalkeeper.
        keeper = np.argmin(np.where(known, d, np.inf), axis=1)
        known[np.arange(len(d)), keeper] = False
        enough = known.sum(axis=1) >= params["min_players"] - 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if params["spread"] == "width":
                ys = np.where(known, y, np.nan)
                value = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
            elif params["spread"] == "gyration":
                dx = np.where(known, d, np.nan)
                dy = np.where(known, y, np.nan)
                value = np.sqrt(np.nanvar(dx, axis=1) + np.nanvar(dy, axis=1))
            else:
                raise ValueError("unknown spread {!r}".format(params["spread"]))
        out.append(np.where(enough, value, np.nan))
    return out[0] - out[1]


def pressure(match, params=PARAMS):
    """Home minus away players within pressure_m of the last-seen ball, up to pressure_hold_s
    after it was seen; NaN where it was not seen recently, or before it ever was in the half."""
    n = match.n_frames
    last = np.full(n, -1, dtype=np.int64)
    for start, stop in match.halves:              # the last sighting never crosses a half
        index = np.where(match.ball_seen[start:stop], np.arange(start, stop), -1)
        last[start:stop] = np.maximum.accumulate(index)
    hold = int(round(params["pressure_hold_s"] * match.fps))
    ok = (last >= 0) & (np.arange(n) - last <= hold)
    bx = np.where(ok, match.ball_x[np.maximum(last, 0)], np.nan)
    by = np.where(ok, match.ball_y[np.maximum(last, 0)], np.nan)
    with np.errstate(invalid="ignore"):
        near = np.hypot(match.player_x - bx[:, None], match.player_y - by[:, None]) <= \
            params["pressure_m"]
    count = near[:, match.team == HOME].sum(axis=1) - near[:, match.team == AWAY].sum(axis=1)
    return np.where(ok, count, np.nan).astype(np.float64)


CUES = {"advance": advance, "line": line, "spread": spread, "pressure": pressure}


def cues(match, params=PARAMS):
    """(T, K) float: the cues in params["cues"], signed towards home, NaN where unknown."""
    return np.stack([CUES[name](match, params) for name in params["cues"]], axis=1)


def bins(values, name):
    """Bin index of each value on CUE_BINS[name]; values must be finite."""
    half, n = CUE_BINS[name]
    edges = np.linspace(-half, half, n + 1)[1:-1]
    return np.searchsorted(edges, np.clip(values, -half, half), side="right")


def fit(matches, params=PARAMS):
    """[(2, n) log tables] per cue: row 0 the cue signed towards the team in possession over
    alive frames; row 1 over dead-ball frames, symmetric."""
    names = params["cues"]
    counts = [np.ones((2, CUE_BINS[name][1])) for name in names]   # add-one smoothing
    for match in matches:
        found = cues(match, params)
        labels = match.labels
        sign = np.where(labels == pm.H, 1.0, np.where(labels == pm.A, -1.0, np.nan))
        dead = labels == pm.D
        for k, name in enumerate(names):
            value = found[:, k]
            towards = value * sign
            alive = towards[np.isfinite(towards)]
            counts[k][0] += np.bincount(bins(alive, name), minlength=CUE_BINS[name][1])
            stopped = value[dead & np.isfinite(value)]
            for mirrored in (stopped, -stopped):
                counts[k][1] += 0.5 * np.bincount(bins(mirrored, name),
                                                  minlength=CUE_BINS[name][1])
    return [np.log(c / c.sum(axis=1, keepdims=True)) for c in counts]


def state_log(values, table, name, dead="fit"):
    """(n, 3) log-likelihoods of H, A, D for finite cue values, from a fit() table."""
    index = bins(values, name)
    n_bins = CUE_BINS[name][1]
    home, away = table[0][index], table[0][n_bins - 1 - index]
    other = table[1][index] if dead == "fit" else np.maximum(home, away)
    return np.column_stack((home, away, other))


@lru_cache(maxsize=None)
def idsse_log(exclude="", items=(), labels=None):
    """fit() on the clean IDSSE dev matches minus `exclude` (a Match name); never the hold-out.

    `items` is sorted (name, value) pairs overriding PARAMS, hashable for the cache.
    `labels` is the label variant to fit on (idsse.LABELS when None), part of the cache key.
    """
    from solution.sources import idsse
    if labels is None:
        return idsse_log(exclude, items, idsse.LABELS)
    ids = [m for m in idsse.DEV_IDS if "idsse_" + m != exclude]
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    return fit([idsse.load(m, labels=labels) for m in ids], dict(PARAMS, **dict(items)))
