"""C2b dead-ball cues for the decoder (PLAN.md C2b, backlog E2).

`tools/deadball_check.py` (Step 0) shows where `touch_kickoff` loses on dead ball: stoppages
start late, a median 0.9 s and a p90 8.8 s; restarts come late; 11% of stoppages are missed;
a perfect dead ball is worth +0.089 spellF1 on the stressed proxy. On the frames it labels
alive inside true dead ball, players are still moving (standing share 0.7%), so speed cannot
see them. What does show there:
  - the ball is still (61% of those frames) or beyond a line (24%). Both are artifacts of the
    DFL feed, which freezes the ball at the restart spot: the stressed proxy has a still ball
    on 13% of graded frames, BAR-ATH on 2% (FINDINGS A2), and a ball beyond a line on 16% of
    visible frames against 4.2% (D7). Neither is read here;
  - a player stands beyond a line: the throw-in taker behind the touchline, the corner taker
    behind the goal line. Players are not frozen by the feed, and simulator v1 only drops and
    ghosts them, so the proxy shows this as IDSSE players do it.

The cue, per frame: some tracked player stands more than `beyond_m` beyond a touchline or a goal
line, moving slower than `slow_ms` over `speed_s`. It enters the decoder like collective speed
does: P(cue | dead) and P(cue | alive) are fitted per frame on the clean IDSSE dev labels,
leave-one-match-out (D3), and each block adds the log-likelihood of its share of cue frames to
the alive and the dead state, times `deadball_w` (decoder.DEFAULTS).

Linesmen also stand beyond the touchline. IDSSE positions do not list them as players; whether
BAR-ATH's ATD does is for `tools/deadball_check.py --bar-ath` to show (its cue rate must look
like the proxy's). numpy only, no randomness.
"""

from functools import lru_cache

import numpy as np

from solution.match import pm

PARAMS = {
    "cues": ("player",),  # which cues the decoder reads: "player", "ball" (see CUES)
    "beyond_m": 0.5,    # a player this far beyond a touchline or goal line ...
    "slow_ms": 2.0,     # ... moving slower than this ...
    "speed_s": 1.0,     # ... over this window (centred) ...
    "ball_m": 1.0,      # "ball": the seen ball this far beyond a line
    "still_ms": 0.5,    # "still": the seen ball slower than this over speed_s
}


def _frames(seconds, fps):
    return max(1, int(round(seconds * fps)))


def player_speed(match, window_s):
    """(T, S) speed of each tracked player in m/s over a centred window; NaN where unknown."""
    half = max(1, _frames(window_s, match.fps) // 2)
    x, y = match.player_x, match.player_y
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if len(x) > 2 * half:
        out[half:-half] = np.hypot(x[2 * half:] - x[:-2 * half],
                                   y[2 * half:] - y[:-2 * half]) * match.fps / (2.0 * half)
    return out


def beyond_line(match, params=PARAMS):
    """Per frame: a tracked player stands beyond a line (a throw-in or corner taker)."""
    length, width = match.pitch
    x, y, d = match.player_x, match.player_y, params["beyond_m"]
    speed = player_speed(match, params["speed_s"])
    with np.errstate(invalid="ignore"):
        out = (x < -d) | (x > length + d) | (y < -d) | (y > width + d)
        return np.any(out & (speed < params["slow_ms"]), axis=1)


def ball_beyond(match, params=PARAMS):
    """Per frame: the seen ball is more than `ball_m` beyond a line.

    The proxy overstates this (D7): score it also under simulate.thin_beyond.
    """
    length, width = match.pitch
    x, y, d = match.ball_x, match.ball_y, params["ball_m"]
    with np.errstate(invalid="ignore"):
        return (x < -d) | (x > length + d) | (y < -d) | (y > width + d)


def ball_still(match, params=PARAMS):
    """Per frame: the seen ball moves slower than `still_ms` over `speed_s` (centred).

    Not fair on the proxy: DFL freezes the dead ball, so the stressed proxy has a still ball on
    13% of graded frames, BAR-ATH on 2% (FINDINGS A2). Scored only as an upper bound.
    """
    half = max(1, _frames(params["speed_s"], match.fps) // 2)
    x, y = match.ball_x, match.ball_y
    speed = np.full(len(x), np.nan, dtype=np.float32)
    if len(x) > 2 * half:
        speed[half:-half] = np.hypot(x[2 * half:] - x[:-2 * half],
                                     y[2 * half:] - y[:-2 * half]) * match.fps / (2.0 * half)
    with np.errstate(invalid="ignore"):
        return speed < params["still_ms"]


CUES = {"player": beyond_line, "ball": ball_beyond, "still": ball_still}


def cues(match, params=PARAMS):
    """(T, K) bool: the cues in params["cues"], in that order."""
    return np.stack([CUES[name](match, params) for name in params["cues"]], axis=1)


def fit(matches, params=PARAMS):
    """(K, 2, 2) log P(cue k = c | state): rows alive, dead; columns cue absent, present."""
    counts = np.ones((len(params["cues"]), 2, 2))           # add-one smoothing
    for match in matches:
        found = cues(match, params)
        alive = (match.labels == pm.H) | (match.labels == pm.A)
        for row, mask in enumerate((alive, match.labels == pm.D)):
            for k in range(found.shape[1]):
                counts[k, row] += np.bincount(found[mask, k].astype(int), minlength=2)
    return np.log(counts / counts.sum(axis=2, keepdims=True))


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
