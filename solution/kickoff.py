"""C4 grammar, first rule: a half is dead ball until its kickoff (PLAN.md finding 9, C2b, C4).

The BAR-ATH second half is graded from about two minutes before its kickoff. Those frames are
all D, yet the ball is tracked on most of them and players stand near it, so a decoder can read
a long possession there. The kickoff has to be found from the tracking, never from a frame
number (RULES.md). The rule runs at the start of every half.

Cues from Law 8 (the start of play), read from the players only:
  - both teams stand in their own halves: each team has at least `min_players` tracked, they sit
    on opposite sides of the halfway line, and at least `own_share` of each team are in their
    own half or within `line_m` beyond the line (players line up on it, and IDSSE shows 0.64 to
    0.91 strictly inside at the kick). Taken over a `smooth_s` majority, it must hold for at
    least `hold_s`; the kick ends the hold, as players cross the line once the ball is played;
  - the players are not playing: median collective speed under `quiet_ms` over the `speed_s`
    before the kick (open play has a median of 2.33 m/s on IDSSE, dead ball 1.02);
  - play speeds up: collective speed over the `speed_s` after the kick exceeds the `speed_s`
    before by at least `rise_ms`;
  - nothing was played before it: in the half up to the kick, collective speed is above
    `play_ms` on at most `play_share` of frames. Open play is above 2.5 m/s on about half its
    frames (IDSSE), so a kickoff after an early goal, with the real one missed, is not taken.
The kick is placed inside the hold, on the frame after the last one whose speed (over
`smooth_s`) is still within `rise_ms` / 2 of the median over the hold's first half: where play
starts to speed up, however long the formation outlasts it. That onset trails the kick, since
only a few players move at once (IDSSE: collective speed is 1.4 to 2.5 m/s in the first 2 s
after a kick, 2.5 to 3.9 m/s in the next 8), so the kick goes `back_s` before it, but not before
the hold starts. If no frame is that calm, the hold's first frame. Placing it early only hands a few dead frames back to the decoder; placing it late
would cut into the first possession.
The kickoff is the end of the first hold in the half's first `search_s` that passes every cue.
If none does, the half is left to the decoder as before: a missed kickoff costs what it cost
until now, while a false one late in a half would wipe out real possessions.

The ball is not read. The rule began with the ball resting on the centre mark, and that cue
works on the synthetic pre-kickoff, but the BAR-ATH tracker never reports a ball there: in the
second half's first 300 s, with the ball seen on 88% of frames, not one sighting is within 2 m
of the mark, and during the pre-kickoff formation it sits a median 24 m away
(`tools/kickoff_check.py --diagnose --bar-ath`, EXPERIMENTS section 7).

Collective speed is simulate.mean_player_speed, in m/s. numpy only, no randomness.
"""

import warnings

import numpy as np

from solution.match import AWAY, HOME
from solution.simulate import mean_player_speed

PARAMS = {
    "search_s": 300.0,   # the kickoff is looked for in this much of the start of each half
    "line_m": 2.0,       # a player counts in his own half up to this beyond the halfway line
    "own_share": 0.8,    # share of each team's tracked players in their own half
    "min_players": 6,    # tracked players per team for the formation to count
    "smooth_s": 1.0,     # majority window for the formation, mean window for the speed onset
    "hold_s": 8.0,       # the formation holds at least this long before the kick
    "back_s": 2.0,       # the kick is placed this long before play starts to speed up
    "speed_s": 5.0,      # windows for collective speed before and after the kick
    "quiet_ms": 2.0,     # median collective speed before the kick under this
    "rise_ms": 0.5,      # mean collective speed rises by at least this after the kick
    "play_ms": 2.5,      # 'played' frames: collective speed above this ...
    "play_share": 0.05,  # ... on at most this share of the half before the kick
}
CHECKS = ("quiet", "rise", "before")


def _frames(seconds, fps):
    return max(1, int(round(seconds * fps)))


def formation(match, params=PARAMS):
    """Per frame: both teams in their own halves, as far as they are tracked (Law 8)."""
    x = match.player_x.astype(np.float64) - match.pitch[0] / 2.0
    ok, sides = np.ones(match.n_frames, dtype=bool), []
    for team in (HOME, AWAY):
        xs = x[:, match.team == team]
        visible = np.isfinite(xs)
        count = visible.sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # frames with nobody
            side = np.sign(np.nanmedian(xs, axis=1)) if xs.shape[1] else np.zeros(len(x))
        side = np.where(np.isfinite(side), side, 0.0)
        with np.errstate(invalid="ignore"):
            inside = visible & (xs * side[:, None] >= -params["line_m"])
        share = inside.sum(axis=1) / np.maximum(count, 1)
        ok &= (count >= params["min_players"]) & (share >= params["own_share"]) & (side != 0)
        sides.append(side)
    return ok & (sides[0] == -sides[1])


def ball_to_spot(match):
    """Distance of the ball from the centre mark, metres; NaN where the ball is not seen.

    Not a cue (see the module docstring); tools/kickoff_check.py reports it.
    """
    return np.hypot(match.ball_x.astype(np.float64) - match.pitch[0] / 2.0,
                    match.ball_y.astype(np.float64) - match.pitch[1] / 2.0)


def _moving_mean(values, frames):
    """Centred moving mean that skips NaN; NaN where the window has nothing."""
    finite = np.isfinite(values)
    kernel = np.ones(frames)
    total = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    count = np.convolve(finite.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def holds(start, stop, form, fps, params=PARAMS):
    """(hold start, hold end) frame pairs of the formation in a half's search window."""
    end = min(stop, start + _frames(params["search_s"], fps))
    held = _moving_mean(form[start:end].astype(float), _frames(params["smooth_s"], fps)) >= 0.5
    edges = np.diff(np.concatenate(([0], held.astype(np.int8), [0])))
    a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = (b - a >= _frames(params["hold_s"], fps)) & (b < end - start)   # it must break
    return [(start + int(i), start + int(j)) for i, j in zip(a[keep], b[keep])]


def kick_frame(a, b, speed, fps, params=PARAMS):
    """Where play speeds up in the hold [a, b): after its last frame as calm as its first half."""
    smooth = _moving_mean(speed[a:b], _frames(params["smooth_s"], fps))
    calm = smooth[:max(1, (b - a) // 2)]
    if not np.isfinite(calm).any():
        return a
    with np.errstate(invalid="ignore"):
        still = np.flatnonzero(smooth <= np.nanmedian(calm) + params["rise_ms"] / 2.0)
    if not len(still):
        return a
    return max(a, a + int(still[-1]) + 1 - _frames(params["back_s"], fps))


def _nan_stat(fn, values):
    values = values[np.isfinite(values)]
    return fn(values) if len(values) else np.nan


def checks(k, start, stop, speed, fps, params=PARAMS):
    """{cue: passed} for a kick at frame k of the half [start, stop)."""
    speed_f = _frames(params["speed_s"], fps)
    pre = speed[max(start, k - speed_f):k]
    post = speed[k:min(stop, k + speed_f)]
    played = speed[start:k][np.isfinite(speed[start:k])]
    return {
        "quiet": bool(_nan_stat(np.median, pre) <= params["quiet_ms"]),
        "rise": bool(_nan_stat(np.mean, post) - _nan_stat(np.mean, pre) >= params["rise_ms"]),
        "before": bool(len(played) == 0 or
                       np.mean(played > params["play_ms"]) <= params["play_share"]),
    }


def detect(match, params=PARAMS):
    """The kickoff frame of each half, or None where no formation hold passes every cue.

    A list in the order of match.halves; frames before a found kickoff are dead ball.
    """
    return [kick for _, kick in explain(match, params)]


def explain(match, params=PARAMS):
    """Per half: ({cue: candidates passing it}, kick frame or None). Counts only, for tools."""
    form = formation(match, params)
    speed = mean_player_speed(match)
    out = []
    for start, stop in match.halves:
        counts, kick = dict.fromkeys(("candidates",) + CHECKS, 0), None
        for a, b in holds(start, stop, form, match.fps, params):
            k = kick_frame(a, b, speed, match.fps, params)
            result = checks(k, start, stop, speed, match.fps, params)
            counts["candidates"] += 1
            for key, passed in result.items():
                counts[key] += passed
            if all(result.values()):
                kick = k
                break
        out.append((counts, kick))
    return out
