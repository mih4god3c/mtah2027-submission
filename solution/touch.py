"""C2a: touches and control from ball kinematics (PLAN.md C2a, backlog E3).

Proximity is not ownership: inside long possessions the nearest player flips team 10 to 24
times (notebooks/eda/FINDINGS.md), while ownership persists through ball flight and changes at
a controlled touch. This module turns the visible ball into last-touch evidence, for the
semi_markov decoder to read next to the nearest-player term:
  - touch candidates: frames where the ball's velocity changes sharply (speed, direction or
    both: the size of the change of the velocity vector) with a player within `touch_m`;
  - control episodes: the ball stays within `control_m` of a team's player, moving with them
    (relative speed under `control_rel_ms`), for at least `control_s`;
  - the holder: a team holds the ball from its control episode or its touch until the other
    team touches it. With `touch_takes` (the default) that touch hands the other team the ball
    at once. Without it, the touch only ends the hold and the other team takes over at its own
    control episode: last touch with hysteresis. A hold survives ball gaps up to `carry_gap_s`
    and ends `expire_s` after its team last touched the ball.

touch_takes is True because the hysteresis loses badly on the proxy league: 0.349 against
0.472 spellF1 under simulator v1 (docs/EXPERIMENTS.md section 5). DFL's possession flips at the
first touch by the other team. Whether Metrica's does too is PLAN.md open question 7; the
public window's 1.3 s Barcelona possession with no Barcelona player ever nearest the ball
(PLAN.md finding 4) suggests it does. See DECISIONS D10.
Each event is attributed to the team of the nearest player at that frame, as tracked: ghosts
and missing players are not corrected for.

The ball is cleaned first. Ball speed comes from positions only, never the ATD speed field
(FINDINGS point 2). A frame-to-frame step above `teleport_ms` is a tracking jump: the track is
cut there, and pieces shorter than `min_piece_s` are dropped, which removes the short hops
BAR-ATH has 216 of. Frames with the ball beyond a line are not used at all (DECISIONS D7: the
proxy league overstates that cue). Positions are smoothed with a centred moving average inside
each piece, then velocities are differences over `lag_frames`.

Thresholds are physical (metres, seconds, m/s), so they mean the same at any frame rate.
numpy only, no randomness.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

PARAMS = {
    "teleport_ms": 40.0,     # a frame-to-frame step faster than this is a tracking jump
    "min_piece_s": 0.24,     # ball pieces between jumps shorter than this are dropped
    "smooth_s": 0.2,         # centred moving average of ball positions, inside a piece
    # The BAR-ATH ball is smoothed by its tracker over about 0.26 s (sd), so a kick shows as a
    # velocity change spread over a few tenths of a second, a little away from the feet: the
    # lag, radius and window are set for that ball (D12, EXPERIMENTS section 6).
    "lag_s": 0.4,            # velocity before and after a frame, over this span
    "touch_dv": 3.0,         # m/s: change of the ball velocity vector that marks a touch
    "touch_m": 3.0,          # ... with a player at most this far from the ball
    "suppress_s": 0.4,       # at most one touch per this window: the sharpest
    "control_m": 1.5,        # control: ball within this of a team's player ...
    "control_rel_ms": 2.0,   # ... moving with them: ball minus player speed under this ...
    "control_s": 0.4,        # ... for at least this long
    "carry_gap_s": 1.0,      # a hold survives ball gaps up to this long
    "expire_s": 4.0,         # ... and ends this long after its team last touched the ball
    "touch_takes": True,     # a touch by the other team hands it the ball at once (D10)
}
NONE = -1


@dataclass
class Touches:
    touch_frame: np.ndarray        # (n,) int64 frames of touch candidates
    touch_team: np.ndarray         # (n,) int8 team of the nearest player there
    control: np.ndarray            # (m, 3) int64 rows of [start, stop) frames and team
    holder: np.ndarray             # (T,) int8 team holding the ball, NONE if unknown
    usable: np.ndarray             # (T,) bool frames whose ball survived cleaning


def _frames(seconds, fps, minimum=1):
    return max(minimum, int(round(seconds * fps)))


def _runs(mask):
    """(starts, stops) of the True runs of a 1-D boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def clean_ball(match, params=PARAMS):
    """(usable, piece_start, piece_stop): frames whose ball we trust, and their piece bounds.

    A piece is a run of usable frames with no jump inside it; piece_start/stop are -1 off it.
    """
    n, fps = match.n_frames, match.fps
    length, width = match.pitch
    x, y = match.ball_x.astype(np.float64), match.ball_y.astype(np.float64)
    usable = (match.ball_seen & match.in_play_mask() &
              (x >= 0) & (x <= length) & (y >= 0) & (y <= width))
    step = np.full(n, np.inf)
    with np.errstate(invalid="ignore"):
        step[1:] = np.hypot(np.diff(x), np.diff(y)) * fps
    # A piece starts wherever the previous frame is unusable or the step into this one jumps.
    joined = np.zeros(n, dtype=bool)
    joined[1:] = usable[1:] & usable[:-1] & (step[1:] <= params["teleport_ms"])
    starts = np.flatnonzero(usable & ~joined)
    stops = np.append(starts[1:], n)
    # A piece ends at its first unusable frame, or where the next piece starts.
    first_unusable = np.searchsorted(np.flatnonzero(~usable), starts)
    unusable_at = np.append(np.flatnonzero(~usable), n)
    stops = np.minimum(stops, unusable_at[first_unusable])
    keep = stops - starts >= _frames(params["min_piece_s"], fps)
    usable = np.zeros(n, dtype=bool)
    piece_start = np.full(n, -1, dtype=np.int64)
    piece_stop = np.full(n, -1, dtype=np.int64)
    for a, b in zip(starts[keep], stops[keep]):
        usable[a:b] = True
        piece_start[a:b] = a
        piece_stop[a:b] = b
    return usable, piece_start, piece_stop


def smoothed_ball(match, usable, piece_start, piece_stop, params=PARAMS):
    """Ball (x, y) averaged over a centred window inside each piece; NaN off the pieces."""
    half = _frames(params["smooth_s"], match.fps) // 2
    t = np.arange(match.n_frames)
    lo = np.where(usable, np.maximum(piece_start, t - half), 0)
    hi = np.where(usable, np.minimum(piece_stop, t + half + 1), 1)
    out = []
    for values in (match.ball_x, match.ball_y):
        filled = np.where(usable, values.astype(np.float64), 0.0)
        total = np.concatenate(([0.0], np.cumsum(filled)))
        out.append(np.where(usable, (total[hi] - total[lo]) / (hi - lo), np.nan))
    return out[0], out[1]


def _shift(values, k):
    """values[t + k] at t, NaN where that falls off the array."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if k > 0:
        out[:-k] = values[k:]
    elif k < 0:
        out[-k:] = values[:k]
    else:
        out[:] = values
    return out


def detect(match, params=PARAMS):
    """Touches for one Match (any source: degraded IDSSE or BAR-ATH alike)."""
    fps, n = match.fps, match.n_frames
    usable, piece_start, piece_stop = clean_ball(match, params)
    bx, by = smoothed_ball(match, usable, piece_start, piece_stop, params)
    lag = _frames(params["lag_s"], fps)
    t = np.arange(n)
    has_before = usable & (t - lag >= piece_start)
    has_after = usable & (t + lag < piece_stop)
    scale = fps / float(lag)
    vin_x, vin_y = (bx - _shift(bx, -lag)) * scale, (by - _shift(by, -lag)) * scale
    vout_x, vout_y = (_shift(bx, lag) - bx) * scale, (_shift(by, lag) - by) * scale

    # Nearest tracked player to the ball on every usable frame.
    px, py = match.player_x.astype(np.float64), match.player_y.astype(np.float64)
    distance = np.hypot(px - bx[:, None], py - by[:, None])
    distance = np.where(np.isfinite(distance), distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    near_m = distance[t, nearest]
    near_team = np.where(np.isfinite(near_m), match.team[nearest], NONE).astype(np.int8)

    # Touch candidates: a big change of the velocity vector, the sharpest in its window.
    dv = np.hypot(vout_x - vin_x, vout_y - vin_y)
    dv = np.where(has_before & has_after & np.isfinite(dv), dv, -np.inf)
    reach = _frames(params["suppress_s"], fps) // 2
    padded = np.concatenate((np.full(reach, -np.inf), dv, np.full(reach, -np.inf)))
    window = np.lib.stride_tricks.sliding_window_view(padded, 2 * reach + 1)
    # Ties go to the earliest frame: it must beat everything before it and match the rest.
    earlier = window[:, :reach].max(axis=1) if reach else np.full(n, -np.inf)
    later = window[:, reach + 1:].max(axis=1) if reach else np.full(n, -np.inf)
    peak = (dv > earlier) & (dv >= later)
    is_touch = peak & (dv >= params["touch_dv"]) & (near_m <= params["touch_m"])
    touch_frame = np.flatnonzero(is_touch)
    touch_team = near_team[touch_frame]

    # Control: near a player and moving with them, for long enough.
    vbx = (_shift(bx, lag) - _shift(bx, -lag)) * scale / 2.0
    vby = (_shift(by, lag) - _shift(by, -lag)) * scale / 2.0
    pick = nearest[:, None]
    pvx = (np.take_along_axis(_shift(px, lag), pick, 1) -
           np.take_along_axis(_shift(px, -lag), pick, 1))[:, 0] * scale / 2.0
    pvy = (np.take_along_axis(_shift(py, lag), pick, 1) -
           np.take_along_axis(_shift(py, -lag), pick, 1))[:, 0] * scale / 2.0
    with np.errstate(invalid="ignore"):
        relative = np.hypot(vbx - pvx, vby - pvy)
        close = usable & (near_m <= params["control_m"]) & (relative <= params["control_rel_ms"])
    control_team = np.where(close, near_team, NONE)
    minimum = _frames(params["control_s"], fps)
    control = []
    for team in (0, 1):
        starts, stops = _runs(control_team == team)
        for a, b in zip(starts, stops):
            if b - a >= minimum:
                control.append((a, b, team))
    control = np.array(sorted(control), dtype=np.int64).reshape(-1, 3)
    holder = sweep_holder(n, touch_frame, touch_team, control, usable, match.halves, fps,
                          params)
    return Touches(touch_frame.astype(np.int64), touch_team.astype(np.int8), control, holder,
                   usable)


def sweep_holder(n, touch_frame, touch_team, control, usable, halves, fps, params=PARAMS,
                 keep_touch=None, keep_control=None):
    """Per-frame holding team from the events, as the module docstring describes.

    `keep_touch` / `keep_control` (bool per touch / per control row; None keeps all) drop the
    events marked False before the sweep, as if never detected (#3 Step 0,
    tools/control_check.py). With both None the result is the engine's own holder.
    """
    carry = _frames(params["carry_gap_s"], fps)
    expire = _frames(params["expire_s"], fps)
    if keep_touch is None:
        keep_touch = np.ones(len(touch_frame), dtype=bool)
    if keep_control is None:
        keep_control = np.ones(len(control), dtype=bool)
    # (frame, order, kind, team, renew_until). At one frame: resets, then controls, then touches.
    events = [(int(f), 2, "touch", int(k), int(f))
              for f, k, keep in zip(touch_frame, touch_team, keep_touch) if k != NONE and keep]
    events += [(int(a), 1, "control", int(k), int(b))
               for (a, b, k), keep in zip(control, keep_control) if keep]
    for start, stop in halves:
        events.append((start, 0, "reset", NONE, start))
        events.append((stop, 0, "reset", NONE, stop))
        gap_starts, gap_stops = _runs(~usable[start:stop])
        for a, b in zip(gap_starts, gap_stops):
            if b - a > carry:
                events.append((start + a + carry, 0, "reset", NONE, 0))
    events.sort()

    holder = np.full(n, NONE, dtype=np.int8)
    current, since, renewed = NONE, 0, 0

    def paint(until):
        stop = min(until, renewed + expire)
        if current != NONE and stop > since:
            holder[since:stop] = current

    for frame, _, kind, team, renew in events:
        if current != NONE and frame > renewed + expire:
            paint(frame)
            current = NONE
        if kind == "reset":
            paint(frame)
            current = NONE
        elif kind == "control":
            if team != current:
                paint(frame)
                current, since = team, frame
            renewed = max(renewed, renew) if team == current else renew
        elif team == current:
            renewed = max(renewed, frame)
        elif current != NONE or params["touch_takes"]:
            paint(frame)
            current = team if params["touch_takes"] else NONE
            since, renewed = frame, frame
    paint(n)
    return holder


# -- the decoder's prior for the holder term ---------------------------------------------------

def holder_agree(matches, params=PARAMS):
    """P(holder is the team in possession | alive, a team holds), on clean labelled matches."""
    agree = known = 0
    for match in matches:
        holder = detect(match, params).holder
        alive = match.labels <= 1                       # scorer codes: H 0, A 1, D 2, X 3
        ok = alive & (holder != NONE)
        agree += int(np.sum(holder[ok] == match.labels[ok]))
        known += int(np.sum(ok))
    return agree / float(known)


@lru_cache(maxsize=None)
def idsse_holder_agree(exclude="", items=(), labels=None):
    """holder_agree on the clean IDSSE dev matches minus `exclude`, as decoder.idsse_priors.

    `items` is sorted (name, value) pairs overriding PARAMS, hashable for the cache.
    `labels` is the label variant to fit on (idsse.LABELS when None), part of the cache key.
    """
    from solution.sources import idsse
    if labels is None:
        return idsse_holder_agree(exclude, items, idsse.LABELS)
    ids = [m for m in idsse.DEV_IDS if "idsse_" + m != exclude]
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    return holder_agree([idsse.load(m, labels=labels) for m in ids], dict(PARAMS, **dict(items)))


# -- C5: boundary snapping (backlog E6) --------------------------------------------------------

def snap_transitions(states, touches, halves, fps, window_s=1.5, pick="first",
                     restarts=False):
    """Copy of per-frame decoder states with each change to a team moved onto that team's event.

    For every H <-> A change (and D -> team with `restarts`), look within +-window_s for a
    touch or a control start by the new team; move the change there: to the earliest such
    event (`pick` "first") or the closest ("nearest"). A move never crosses the neighbouring
    changes, so segments keep their order and none vanishes.
    """
    reach = int(round(window_s * fps))
    frames = np.concatenate((touches.touch_frame, touches.control[:, 0]))
    teams = np.concatenate((touches.touch_team, touches.control[:, 2]))
    order = np.argsort(frames, kind="stable")
    frames, teams = frames[order], teams[order]
    out = states.copy()
    for start, stop in halves:
        window = states[start:stop]
        changes = (np.flatnonzero(window[1:] != window[:-1]) + 1 + start).tolist()
        moved = list(changes)
        for i, f in enumerate(changes):
            before, after = int(states[f - 1]), int(states[f])
            if after > 1 or (before > 1 and not restarts):    # states: H 0, A 1, D 2
                continue
            low = moved[i - 1] if i else start
            high = changes[i + 1] if i + 1 < len(changes) else stop
            a, b = np.searchsorted(frames, [max(low + 1, f - reach), min(high, f + reach + 1)])
            mine = frames[a:b][teams[a:b] == after]
            if not len(mine):
                continue
            moved[i] = int(mine[0] if pick == "first" else mine[np.argmin(np.abs(mine - f))])
        bounds = [start] + moved + [stop]
        codes = [int(states[start])] + [int(states[f]) for f in changes]
        for a, b, code in zip(bounds[:-1], bounds[1:], codes):
            out[a:b] = code
    return out
