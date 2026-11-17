"""Is the ATD ball late against the players, and how blurred is it? Label-free, aggregates only.

    .venv/bin/python tools/ball_lag_check.py             # IDSSE dev: clean, and the D12 ball at
                                                         # smoothing 0 / 0.13 / 0.26 / 0.40 s and
                                                         # past-only 0.26 s (the reference rows)
    .venv/bin/python tools/ball_lag_check.py --bar-ath   # (needs data/) BAR-ATH, for a human to run

Why: the proxy's D12 ball is blurred with a CENTRED Gaussian of sd 0.26 s, fitted to BAR-ATH's
velocity-change percentiles (D12). Removing that blur is worth +0.046 on the proxy (#4 W0). Two
things were never measured: whether the BAR-ATH ball also trails the players (a tracker that
smooths with past frames only makes it late), and whether 0.26 s holds on a statistic it was
not fitted to. Both are read here from the tracking alone, the ball against the players.

1. Lag test. Dribble frames: the ball is seen, a tracked player is within DRIBBLE_M of it, the
   same player has been nearest for at least DRIBBLE_S, and the ball moves at DRIBBLE_MS or more
   (a player running with the ball, so a time offset shows as a distance). Frames whose window of
   +-MAX_TAU_S is not fully seen are dropped, so every shift uses the same frames. For each shift
   tau the median distance from ball(t + tau) to that player at t. tau > 0 at the minimum means
   the ball shows later what the player did: it trails. Selecting frames by closeness at tau = 0
   pulls the minimum towards 0, so read BAR-ATH against the reference rows, not against 0.
2. Kick shape. A kick: ball speed below KICK_LOW_MS, then above KICK_HIGH_MS within KICK_WITHIN_S
   and staying above it for KICK_HOLD_S (so a one-frame teleport is not a kick), inside one clean piece of ball (touch.clean_ball: teleports removed), with the ball seen from
   1 s before to KICK_AFTER_S after. Its speed rises from the slowest frame in the second before
   it passes KICK_HIGH_MS to the peak in the KICK_AFTER_S after that frame. The 10% point is the
   last rise through 10% before the half-peak, so jitter before the kick is not counted. Reported: the 10-90% rise time, and the early (10-50%) and late (50-90%)
   halves of it. A centred blur rises symmetrically (late / early about 1); a past-only filter
   jumps and then creeps (an exponential gives late / early about 2.7); a clean kick rises
   within a frame or two.
Speeds come from positions (central differences), never the ATD speed field (FINDINGS point 2).
Nothing frame-level is printed, so the BAR-ATH output is safe to paste back.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import touch  # noqa: E402
from solution.simulate import add_ball_noise, degrade  # noqa: E402

DRIBBLE_M = 2.0        # a player this close to the ball
DRIBBLE_S = 1.0        # ... has been the nearest one for this long
DRIBBLE_MS = 1.5       # ... and the ball moves at least this fast: a player running with it
MAX_TAU_S = 1.0        # shifts tried: -1 .. +1 s, one frame (0.04 s) apart
SHOW_TAU_S = (-0.4, -0.2, -0.08, 0.0, 0.08, 0.2, 0.4)
KICK_LOW_MS = 3.0      # a kick starts from a ball slower than this
KICK_HIGH_MS = 10.0    # ... and passes this speed
KICK_WITHIN_S = 1.0    # ... within this long
KICK_HOLD_S = 0.2      # ... and stays above it this long (a one-frame teleport is not a kick)
KICK_AFTER_S = 1.5     # the peak is looked for this long after the start
# The proxy reference rows: (label, smoothing s, past-only). Ball noise and jumps are D12's.
NOISE, JUMPS = 0.05, 3.4
VARIANTS = (("clean IDSSE", None, False), ("D12, smoothing 0", 0.0, False),
            ("D12, smoothing 0.13 s", 0.13, False), ("D12, smoothing 0.26 s", 0.26, False),
            ("D12, smoothing 0.40 s", 0.40, False), ("D12, past-only 0.26 s", 0.26, True))


def ball_speed(x, y, fps):
    """Per-frame speed from positions, central differences; NaN where a neighbour is missing."""
    speed = np.full(len(x), np.nan)
    speed[1:-1] = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]) * fps / 2.0
    return speed


def nearest_slot(match):
    """(slot, distance) of the tracked player nearest the ball per frame; -1, inf if none."""
    bx, by = match.ball_x.astype(np.float64), match.ball_y.astype(np.float64)
    slot = np.full(match.n_frames, -1, dtype=np.int64)
    distance = np.full(match.n_frames, np.inf)
    for start in range(0, match.n_frames, 20000):
        stop = min(match.n_frames, start + 20000)
        d = np.hypot(match.player_x[start:stop] - bx[start:stop, None],
                     match.player_y[start:stop] - by[start:stop, None])
        d = np.where(np.isfinite(d), d, np.inf)
        pick = np.argmin(d, axis=1)
        slot[start:stop] = pick
        distance[start:stop] = d[np.arange(stop - start), pick]
    slot[~np.isfinite(distance)] = -1
    return slot, distance


def dribble_frames(match, slot, distance, speed):
    """Frames that pass the dribble test, and their nearest slot."""
    fps = match.fps
    reach = int(round(MAX_TAU_S * fps))
    hold = int(round(DRIBBLE_S * fps))
    close = (distance <= DRIBBLE_M) & (slot >= 0)
    # Length of the current run of frames close to the same slot, ending at each frame.
    run = np.zeros(match.n_frames, dtype=np.int64)
    for t in range(match.n_frames):
        if close[t]:
            run[t] = run[t - 1] + 1 if t and close[t - 1] and slot[t - 1] == slot[t] else 1
    seen = match.ball_seen.astype(np.int64)
    window_seen = np.convolve(seen, np.ones(2 * reach + 1, dtype=np.int64), mode="same")
    inside = np.zeros(match.n_frames, dtype=bool)
    for start, stop in match.halves:
        inside[start + reach:stop - reach] = True
    with np.errstate(invalid="ignore"):
        fast = speed >= DRIBBLE_MS
    ok = (run >= hold) & fast & inside & (window_seen == 2 * reach + 1)
    frames = np.flatnonzero(ok)
    return frames, slot[frames]


def lag_curve(match, frames, slots):
    """(taus in s, (n_frames_used, n_taus) distances): ball(t + tau) to player slot at t."""
    fps = match.fps
    reach = int(round(MAX_TAU_S * fps))
    shifts = np.arange(-reach, reach + 1)
    px = match.player_x[frames, slots].astype(np.float64)
    py = match.player_y[frames, slots].astype(np.float64)
    bx = match.ball_x.astype(np.float64)[frames[:, None] + shifts[None, :]]
    by = match.ball_y.astype(np.float64)[frames[:, None] + shifts[None, :]]
    return shifts / float(fps), np.hypot(bx - px[:, None], by - py[:, None])


def crossing(times, values, level):
    """First time `values` reaches `level`, linearly interpolated; NaN if never."""
    above = np.flatnonzero(values >= level)
    if not len(above):
        return np.nan
    i = above[0]
    if i == 0:
        return times[0]
    v0, v1 = values[i - 1], values[i]
    return times[i - 1] + (level - v0) / (v1 - v0) * (times[i] - times[i - 1])


def last_below(times, values, level, before):
    """The last time before `before` at which `values` rises through `level`, interpolated.

    Measured backwards from the half-peak, so jitter that brushes `level` well before the kick
    does not count as the start of the rise."""
    below = np.flatnonzero((values < level) & (times < before))
    if not len(below):
        return np.nan
    i = below[-1]
    if i + 1 >= len(values):
        return np.nan
    v0, v1 = values[i], values[i + 1]
    return times[i] + (level - v0) / (v1 - v0) * (times[i + 1] - times[i])


def kick_shapes(match, speed):
    """[(rise 10-90 s, early 10-50 s, late 50-90 s)] per kick, on the cleaned ball."""
    fps = match.fps
    usable, piece_start, piece_stop = touch.clean_ball(match)
    within = int(round(KICK_WITHIN_S * fps))
    after = int(round(KICK_AFTER_S * fps))
    before = int(round(1.0 * fps))
    ok_speed = usable & np.isfinite(speed)
    out, t, n = [], before, match.n_frames - after - 1
    while t < n:
        if not (ok_speed[t] and speed[t] < KICK_LOW_MS):
            t += 1
            continue
        stop = min(t + within + 1, piece_stop[t])
        ahead = speed[t + 1:stop]
        high = np.flatnonzero(ok_speed[t + 1:stop] & (ahead > KICK_HIGH_MS))
        if not len(high):
            t += 1
            continue
        cross = t + 1 + high[0]
        hold = speed[cross:cross + int(round(KICK_HOLD_S * fps))]
        if not (ok_speed[cross:cross + len(hold)].all() and np.all(hold > KICK_HIGH_MS)):
            t = cross
            continue
        # The kick starts at the slowest frame in the second before the crossing, so the rise
        # is measured from rest and not from KICK_LOW_MS, which would cut its early half.
        first = max(cross - within, piece_start[cross] if piece_start[cross] >= 0 else cross)
        window = np.where(ok_speed[first:cross], speed[first:cross], np.inf)
        start = first + int(np.argmin(window))
        end = start + after
        whole = (piece_start[start] == piece_start[end] and piece_start[start] <= start - before
                 and ok_speed[start:end + 1].all() and match.ball_seen[start - before:end].all())
        if whole:
            seg = speed[start:end + 1]
            peak = int(np.argmax(seg))
            low, top = seg[0], seg[peak]
            times = np.arange(peak + 1) / float(fps)
            levels = [low + q * (top - low) for q in (0.1, 0.5, 0.9)]
            t50 = crossing(times, seg[:peak + 1], levels[1])
            t90 = crossing(times, seg[:peak + 1], levels[2])
            t10 = last_below(times, seg[:peak + 1], levels[0], t50)
            if np.isfinite([t10, t50, t90]).all():
                out.append((t90 - t10, t50 - t10, t90 - t50))
        t = cross + within                      # one kick per second at most
    return out


def measure(match):
    """Everything for one match: lag-curve distances and kick shapes, pooled later."""
    fps = match.fps
    x, y = match.ball_x.astype(np.float64), match.ball_y.astype(np.float64)
    speed = ball_speed(x, y, fps)
    slot, distance = nearest_slot(match)
    frames, slots = dribble_frames(match, slot, distance, speed)
    taus, dist = lag_curve(match, frames, slots)
    seen_min = float(np.sum(match.ball_seen & match.in_play_mask())) / fps / 60.0
    return {"taus": taus, "dist": dist, "kicks": kick_shapes(match, speed),
            "seen_min": seen_min}


def lag_summary(rows):
    """(taus, pooled median distance per tau, index of the best tau, sub-frame best tau)."""
    taus = rows[0]["taus"]
    dist = np.vstack([r["dist"] for r in rows if len(r["dist"])])
    median = np.median(dist, axis=0)
    best = int(np.argmin(median))
    # Sub-frame minimum from a parabola through the best point and its neighbours.
    refined = float(taus[best])
    if 0 < best < len(taus) - 1:
        a, b, c = median[best - 1], median[best], median[best + 1]
        denom = a - 2 * b + c
        if denom > 0:
            refined = float(taus[best] + 0.5 * (a - c) / denom * (taus[1] - taus[0]))
    return taus, median, best, refined


def kick_summary(rows):
    """(n kicks, median rise 10-90 s, median early, median late, median late / early)."""
    kicks = np.array([k for r in rows for k in r["kicks"]]).reshape(-1, 3)
    if not len(kicks):
        return 0, np.nan, np.nan, np.nan, np.nan
    ratio = np.median(kicks[:, 2] / np.maximum(kicks[:, 1], 1e-3))
    return (len(kicks), float(np.median(kicks[:, 0])), float(np.median(kicks[:, 1])),
            float(np.median(kicks[:, 2])), float(ratio))


def report(title, rows):
    """Print the pooled aggregates of several measure() rows."""
    taus, median, best, refined = lag_summary(rows)
    dist = np.vstack([r["dist"] for r in rows if len(r["dist"])])
    seen_min = sum(r["seen_min"] for r in rows)
    kicks = np.array([k for r in rows for k in r["kicks"]]).reshape(-1, 3)
    zero = int(np.argmin(np.abs(taus)))
    shown = [int(np.argmin(np.abs(taus - s))) for s in SHOW_TAU_S]
    print("\n== {} ==".format(title))
    print("lag test: {} dribble frames ({:.1f} per seen minute)".format(
        dist.shape[0], dist.shape[0] / seen_min))
    print("  best tau {:+.2f} s (parabola {:+.3f} s); median distance {:.2f} m at tau 0, "
          "{:.2f} m at best".format(taus[best], refined, median[zero], median[best]))
    print("  median distance by tau: " + "  ".join(
        "{:+.2f}s {:.2f}".format(taus[i], median[i]) for i in shown))
    if len(kicks):
        _, rise, early, late, ratio = kick_summary(rows)
        p10, p90 = np.percentile(kicks[:, 0], (10, 90))
        print("kick shape: {} kicks ({:.2f} per seen minute)".format(
            len(kicks), len(kicks) / seen_min))
        print("  rise 10-90% median {:.3f} s (p10 {:.3f}, p90 {:.3f}); early 10-50% {:.3f} s, "
              "late 50-90% {:.3f} s; late/early median {:.2f}".format(
                  rise, p10, p90, early, late, ratio))
    else:
        print("kick shape: no kicks found")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar-ath", action="store_true",
                        help="the competition match (needs data/; for a human to run)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    print("ball lag check: dribble = player within {} m, nearest for >= {} s, ball >= {} m/s; "
          "kick = speed < {} then > {} m/s within {} s, held {} s".format(
              DRIBBLE_M, DRIBBLE_S, DRIBBLE_MS, KICK_LOW_MS, KICK_HIGH_MS, KICK_WITHIN_S,
              KICK_HOLD_S))
    if args.bar_ath:
        from solution.sources import atd
        report("BAR-ATH ATD", [measure(atd.load())])
        print("\nCompare with: .venv/bin/python tools/ball_lag_check.py (IDSSE dev reference "
              "rows).")
        return
    from solution.sources import idsse
    matches = [idsse.load(m) for m in idsse.DEV_IDS]
    for label, smooth, causal in VARIANTS:
        if smooth is None:
            rows = [measure(m) for m in matches]
        else:
            rows = [measure(add_ball_noise(degrade(m, seed), NOISE, JUMPS, seed, smooth_s=smooth,
                                           causal=causal))
                    for m in matches for seed in args.seeds]
        report("IDSSE dev, {}{}".format(label, "" if smooth is None else ", seeds {}".format(
            args.seeds)), rows)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
