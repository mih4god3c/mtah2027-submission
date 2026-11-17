"""Does the touch engine see BAR-ATH the way it sees the proxy league? Aggregates only.

    .venv/bin/python tools/touch_check.py              # IDSSE dev: clean, degraded v1, + ball noise
    .venv/bin/python tools/touch_check.py --bar-ath    # (needs data/) BAR-ATH, for a human to run
    .venv/bin/python tools/touch_check.py --diagnose [--bar-ath]   # why touches do or do not fire

Why: v2_touch (D10) gets its gain from solution/touch.py, which reads ball kinematics. The
simulator's ball is clean; BAR-ATH's has 216 teleports and unknown noise. If the engine fires
at a very different rate on BAR-ATH, or its holder covers much less of the match, the proxy
gain will not transfer. The ball-noise stress (simulate.add_ball_noise) is calibrated to the
EDA's ball-speed aggregates; this compares the engine's own outputs.

Everything printed is an aggregate over the graded frames of a whole match: ball speed
percentiles from consecutive positions (as the EDA computes them), jumps per visible minute,
the share of the ball the cleaning keeps, event rates per minute of clean ball, the share of
frames a team holds, holder changes and decoded possessions per 90 minutes. No frame-level
value is printed, so the BAR-ATH output is safe to paste back.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import touch  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.simulate import add_ball_noise, degrade  # noqa: E402

# (sd m, jumps per visible min, smoothing s). "BAR-ATH-like" is calibrated to --diagnose.
STRESS = {"BAR-ATH-like": (0.05, 3.4, 0.26), "unsmoothed": (0.05, 3.4, 0.0)}


def stats(match, predict=True):
    graded = match.in_play_mask()
    fps = float(match.fps)
    x, y = match.ball_x.astype(np.float64), match.ball_y.astype(np.float64)
    step = np.full(match.n_frames, np.nan)
    step[1:] = np.hypot(np.diff(x), np.diff(y)) * fps
    speeds = step[graded & np.isfinite(step)]
    seen = graded & match.ball_seen
    found = touch.detect(match)
    clean_min = found.usable.sum() / fps / 60.0
    graded_min = graded.sum() / fps / 60.0
    held = graded & (found.holder != touch.NONE)
    changes = 0
    for start, stop in match.halves:
        h = found.holder[start:stop]
        known = h[h != touch.NONE]
        changes += int(np.sum(known[1:] != known[:-1]))
    row = {
        "ball_seen": seen.sum() / float(graded.sum()),
        "speed_p": np.percentile(speeds, (50, 90, 99)),
        "jumps_per_min": np.sum(speeds > touch.PARAMS["teleport_ms"]) / (seen.sum() / fps / 60.0),
        "kept": found.usable.sum() / float(seen.sum()),
        "touch_per_min": len(found.touch_frame) / clean_min,
        "control_per_min": len(found.control) / clean_min,
        "control_s": float(np.median(found.control[:, 1] - found.control[:, 0]) / fps)
        if len(found.control) else 0.0,
        "held": held.sum() / float(graded.sum()),
        "held_home": float(np.mean(found.holder[held] == 0)) if held.any() else 0.0,
        "changes_per90": changes * 90.0 / graded_min,
    }
    if predict:
        for name in ("v1_decoder", "v2_touch"):
            codes = METHODS[name](match)
            spells = pm.possession_spells(codes, match.halves)
            row[name] = (len(spells) * 90.0 / graded_min,
                         float(np.median([b - a for a, b, _ in spells]) / fps),
                         float(np.mean(codes[graded] == pm.D)))
    if match.labels is not None:
        spells = pm.possession_spells(match.labels, match.halves)
        row["truth"] = (len(spells) * 90.0 / graded_min,
                        float(np.median([b - a for a, b, _ in spells]) / fps),
                        float(np.mean(match.labels[graded] == pm.D)))
    return row


DV_THRESHOLDS = (1.0, 2.0, 3.0, 5.0)
LAGS_S = (0.12, 0.24, 0.4)


def diagnose(match):
    """Why touch candidates fire or not: smoothness of the ball and distance to players.

    All on the cleaned ball (touch.clean_ball). Separates two explanations of a low touch
    rate: a ball smoothed over time (velocity changes spread out: small dv at short lags,
    strongly positive autocorrelation of frame accelerations) or velocity changes far from
    any tracked player (large nearest distance at dv peaks).
    """
    fps = match.fps
    usable, piece_start, piece_stop = touch.clean_ball(match)
    x, y = match.ball_x.astype(np.float64), match.ball_y.astype(np.float64)
    t = np.arange(match.n_frames)
    clean_min = usable.sum() / float(fps) / 60.0
    row = {}
    # Raw frame-to-frame steps inside pieces: repeated positions and acceleration memory.
    inside = usable[1:] & usable[:-1] & (piece_start[1:] == piece_start[:-1])
    vx, vy = np.diff(x) * fps, np.diff(y) * fps
    row["zero_steps"] = float(np.mean((np.abs(vx) + np.abs(vy))[inside] < 1e-6))
    both = inside[1:] & inside[:-1]
    ax, ay = np.diff(vx), np.diff(vy)
    a = np.where(both, ax, np.nan)
    ok = np.isfinite(a[1:]) & np.isfinite(a[:-1])
    row["acc_autocorr"] = float(np.corrcoef(a[1:][ok], a[:-1][ok])[0, 1])
    row["acc_p50_p90"] = np.percentile(np.hypot(ax, ay)[both], (50, 90)) / fps   # m/s per frame
    # Nearest tracked player to the ball on clean frames.
    px, py = match.player_x.astype(np.float64), match.player_y.astype(np.float64)
    distance = np.hypot(px - x[:, None], py - y[:, None])
    distance = np.where(np.isfinite(distance), distance, np.inf)
    nearest = distance.min(axis=1)
    near = nearest[usable & np.isfinite(nearest)]
    row["near_p"] = np.percentile(near, (10, 50, 90))
    row["near_within2"] = float(np.mean(near <= 2.0))
    for lag_s in LAGS_S:
        params = dict(touch.PARAMS, lag_s=lag_s)
        bx, by = touch.smoothed_ball(match, usable, piece_start, piece_stop, params)
        lag = max(1, int(round(lag_s * fps)))
        ready = usable & (t - lag >= piece_start) & (t + lag < piece_stop)
        dv = np.hypot(touch._shift(bx, lag) - 2 * bx + touch._shift(bx, -lag),
                      touch._shift(by, lag) - 2 * by + touch._shift(by, -lag)) * fps / lag
        dv = np.where(ready & np.isfinite(dv), dv, -np.inf)
        row["dv_p", lag_s] = np.percentile(dv[ready], (50, 90, 99))
        reach = max(1, int(round(touch.PARAMS["suppress_s"] * fps)) // 2)
        padded = np.concatenate((np.full(reach, -np.inf), dv, np.full(reach, -np.inf)))
        window = np.lib.stride_tricks.sliding_window_view(padded, 2 * reach + 1)
        peak = (dv > window[:, :reach].max(axis=1)) & (dv >= window[:, reach + 1:].max(axis=1))
        for threshold in DV_THRESHOLDS:
            row["peaks", lag_s, threshold] = np.sum(peak & (dv >= threshold)) / clean_min
        strong = peak & (dv >= touch.PARAMS["touch_dv"]) & np.isfinite(nearest)
        row["peak_near", lag_s] = (np.percentile(nearest[strong], (10, 50, 90))
                                   if strong.any() else np.full(3, np.nan))
        row["peak_within2", lag_s] = float(np.mean(nearest[strong] <= 2.0)) if strong.any() \
            else float("nan")
    return row


def report_diagnosis(title, rows):
    print("\n{} -- diagnosis  ({} match-seed{})".format(title, len(rows),
                                                      "" if len(rows) == 1 else "s"))

    def mean(key):
        return np.mean([r[key] for r in rows], axis=0)
    print("  raw ball steps: repeated positions {:.3f}; frame acceleration p50 / p90 {:.2f} / "
          "{:.2f} m/s per frame; lag-1 autocorrelation of acceleration {:+.2f}".format(
              mean("zero_steps"), *mean("acc_p50_p90"), mean("acc_autocorr")))
    print("  nearest tracked player to the clean ball p10 / p50 / p90 {:.2f} / {:.2f} / {:.2f} m;"
          " within 2 m {:.3f}".format(*mean("near_p"), mean("near_within2")))
    for lag_s in LAGS_S:
        print("  lag {:.2f} s: dv p50 / p90 / p99 {:.2f} / {:.2f} / {:.2f} m/s; peaks per clean "
              "minute at dv >= {}: {}".format(
                  lag_s, *mean(("dv_p", lag_s)), " / ".join("{:g}".format(v) for v in DV_THRESHOLDS),
                  " / ".join("{:.1f}".format(mean(("peaks", lag_s, v))) for v in DV_THRESHOLDS)))
        print("             at dv >= {:g}: nearest player p10 / p50 / p90 {:.2f} / {:.2f} / "
              "{:.2f} m, within 2 m {:.3f}".format(
                  touch.PARAMS["touch_dv"], *mean(("peak_near", lag_s)),
                  mean(("peak_within2", lag_s))))


def report(title, rows):
    print("\n{}  ({} match-seed{})".format(title, len(rows), "" if len(rows) == 1 else "s"))

    def mean(key):
        return np.mean([r[key] for r in rows], axis=0)
    print("  ball seen {:.3f} of graded frames; speed from positions p50 / p90 / p99 "
          "{:.2f} / {:.2f} / {:.2f} m/s; jumps > 40 m/s per visible minute {:.2f}".format(
              mean("ball_seen"), *mean("speed_p"), mean("jumps_per_min")))
    print("  cleaning keeps {:.3f} of the seen ball; per clean-ball minute: touches {:.1f}, "
          "control episodes {:.1f} (median {:.2f} s)".format(
              mean("kept"), mean("touch_per_min"), mean("control_per_min"), mean("control_s")))
    print("  a team holds {:.3f} of graded frames (home {:.3f} of those); holder changes per "
          "90 min {:.0f}".format(mean("held"), mean("held_home"), mean("changes_per90")))
    for name in ("v1_decoder", "v2_touch", "truth"):
        if name in rows[0]:
            per90, median_s, dead = mean(name)
            print("  {:<13} possessions per 90 min {:.0f}, median possession {:.1f} s, "
                  "dead share {:.3f}".format(name, per90, median_s, dead))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar-ath", action="store_true",
                        help="the competition match (needs data/; for a human to run)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--diagnose", action="store_true",
                        help="only the diagnosis of why touch candidates fire or not")
    args = parser.parse_args()

    if args.bar_ath:
        from solution.sources import atd
        if args.diagnose:
            report_diagnosis("BAR-ATH ATD", [diagnose(atd.load())])
            print("\nCompare with: python tools/touch_check.py --diagnose (IDSSE dev).")
            return
        report("BAR-ATH ATD (priors fitted on all five IDSSE dev matches)", [stats(atd.load())])
        print("\nCompare with: python tools/touch_check.py (IDSSE dev, degraded v1).")
        return
    if args.diagnose:
        from solution.sources import idsse
        matches = [idsse.load(m) for m in idsse.DEV_IDS]
        degraded = [degrade(m, s) for m in matches for s in args.seeds]
        report_diagnosis("IDSSE dev, clean", [diagnose(m) for m in matches])
        report_diagnosis("IDSSE dev, degraded v1", [diagnose(m) for m in degraded])
        sd, jumps, smooth = STRESS["BAR-ATH-like"]
        report_diagnosis("IDSSE dev, degraded v1 + BAR-ATH-like ball",
                         [diagnose(add_ball_noise(m, sd, jumps, i % len(args.seeds),
                                                  smooth_s=smooth))
                          for i, m in enumerate(degraded)])
        return

    from solution.sources import idsse
    matches = [idsse.load(m) for m in idsse.DEV_IDS]
    report("IDSSE dev, clean", [stats(m, predict=False) for m in matches])
    degraded = [degrade(m, s) for m in matches for s in args.seeds]
    report("IDSSE dev, degraded v1", [stats(m) for m in degraded])
    for label, (sd, jumps, smooth) in STRESS.items():
        report("IDSSE dev, degraded v1 + {} ball (sd {} m, {} jumps/min, smoothing {} s)".format(
            label, sd, jumps, smooth),
            [stats(add_ball_noise(m, sd, jumps, i % len(args.seeds), smooth_s=smooth))
             for i, m in enumerate(degraded)])


if __name__ == "__main__":
    main()
