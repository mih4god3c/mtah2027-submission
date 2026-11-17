"""What does the end of a half look like? Aggregates only, for an end-of-half rule (Step 0).

    .venv/bin/python tools/halfend_check.py --diagnose --bar-ath   # (needs data/) for a human to run
    .venv/bin/python tools/halfend_check.py --diagnose             # IDSSE dev, plain halves

The failure: BAR-ATH's tracking blocks, which are the graded halves, run past the half-time
whistle while the players leave, and the reference decodes possessions there. IDSSE tracking
stops at the whistle, so the proxy league has no such frames and cannot show the failure.

Before a rule is built, this prints the last `--tail` seconds of each half in `--bin`-second bins
counted back from the half's last frame, so a human who knows where the whistle is can say which
cues change there. Per bin: tracked players and those inside the pitch (median), collective
speed (median), the fastest player (p90), seconds inside sprint events, ball seen share, ball
speed where seen (median), the reference's H / A / D shares and the possessions it starts there.

Candidate cue, settings fixed from IDSSE before any BAR-ATH run (2026-11-06): a sprint event is
the fastest tracked player above SPRINT_MS for at least SPRINT_S. In fully live IDSSE play the gap
between sprint events is at most 43 s (p99 24 s; dev, v1 + D12, seeds 0-1), but some IDSSE halves
end in a sprint-free 41-89 s that is mostly dead ball, so a sprint-free tail is not enough alone.
The tool also prints, per half, where the last sprint event ends and where kick + 45:00 falls
(Law 7: the half cannot end earlier), as seconds before the half's last frame.
"""

import argparse
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import kickoff  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.simulate import add_ball_noise, degrade, mean_player_speed  # noqa: E402

STRESS = (0.05, 3.4, 0.26)          # D12: sd m, jumps per visible minute, smoothing s
REFERENCE = "v5_frame_model"      # what solution/run.py submits
SPRINT_MS = (5.0, 4.5)              # fastest player above this (4.5: BAR-ATH runs ~0.9x IDSSE)
SPRINT_S = 1.0                      # ... for at least this long
LAW7_S = 45 * 60.0                  # a half lasts at least this from its kick
MARGIN_M = 1.0                      # 'inside the pitch' allows this beyond the lines


def player_speeds(match, lag=2):
    """(T, S) speed of every player in m/s from positions over +-lag frames; NaN if untracked."""
    x, y = match.player_x, match.player_y
    dx = np.full(x.shape, np.nan, dtype=np.float32)
    dy = np.full(y.shape, np.nan, dtype=np.float32)
    dx[lag:-lag] = x[2 * lag:] - x[:-2 * lag]
    dy[lag:-lag] = y[2 * lag:] - y[:-2 * lag]
    return np.hypot(dx, dy) * match.fps / (2.0 * lag)


def ball_speed(match, lag=2):
    out = np.full(match.n_frames, np.nan)
    out[lag:-lag] = np.hypot(match.ball_x[2 * lag:] - match.ball_x[:-2 * lag],
                             match.ball_y[2 * lag:] - match.ball_y[:-2 * lag])
    return out * match.fps / (2.0 * lag)


def fastest(match):
    """Speed of the fastest tracked player per frame, m/s; NaN with nobody tracked."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(player_speeds(match), axis=1)


def sprint_mask(top, fps, threshold, seconds=SPRINT_S):
    """Frames inside sprint events: the fastest player above `threshold` for >= `seconds`."""
    fast = np.nan_to_num(top, nan=0.0) > threshold
    edges = np.diff(np.concatenate(([0], fast.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(top), dtype=bool)
    for a, b in zip(starts, stops):
        if b - a >= seconds * fps:
            out[a:b] = True
    return out


def inside_counts(match):
    """(tracked, inside the pitch within MARGIN_M) players per frame."""
    x, y = match.player_x, match.player_y
    seen = np.isfinite(x) & np.isfinite(y)
    length, width = match.pitch
    with np.errstate(invalid="ignore"):
        inside = (seen & (x >= -MARGIN_M) & (x <= length + MARGIN_M)
                  & (y >= -MARGIN_M) & (y <= width + MARGIN_M))
    return seen.sum(axis=1), inside.sum(axis=1)


def _median(values):
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def _p90(values):
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 90)) if len(values) else float("nan")


def tail_rows(match, codes, start, stop, tail_s, bin_s):
    """One row of aggregates per bin over the last tail_s of [start, stop), latest bin last."""
    fps = match.fps
    tracked, inside = inside_counts(match)
    speed, top, ball = mean_player_speed(match), fastest(match), ball_speed(match)
    sprint = sprint_mask(top, fps, SPRINT_MS[0])
    alive = np.isin(codes, (pm.H, pm.A))
    rows = []
    width = int(round(bin_s * fps))
    n_bins = int(np.ceil(tail_s / bin_s))
    for i in range(n_bins, 0, -1):
        b = stop - (i - 1) * width
        a = max(start, b - width)
        if a >= b:
            continue
        part = codes[a:b]
        begins = int(np.sum(alive[a:b] & ~alive[a - 1:b - 1])) if a > start else int(alive[a])
        seen = match.ball_seen[a:b]
        rows.append({
            "from_end_s": (stop - a) / float(fps),
            "tracked": _median(tracked[a:b].astype(float)),
            "inside": _median(inside[a:b].astype(float)),
            "speed": _median(speed[a:b]), "top_p90": _p90(top[a:b]),
            "sprint_s": float(np.sum(sprint[a:b])) / fps,
            "ball_seen": float(np.mean(seen)),
            "ball_ms": _median(ball[a:b][seen[:len(ball[a:b])]]),
            "H": float(np.mean(part == pm.H)), "A": float(np.mean(part == pm.A)),
            "D": float(np.mean(part == pm.D)), "starts": begins})
    return rows


def half_marks(match, start, stop, kick):
    """Seconds before the half's last frame: end of the last sprint event (each threshold),
    and kick + 45:00 (Law 7)."""
    fps = float(match.fps)
    top = fastest(match)
    out = {}
    for threshold in SPRINT_MS:
        events = np.flatnonzero(sprint_mask(top[start:stop], fps, threshold))
        out["last_sprint_{}".format(threshold)] = ((stop - start - (events[-1] + 1)) / fps
                                                   if len(events) else float("nan"))
    out["law7"] = (stop - (kick + LAW7_S * fps)) / fps
    return out


HEADER = ("  from end s | tracked inside | speed top_p90 sprint_s | ball_seen ball_ms | "
          "H     A     D    starts")


def print_rows(rows):
    print(HEADER)
    for r in rows:
        print("  {from_end_s:10.0f} | {tracked:7.1f} {inside:6.1f} | {speed:5.2f} {top_p90:7.2f} "
              "{sprint_s:8.1f} | {ball_seen:9.2f} {ball_ms:7.2f} | {H:.2f}  {A:.2f}  {D:.2f}  "
              "{starts:3d}".format(**r))


def print_marks(marks):
    print("  last sprint event ends {} s before the half's last frame; kick + 45:00 is {:.1f} s "
          "before it".format(" / ".join("{:.1f} (> {} m/s)".format(marks["last_sprint_" + str(t)],
                                                                   t) for t in SPRINT_MS),
                             marks["law7"]))


def kicks_of(match):
    """Each half's kick: the kickoff rule's frame, else the half's first frame (IDSSE)."""
    found = kickoff.detect(match)
    return [k if k is not None else start for k, (start, _) in zip(found, match.halves)]


def diagnose_bar_ath(tail_s, bin_s):
    from solution.methods import METHODS
    from solution.sources import atd
    match = atd.load()
    codes = METHODS[REFERENCE](match)
    print("BAR-ATH, last {:.0f} s of each half in {:.0f} s bins; reference {}".format(
        tail_s, bin_s, REFERENCE))
    for h, ((start, stop), kick) in enumerate(zip(match.halves, kicks_of(match))):
        print("half {} ({:.1f} min from its first frame, kick {:.1f} s after it)".format(
            h + 1, (stop - start) / match.fps / 60.0, (kick - start) / float(match.fps)))
        print_rows(tail_rows(match, codes, start, stop, tail_s, bin_s))
        print_marks(half_marks(match, start, stop, kick))


def diagnose_idsse(tail_s, bin_s, seeds):
    """The same table on IDSSE dev plain halves (tracking stops at the whistle), medians over
    halves x seeds, with the truth's shares in place of the reference's."""
    from solution.sources import idsse
    tables, marks = [], []
    for match_id in idsse.DEV_IDS:
        raw = idsse.load(match_id)
        for seed in seeds:
            match = add_ball_noise(degrade(raw, seed), *STRESS[:2], seed=seed,
                                   smooth_s=STRESS[2])
            for (start, stop), kick in zip(match.halves, kicks_of(match)):
                tables.append(tail_rows(match, match.labels, start, stop, tail_s, bin_s))
                marks.append(half_marks(match, start, stop, kick))
    print("IDSSE dev x seeds {}, v1 + D12, plain halves: medians over {} halves (H / A / D: the "
          "truth's, means; starts: total)".format(seeds, len(tables)))
    rows = []
    for i in range(len(tables[0])):
        row = {}
        for key in tables[0][i]:
            values = np.array([t[i][key] for t in tables], dtype=float)
            if key in ("H", "A", "D"):
                row[key] = float(np.mean(values))
            elif key == "starts":
                row[key] = int(values.sum())
            else:
                row[key] = _median(values)
        rows.append(row)
    print_rows(rows)
    print_marks({key: _median(np.array([m[key] for m in marks])) for key in marks[0]})
    print("  per half, last sprint event (> {} m/s) s before the end: {}".format(
        SPRINT_MS[0], " ".join("{:.0f}".format(m["last_sprint_" + str(SPRINT_MS[0])])
                               for m in marks)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diagnose", action="store_true", help="the end-of-half bins")
    parser.add_argument("--bar-ath", action="store_true",
                        help="(needs data/) BAR-ATH aggregates, for a human to run")
    parser.add_argument("--tail", type=float, default=180.0, metavar="S")
    parser.add_argument("--bin", type=float, default=10.0, metavar="S")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if not args.diagnose:
        parser.error("only --diagnose exists so far")
    if args.bar_ath:
        diagnose_bar_ath(args.tail, args.bin)
    else:
        diagnose_idsse(args.tail, args.bin, args.seeds)


if __name__ == "__main__":
    main()
