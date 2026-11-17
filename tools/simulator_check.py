"""Does the simulator make IDSSE look like the BAR-ATH ATD? Prints each target next to its value.

    .venv/bin/python tools/simulator_check.py                 # v0 and v1, dev matches, seeds 0 1 2
    .venv/bin/python tools/simulator_check.py --versions v1 --seeds 0
    make sim-check

The statistics are computed the way notebooks/eda/eda_aggregates.py computed them on BAR-ATH
(sections A2-A4), pooled over every degraded match and seed, inside the halves. The targets are
simulate.BAR_ATH, except the gap and visible-stretch lengths, whose target is the distribution
both simulator versions draw from (simulate.ball_profile). Two differences from the EDA: player
speed comes from positions here (the ATD has a speed field; their quantiles agree), and IDSSE
halves start at kickoff, where BAR-ATH's graded frames do not.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import simulate  # noqa: E402
from solution.simulate import BAR_ATH, DUPLICATE_M  # noqa: E402
from solution.sources import idsse  # noqa: E402


def runs(mask, halves):
    """[(start, stop)] runs of True inside each half."""
    out = []
    for start, stop in halves:
        window = np.concatenate(([False], mask[start:stop], [False])).astype(np.int8)
        edges = np.diff(window)
        out.extend(zip(start + np.flatnonzero(edges == 1), start + np.flatnonzero(edges == -1)))
    return out


def duplicate_frames(match, frames, chunk=4000):
    """Per frame in `frames`: does some team have two points closer than DUPLICATE_M?"""
    has = np.zeros(len(frames), dtype=bool)
    for begin in range(0, len(frames), chunk):
        part = frames[begin:begin + chunk]
        for code in (0, 1):
            x = match.player_x[part][:, match.team == code]
            y = match.player_y[part][:, match.team == code]
            distance = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
            upper = np.triu(np.ones(distance.shape[1:], dtype=bool), k=1)
            has[begin:begin + chunk] |= ((distance < DUPLICATE_M) & upper[None]).any(axis=(1, 2))
    return has


def measure(clean, degraded):
    """Raw per-match pieces; pooled by summarise(). Speed and zones come from the clean match."""
    play = clean.in_play_mask()
    fps = float(clean.fps)
    seen = degraded.ball_seen
    speed = simulate.mean_player_speed(clean)
    zone = simulate.zones(clean)
    gaps = runs(~seen & play, clean.halves)
    present = np.isfinite(degraded.player_x) & np.isfinite(degraded.player_y)
    frames = np.flatnonzero(play)
    lifetimes = [b - a for slot in range(present.shape[1])
                 for a, b in runs(present[:, slot], clean.halves)]
    return {
        "play": int(play.sum()),
        "seen": int((seen & play).sum()),
        "gap_s": np.array([b - a for a, b in gaps]) / fps,
        "visible_s": np.array([b - a for a, b in runs(seen & play, clean.halves)]) / fps,
        "speed_gap": speed[play & ~seen],
        "speed_seen": speed[play & seen],
        "gap_zone": np.array([zone[a - 1] if a > 0 and seen[a - 1] else -1 for a, b in gaps]),
        "seen_beyond": int(((zone == 3) & seen & play).sum()),
        "objects": present[play].sum(axis=1),
        "team_objects": [present[play][:, degraded.team == code].sum(axis=1) for code in (0, 1)],
        "tracklet_s": np.array(lifetimes) / fps,
        "duplicates": duplicate_frames(degraded, frames),
    }


def summarise(parts):
    cat = lambda key: np.concatenate([p[key] for p in parts])  # noqa: E731
    visible, missing, _ = simulate.ball_profile()
    visible, missing = visible / 25.0, missing / 25.0           # BAR-ATH is 25 fps
    play = sum(p["play"] for p in parts)
    seen = sum(p["seen"] for p in parts)
    gap_s, visible_s, zone = cat("gap_s"), cat("visible_s"), cat("gap_zone")
    objects, tracklets = cat("objects"), cat("tracklet_s")
    teams = [np.concatenate([p["team_objects"][c] for p in parts]) for c in (0, 1)]
    speed_gap, speed_seen = np.nanmean(cat("speed_gap")), np.nanmean(cat("speed_seen"))
    return [
        ("ball visible share", BAR_ATH["ball_visible"], seen / float(play)),
        # p10 sits on a cliff (11% of gaps are under 1 s, p25 is 6 s), so count short gaps.
        ("gaps under 1 s, share", np.mean(missing < 1), np.mean(gap_s < 1)),
        ("gap length p25 (s)", np.percentile(missing, 25), np.percentile(gap_s, 25)),
        ("gap length p50 (s)", np.percentile(missing, 50), np.percentile(gap_s, 50)),
        ("gap length p75 (s)", np.percentile(missing, 75), np.percentile(gap_s, 75)),
        ("gap length mean (s)", missing.mean(), gap_s.mean()),
        ("visible stretch p25 (s)", np.percentile(visible, 25), np.percentile(visible_s, 25)),
        ("visible stretch p50 (s)", np.percentile(visible, 50), np.percentile(visible_s, 50)),
        ("visible stretch p75 (s)", np.percentile(visible, 75), np.percentile(visible_s, 75)),
        ("visible stretch mean (s)", visible.mean(), visible_s.mean()),
        ("player speed in gaps (m/s)", BAR_ATH["speed_in_gaps"], speed_gap),
        ("player speed, ball seen (m/s)", BAR_ATH["speed_ball_visible"], speed_seen),
        ("speed ratio gaps / seen", BAR_ATH["speed_in_gaps"] / BAR_ATH["speed_ball_visible"],
         speed_gap / speed_seen),
        ("gaps after ball wide/beyond", BAR_ATH["gap_after_wide_or_beyond"],
         np.mean(np.isin(zone, (2, 3)))),
        ("seen frames, ball beyond line", BAR_ATH["visible_ball_beyond_line"],
         sum(p["seen_beyond"] for p in parts) / float(seen)),
        ("objects per frame, mean", BAR_ATH["objects_mean"], objects.mean()),
        ("objects per frame, p5", BAR_ATH["objects_p"][5], np.percentile(objects, 5)),
        ("objects per frame, p25", BAR_ATH["objects_p"][25], np.percentile(objects, 25)),
        ("objects per frame, p50", BAR_ATH["objects_p"][50], np.percentile(objects, 50)),
        ("team frames with > 11", np.mean(BAR_ATH["team_over_11"]),
         np.mean([np.mean(t > 11) for t in teams])),
        ("team frames with < 9", np.mean(BAR_ATH["team_under_9"]),
         np.mean([np.mean(t < 9) for t in teams])),
        ("tracklet life p10 (s)", BAR_ATH["tracklet_s"][10], np.percentile(tracklets, 10)),
        ("tracklet life p25 (s)", BAR_ATH["tracklet_s"][25], np.percentile(tracklets, 25)),
        ("tracklet life p50 (s)", BAR_ATH["tracklet_s"][50], np.percentile(tracklets, 50)),
        ("tracklet life p75 (s)", BAR_ATH["tracklet_s"][75], np.percentile(tracklets, 75)),
        ("tracklet life p90 (s)", BAR_ATH["tracklet_s"][90], np.percentile(tracklets, 90)),
        ("frames with a duplicate", BAR_ATH["duplicate_frames"], cat("duplicates").mean()),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--versions", nargs="+", default=list(simulate.VERSIONS),
                        choices=simulate.VERSIONS)
    parser.add_argument("--matches", nargs="+", default=list(idsse.DEV_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    started = time.time()
    matches = [idsse.load(match_id) for match_id in args.matches]
    columns = {}
    for version in args.versions:
        parts = [measure(m, simulate.degrade(m, seed, version))
                 for m in matches for seed in args.seeds]
        columns[version] = summarise(parts)
    names = [row[0] for row in columns[args.versions[0]]]
    targets = [row[1] for row in columns[args.versions[0]]]
    print("{} matches x {} seeds, pooled inside the halves".format(len(matches), len(args.seeds)))
    print("{:<32}{:>10}".format("statistic", "BAR-ATH") +
          "".join("{:>10}".format(v) for v in args.versions))
    for i, name in enumerate(names):
        print("{:<32}{:>10.3f}".format(name, targets[i]) +
              "".join("{:>10.3f}".format(columns[v][i][2]) for v in args.versions))
    print("({:.0f} s)".format(time.time() - started))


if __name__ == "__main__":
    main()
