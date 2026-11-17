"""Which kinds of stoppage does a method miss? Dead ball by restart type, on the IDSSE dev matches.

    .venv/bin/python tools/restart_check.py [--method v7_keep2_control] [--seed 0]
    make restart-check [ARGS=...]

Step 0 for a restart grammar (goal kicks first; EXPERIMENTS section 31). No BAR-ATH data is read.

Every DFL restart event (idsse.restarts: kickoff, goal, throw_in, goal_kick, corner, free_kick,
penalty, referee_ball) is matched to the stoppage it ends: the run of D in the truth (keep2
relabels, D21) whose last frame is nearest the event, within MATCH_S (the feed's times are
manual, about a second off). Free kicks within OFFSIDE_S after a DFL Offside event are
reported as "offside_fk", the rest as "free_kick". D runs no restart ends (the end of a half,
or a missing event) are "unmatched".

Per type: how many stoppages, their median length, how much of their time the method labels H
or A (missed), how many it misses entirely or mostly (over half), and the type's share of all
missed dead-ball time. The matches are degraded as in the proxy league's decision run: simulator
v1 with D12's ball stress (noise 0.05 m, 3.4 jumps per minute, smoothing 0.26 s).
Aggregates only; numpy, deterministic for a seed.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.simulate import add_ball_noise, degrade  # noqa: E402
from solution.sources import idsse  # noqa: E402

MATCH_S = 3.0     # a restart ends the D run whose last frame is within this of it
OFFSIDE_S = 60.0  # a free kick this soon after an Offside event is an offside free kick
ORDER = ("goal_kick", "throw_in", "corner", "free_kick", "offside_fk", "kickoff", "goal",
         "penalty", "referee_ball", "unmatched")


def offside_frames(match_id):
    """Frames of DFL Offside events, on load()'s timeline."""
    return np.array([f for f, event in idsse._events(match_id)
                     if next(iter(event), None) is not None and next(iter(event)).tag == "Offside"],
                    dtype=np.int64)


def dead_runs(truth, halves):
    """[(start, stop)] D runs of the truth inside the halves."""
    out = []
    for start, stop, code in pm.segments(truth, halves):
        if code == pm.D:
            out.append((start, stop))
    return out


def classify(runs, events, offsides, fps):
    """One kind per D run: the restart nearest its end, within MATCH_S; else 'unmatched'."""
    kinds = ["unmatched"] * len(runs)
    ends = np.array([stop - 1 for _, stop in runs])
    taken = set()
    for frame, kind in events:
        if not len(ends):
            break
        i = int(np.argmin(np.abs(ends - frame)))
        if abs(ends[i] - frame) > MATCH_S * fps or i in taken:
            continue
        if kind == "free_kick" and len(offsides) and np.any(
                (offsides <= frame) & (offsides >= frame - OFFSIDE_S * fps)):
            kind = "offside_fk"
        kinds[i] = kind
        taken.add(i)
    return kinds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="v7_keep2_control", choices=sorted(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    method = METHODS[args.method]
    rows = []        # (kind, length_s, missed_s)
    total_d = total_missed = 0.0
    for match_id in idsse.DEV_IDS:
        clean = idsse.load(match_id, labels="keep2")
        fps = clean.fps
        match = add_ball_noise(degrade(clean, args.seed), 0.05, 3.4, args.seed, smooth_s=0.26)
        pred = method(match)
        truth = clean.labels
        runs = dead_runs(truth, clean.halves)
        kinds = classify(runs, idsse.restarts(match_id), offside_frames(match_id), fps)
        for (start, stop), kind in zip(runs, kinds):
            missed = np.isin(pred[start:stop], (pm.H, pm.A)).sum()
            rows.append((kind, (stop - start) / fps, missed / fps))
        dead = truth == pm.D
        total_d += dead.sum() / fps
        total_missed += (dead & np.isin(pred, (pm.H, pm.A))).sum() / fps

    print("{} on {} IDSSE dev matches, seed {}, keep2 truth, D12 ball stress".format(
        args.method, len(idsse.DEV_IDS), args.seed))
    print("dead ball {:.0f} s, of which {:.0f} s ({:.1f}%) labelled H or A\n".format(
        total_d, total_missed, 100 * total_missed / total_d))
    head = "{:<13} {:>6} {:>9} {:>9} {:>9} {:>10} {:>9} {:>10}"
    print(head.format("restart", "count", "median s", "dead s", "missed s", "missed %",
                      "mostly", "share of"))
    print(head.format("", "", "", "", "", "of its time", "missed", "all missed"))
    for kind in ORDER:
        pick = [r for r in rows if r[0] == kind]
        if not pick:
            continue
        length = np.array([r[1] for r in pick])
        missed = np.array([r[2] for r in pick])
        print(head.format(kind, len(pick), "{:.1f}".format(np.median(length)),
                          "{:.0f}".format(length.sum()), "{:.0f}".format(missed.sum()),
                          "{:.1f}".format(100 * missed.sum() / length.sum()),
                          "{}".format(int(np.sum(missed > 0.5 * length))),
                          "{:.1f}".format(100 * missed.sum() / total_missed)))
    print("\n'mostly missed' = stoppages with over half their time labelled H or A. Matching by "
          "the nearest restart event within {:g} s of the run's end; the event feed's times are "
          "manual.".format(MATCH_S))


if __name__ == "__main__":
    main()
