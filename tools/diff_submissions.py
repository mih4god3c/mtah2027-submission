"""Where two result files disagree, as video times, for a person to check against the video.

    .venv/bin/python tools/diff_submissions.py evaluation/results/A.csv evaluation/results/B.csv
    make diff A=our_team B=v5_plus_tv_cue_merged_ball

A is the reference (what we submit), B the comparison file (another registered method, written
with `make submission METHOD=... TEAM=...`). The tool lists every run of frames where the labels
differ, with the tactical video time, the length, and both labels, plus the broadcast time and
what the TV was showing there when the alignment cache exists (cache/broadcast/BAR-ATH_align.npz,
written by `make broadcast-align`). Runs are listed longest first and, with --order time, in
match order. It reads the two label files and, optionally, that cache; nothing under data/.

What this is for (RULES.md): watching the video to judge which of two algorithmic outputs is
right at a moment is inspection, which is allowed. Writing what was seen into labels, frame
constants, or a variant tuned to the moments watched is hand annotation, which is not. The
outcome of a viewing can inform a decision between registered methods (docs/DECISIONS.md), and
should say so there before the file changes.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from solution.match import pm  # noqa: E402

NAME = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}


def clock(frame, fps):
    """Video time of a frame as mm:ss.s from the start of the recording."""
    seconds = frame / float(fps)
    return "{:02d}:{:04.1f}".format(int(seconds // 60), seconds % 60)


def diff_runs(a, b, halves):
    """[(start, stop, a_label, b_label)] maximal runs inside the halves where a != b and both
    labels are constant. Half-open frames; never across a half boundary."""
    out = []
    for start, stop in halves:
        wa, wb = a[start:stop], b[start:stop]
        differ = wa != wb
        change = np.ones(stop - start, dtype=bool)
        change[1:] = (wa[1:] != wa[:-1]) | (wb[1:] != wb[:-1]) | (differ[1:] != differ[:-1])
        bounds = np.flatnonzero(change).tolist() + [stop - start]
        for i, j in zip(bounds[:-1], bounds[1:]):
            if differ[i]:
                out.append((start + i, start + j, int(wa[i]), int(wb[i])))
    return out


def summary(a, b, halves, fps):
    """Aggregates per file and per label pair, as a list of lines."""
    lines = []
    for name, codes in (("A", a), ("B", b)):
        runs = [(s, t) for s, t in halves]
        spells = pm.possession_spells(codes, runs)
        dead = [(s, t) for s, t, k in pm.segments(codes, runs) if k == pm.D]
        graded = sum(t - s for s, t in halves)
        d_share = sum(int((codes[s:t] == pm.D).sum()) for s, t in halves) / float(graded)
        lines.append("  {}: {} possessions of mean {:.1f} s, {} stoppages of median {:.1f} s, "
                     "D share {:.3f}".format(
                         name, len(spells),
                         np.mean([t - s for s, t, _ in spells]) / fps if spells else 0.0,
                         len(dead), np.median([t - s for s, t in dead]) / fps if dead else 0.0,
                         d_share))
    counts = {}
    for start, stop in halves:
        for ka in (pm.H, pm.A, pm.D):
            for kb in (pm.H, pm.A, pm.D):
                if ka != kb:
                    n = int(((a[start:stop] == ka) & (b[start:stop] == kb)).sum())
                    counts[(ka, kb)] = counts.get((ka, kb), 0) + n
    total = sum(counts.values())
    graded = sum(t - s for s, t in halves)
    lines.append("  frames that differ: {} of {} graded ({:.1f}%)".format(
        total, graded, 100.0 * total / graded))
    for (ka, kb), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n:
            lines.append("    A {} -> B {}: {:6d} frames ({:.1f} s)".format(
                NAME[ka], NAME[kb], n, n / fps))
    return lines


def load_alignment(path):
    """(bro_frame, state, state_names) from the alignment cache, or None if it is not there."""
    if not path or not os.path.isfile(path):
        return None
    from solution import broadcast as bc
    z = np.load(path, allow_pickle=False)
    return z["bro_frame"], z["state"], bc.STATE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", help="reference result CSV (the submission)")
    parser.add_argument("b", help="comparison result CSV")
    parser.add_argument("--min-s", type=float, default=0.0,
                        help="skip differing runs shorter than this many seconds (default 0)")
    parser.add_argument("--order", choices=("length", "time"), default="length",
                        help="list runs longest first (default) or in match order")
    parser.add_argument("--limit", type=int, default=0, help="print at most this many runs (0 = all)")
    parser.add_argument("--align", default=None,
                        help="broadcast alignment cache (default: cache/broadcast/<game>_align.npz)")
    args = parser.parse_args()

    fps = GAMES[GAME]["fps"]
    halves = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    a = pm.label_codes(pm.read_label_csv(args.a))
    b = pm.label_codes(pm.read_label_csv(args.b))
    if len(a) != len(b):
        raise SystemExit("row counts differ: {} against {}".format(len(a), len(b)))
    for name, codes in (("A", a), ("B", b)):
        if (codes[np.concatenate([np.arange(s, t) for s, t in halves])] == pm.X).any():
            print("note: {} has X on graded frames; the scorer forward-fills them".format(name))

    align_path = args.align
    if align_path is None:
        from solution.sources.fused import ALIGN_PATH
        align_path = ALIGN_PATH
    alignment = load_alignment(align_path)

    print("A = {}\nB = {}".format(os.path.relpath(args.a, ROOT), os.path.relpath(args.b, ROOT)))
    for line in summary(a, b, halves, fps):
        print(line)

    runs = [r for r in diff_runs(a, b, halves) if (r[1] - r[0]) / fps >= args.min_s]
    lengths = np.array([r[1] - r[0] for r in runs], dtype=float) / fps
    print("\n{} differing runs{}: length p50 {:.1f} s, p90 {:.1f} s, max {:.1f} s".format(
        len(runs), " of at least {:g} s".format(args.min_s) if args.min_s else "",
        np.median(lengths) if len(lengths) else 0.0,
        np.percentile(lengths, 90) if len(lengths) else 0.0,
        lengths.max() if len(lengths) else 0.0))
    if args.order == "length":
        runs.sort(key=lambda r: -(r[1] - r[0]))
    if args.limit:
        runs = runs[:args.limit]

    header = "{:>4} {:>4}  {:>8} {:>8}  {:>7}  {:>5}  {:>3} {:>3}".format(
        "#", "half", "tactical", "to", "frames", "len_s", "A", "B")
    if alignment is not None:
        header += "  {:>9}  {}".format("broadcast", "TV at the start of the run")
    print("\n" + header)
    print("tactical = mm:ss.s into the tactical video; broadcast = mm:ss.s into the broadcast "
          "video at the run's first frame")
    for n, (start, stop, ka, kb) in enumerate(runs, start=1):
        half = next(i + 1 for i, (s, t) in enumerate(halves) if s <= start < t)
        line = "{:>4} {:>4}  {:>8} {:>8}  {:>7}  {:>5.1f}  {:>3} {:>3}".format(
            n, half, clock(start, fps), clock(stop, fps), "{}-{}".format(start, stop - 1),
            (stop - start) / fps, NAME[ka], NAME[kb])
        if alignment is not None:
            bro_frame, state, names = alignment
            bf = int(bro_frame[start])
            line += "  {:>9}  {}".format(clock(bf, fps) if bf >= 0 else "--",
                                         names.get(int(state[start]), "?"))
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
