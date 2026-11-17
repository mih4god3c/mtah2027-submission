"""A frame,label result file as segments: one row per possession (and, with --all, per stoppage).

    .venv/bin/python tools/segments.py evaluation/results/our_team.csv
    make segments [FILE=our_team] [ARGS="--all"]

Writes evaluation/reports/<name>_segments.csv (gitignored: a whole-match label set) with
    segment, half, label (H / A / D), start_frame, end_frame (inclusive, video frames),
    start_s, end_s (video seconds; end_s is the end of the last frame, so end_s - start_s is the
    duration), duration_s, start_time, end_time (video clock, H:MM:SS.ss).

Segments are cut inside each half (games.half_video_blocks), never across halftime, and an X
inside a half takes the label before it, as the scorer does for spellF1
(possession_metrics.fill_abstentions). Times are on the tactical video's own timeline (25 fps,
DATA.md), not the match clock. Prints counts only, never rows.
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25.0
LETTER = {pm.H: "H", pm.A: "A", pm.D: "D"}
COLUMNS = ["segment", "half", "label", "start_frame", "end_frame", "start_s", "end_s",
           "duration_s", "start_time", "end_time"]


def clock(seconds):
    """H:MM:SS.ss"""
    h, rest = divmod(seconds, 3600.0)
    m, s = divmod(rest, 60.0)
    return "{:d}:{:02d}:{:05.2f}".format(int(h), int(m), s)


def to_segments(codes, blocks, fps=FPS, keep=("H", "A")):
    """Rows (dicts) for the segments of `codes` inside `blocks` [(half, first, last)]."""
    runs = [(first, last + 1) for _, first, last in blocks]
    halves = {first: half for half, first, _ in blocks}
    filled = pm.fill_abstentions(np.asarray(codes), runs)
    rows = []
    for (start, stop), (half, first, last) in zip(runs, blocks):
        for s, e, code in pm.segments(filled, [(start, stop)]):
            label = LETTER.get(code, "X")
            if label not in keep:
                continue
            rows.append({"segment": len(rows) + 1, "half": halves[first], "label": label,
                         "start_frame": s, "end_frame": e - 1,
                         "start_s": round(s / fps, 2), "end_s": round(e / fps, 2),
                         "duration_s": round((e - s) / fps, 2),
                         "start_time": clock(s / fps), "end_time": clock(e / fps)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file", help="a frame,label CSV, e.g. evaluation/results/our_team.csv")
    parser.add_argument("--all", action="store_true", help="also write dead-ball (D) segments")
    parser.add_argument("--out", help="output path (default evaluation/reports/<name>_segments.csv)")
    parser.add_argument("--game", default="BAR-ATH")
    args = parser.parse_args(argv)

    frame = pm.read_label_csv(args.file)
    codes = pm.label_codes(frame["label"])
    rows = to_segments(codes, games.half_video_blocks(args.game),
                       keep=("H", "A", "D") if args.all else ("H", "A"))
    name = os.path.splitext(os.path.basename(args.file))[0]
    out = args.out or os.path.join(ROOT, "evaluation", "reports", name + "_segments.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {k: sum(r["label"] == k for r in rows) for k in ("H", "A", "D")}
    print("wrote {} ({} segments: H {}, A {}, D {})".format(
        os.path.relpath(out, ROOT), len(rows), counts["H"], counts["A"], counts["D"]))


if __name__ == "__main__":
    main()
