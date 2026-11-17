"""How many short possessions ("flickers") a frame,label result file has.

    .venv/bin/python tools/flickers.py evaluation/results/our_team.csv
    .venv/bin/python tools/flickers.py evaluation/results/our_team.csv --thresholds 10 25 50 --all
    make flickers [FILE=our_team] [ARGS="--all"]

Segments are cut exactly as tools/segments.py cuts them (inside each half, an X takes the label
before it, as the scorer does for spellF1), so the counts match what spellF1 sees. A segment is
"under N" when it lasts fewer than N frames (25 fps: 25 frames = 1 s). Thresholds are
cumulative: a 12-frame possession counts under 20, 30, 40 and 50. Prints counts only.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from segments import FPS, to_segments  # noqa: E402

DEFAULT_THRESHOLDS = (20, 30, 40, 50)


def lengths_by_label(rows):
    """{label: [length in frames, ...]} for segment rows from segments.to_segments."""
    out = {}
    for r in rows:
        out.setdefault(r["label"], []).append(r["end_frame"] - r["start_frame"] + 1)
    return out


def flicker_table(lengths, thresholds):
    """Rows (threshold, count, share of segments, frames in them) for one list of lengths."""
    total = len(lengths)
    table = []
    for t in thresholds:
        short = [n for n in lengths if n < t]
        table.append((t, len(short), len(short) / total if total else 0.0, sum(short)))
    return table


def report(by_label, thresholds, labels):
    groups = [("H+A", by_label.get("H", []) + by_label.get("A", []))]
    groups += [(k, by_label.get(k, [])) for k in labels]
    lines = []
    for name, lengths in groups:
        lines.append("{}: {} segments, {} frames".format(name, len(lengths), sum(lengths)))
        for t, count, share, frames in flicker_table(lengths, thresholds):
            lines.append("  under {:>3d} frames ({:4.1f} s): {:5d}  ({:5.1%} of segments, "
                         "{} frames)".format(t, t / FPS, count, share, frames))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file", help="a frame,label CSV, e.g. evaluation/results/our_team.csv")
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(DEFAULT_THRESHOLDS),
                        help="frame counts (default 20 30 40 50)")
    parser.add_argument("--all", action="store_true", help="also report dead-ball (D) segments")
    parser.add_argument("--game", default="BAR-ATH")
    args = parser.parse_args(argv)

    frame = pm.read_label_csv(args.file)
    codes = pm.label_codes(frame["label"])
    keep = ("H", "A", "D") if args.all else ("H", "A")
    rows = to_segments(codes, games.half_video_blocks(args.game), keep=keep)
    print(os.path.relpath(os.path.abspath(args.file), ROOT))
    print(report(lengths_by_label(rows), sorted(args.thresholds), keep))


if __name__ == "__main__":
    main()
