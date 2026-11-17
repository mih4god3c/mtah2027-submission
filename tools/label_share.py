"""Share of each label (H / A / D / X) in a frame,label result file, per half and for the match.

    .venv/bin/python tools/label_share.py evaluation/results/our_team.csv
    make label-share [FILE=our_team]

Counts only the frames inside the two halves (games.half_video_blocks); halftime and the video
before kickoff and after the final whistle are left out. "D share" is D frames over all frames in
the halves; "D of labelled" leaves out X. Prints aggregates only, never rows.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25.0


def shares(codes):
    """{label: (frames, share)} over `codes`, plus D over the non-X frames."""
    n = len(codes)
    out = {name: (int(np.sum(codes == i)), np.sum(codes == i) / n if n else 0.0)
           for i, name in enumerate(pm.LABELS)}
    labelled = n - out["X"][0]
    out["D_of_labelled"] = out["D"][0] / labelled if labelled else 0.0
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file", help="a frame,label CSV, e.g. evaluation/results/our_team.csv")
    parser.add_argument("--game", default="BAR-ATH")
    args = parser.parse_args(argv)

    frame = pm.read_label_csv(args.file)
    codes = pm.label_codes(frame["label"])
    frames = frame["frame"].values
    parts = []
    for half, first, last in games.half_video_blocks(args.game):
        parts.append(("half {}".format(half), codes[(frames >= first) & (frames <= last)]))
    parts.append(("match", np.concatenate([c for _, c in parts])))

    print(os.path.relpath(args.file, ROOT))
    head = "{:<8} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>9}"
    print(head.format("", "frames", "minutes", "H %", "A %", "D %", "X %", "D/labelled"))
    for name, part in parts:
        s = shares(part)
        print(head.format(name, len(part), "{:.1f}".format(len(part) / FPS / 60),
                          *["{:.1f}".format(100 * s[k][1]) for k in pm.LABELS],
                          "{:.1f}".format(100 * s["D_of_labelled"])))
    s = shares(parts[-1][1])
    print("\nD over the match: {} frames = {:.1f} min".format(s["D"][0], s["D"][0] / FPS / 60))


if __name__ == "__main__":
    main()
