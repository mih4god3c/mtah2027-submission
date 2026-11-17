"""What do SkillCorner's open-data possession labels look like? A Step 0 description, no decision.

    .venv/bin/python tools/skillcorner_check.py            # 10 A-League matches, phases of play
    .venv/bin/python tools/skillcorner_check.py --idsse    # + IDSSE dev rows (dfl, keep2) beside it
    .venv/bin/python tools/skillcorner_check.py --public   # + the public window (structure only)

Source: github.com/SkillCorner/opendata (MIT), pinned to commit REV. Only the small files are
fetched, into external_data/skillcorner/<rev>/ (gitignored): matches.json and, per match,
{id}_match.json and {id}_phases_of_play.csv. The tracking (.jsonl, Git LFS, about 90 MB each) and
the dynamic events are not needed for this question.

Labels, at SkillCorner's 10 fps: inside each match period (match.json `match_periods`, inclusive
frames) a frame is H or A when a phase of play covers it (`team_in_possession_id` against the
home team's id) and D otherwise: SkillCorner defines phases only while the ball is in play.
A phase covers [frame_start, frame_end): consecutive phases share their boundary frame (a
phase's frame_end is the next one's frame_start on 6,754 of 8,794 joins), so frame_end is
read as exclusive.
Outside the periods it is X, so the scorer's helpers (relabel.stats) treat each period as a run.

Printed, pooled over the matches, in the units of EXPERIMENTS section 4 (label_check.py's):
possessions per 90, share under 1 and 2 s, median and mean length, direct H<->A changes per
alive minute; then stoppages (D runs) per 90 and their median, and the D share. Also how the
phases join up (gaps and overlaps between consecutive phases, same-team neighbours), because the
reading of "no phase = dead ball" rests on it, and what keep2 (solution/relabel.py) changes.

Reading: PLAN L2. The question is whether SkillCorner's possessions look like controlled
possession (the public window: about 12% under 2 s, median 7.6 s) or like first touches (DFL:
30%, 3.7 s). Nothing here chooses a method or a setting; --public is structure only (D1, L2).
"""

import argparse
import csv
import json
import os
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import relabel  # noqa: E402
from solution.match import pm  # noqa: E402
from tools.label_check import PUBLIC_GT, print_rows  # noqa: E402

REV = "4340d274572876239c154c90bc507a9b3250a656"   # SkillCorner/opendata master, 2026-09-14
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/data/{}"
CACHE = os.path.join(ROOT, "external_data", "skillcorner", REV[:12])
FPS = 10


def fetch(path, cache=CACHE):
    """The file at data/<path> of the pinned commit, downloaded once into the cache."""
    local = os.path.join(cache, path)
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with urllib.request.urlopen(RAW.format(REV, path)) as response:
            body = response.read()
        if body.startswith(b"version https://git-lfs"):
            raise ValueError("{} is a Git LFS pointer, not data".format(path))
        with open(local + ".part", "wb") as handle:
            handle.write(body)
        os.replace(local + ".part", local)
    return local


def read_phases(path):
    with open(path, newline="") as handle:
        return [{"start": int(r["frame_start"]), "end": int(r["frame_end"]),
                 "team": int(r["team_in_possession_id"]), "period": int(r["period"])}
                for r in csv.DictReader(handle)]


def labels(meta, phases):
    """Scorer codes per 10 fps frame: X outside the periods, D inside, H/A where a phase is."""
    periods = [(p["start_frame"], p["end_frame"]) for p in meta["match_periods"]]
    home = meta["home_team"]["id"]
    out = np.full(max(b for _, b in periods) + 1, pm.X, dtype=np.int8)
    for a, b in periods:
        out[a:b + 1] = pm.D
    inside = np.zeros(len(out), dtype=bool)
    for a, b in periods:
        inside[a:b + 1] = True
    outside = 0
    for ph in phases:
        span = slice(max(ph["start"], 0), min(ph["end"], len(out)))
        outside += int((~inside[span]).sum())
        out[span] = np.where(inside[span], pm.H if ph["team"] == home else pm.A, out[span])
    return out, outside


def joins(phases):
    """How consecutive phases of one period meet: gap in frames, and same-team neighbours."""
    gaps, same_team_gaps = [], []
    ordered = sorted(phases, key=lambda p: (p["period"], p["start"]))
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev["period"] != nxt["period"]:
            continue
        gap = nxt["start"] - prev["end"]              # 0: they tile, < 0: overlap
        gaps.append(gap)
        if prev["team"] == nxt["team"]:
            same_team_gaps.append(gap)
    return np.array(gaps), np.array(same_team_gaps)


def stoppages(codes, fps):
    """Lengths (s) of the D runs inside the graded runs."""
    out = []
    for a, b, k in pm.segments(codes, pm.valid_runs(codes)):
        if k == pm.D:
            out.append((b - a) / fps)
    return np.array(out)


def dead_rows(rows):
    print("\n{:<28} {:>9} {:>7} {:>9} {:>8}".format(
        "labels", "stoppages", "per 90", "median s", "D share"))
    for name, parts, fps in rows:
        lengths = np.concatenate([stoppages(c, fps) for c in parts])
        graded = sum(int(np.sum(c != pm.X)) for c in parts) / fps
        dead = sum(int(np.sum(c == pm.D)) for c in parts) / fps
        print("{:<28} {:>9} {:>7.0f} {:>9.1f} {:>8.3f}".format(
            name, len(lengths), len(lengths) / graded * 5400.0,
            float(np.median(lengths)) if len(lengths) else float("nan"), dead / graded))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--idsse", action="store_true",
                        help="also print the IDSSE dev matches' dfl and keep2 rows")
    parser.add_argument("--public", action="store_true",
                        help="also describe the public window (structure only, never choose)")
    args = parser.parse_args()

    with open(fetch("matches.json")) as handle:
        ids = sorted(m["id"] for m in json.load(handle))
    codes, joined, same, outside, n_phases = [], [], [], 0, 0
    for mid in ids:
        with open(fetch("matches/{0}/{0}_match.json".format(mid))) as handle:
            meta = json.load(handle)
        phases = read_phases(fetch("matches/{0}/{0}_phases_of_play.csv".format(mid)))
        c, out = labels(meta, phases)
        g, s = joins(phases)
        codes.append(c)
        joined.append(g)
        same.append(s)
        outside += out
        n_phases += len(phases)
    gaps, same_gaps = np.concatenate(joined), np.concatenate(same)

    print("SkillCorner open data @ {}: {} matches, {} phases, 10 fps".format(
        REV[:12], len(ids), n_phases))
    print("phase joins within a period: {} | overlap {} | tiling {} | gap 1-9 frames {} | "
          "gap >= 1 s {} (median {:.1f} s)".format(
              len(gaps), int((gaps < 0).sum()), int((gaps == 0).sum()),
              int(((gaps > 0) & (gaps < FPS)).sum()), int((gaps >= FPS).sum()),
              float(np.median(gaps[gaps >= FPS])) / FPS if (gaps >= FPS).any() else float("nan")))
    print("same-team neighbours: {} | of them tiling or overlapping {}, with a gap {}".format(
        len(same_gaps), int((same_gaps <= 0).sum()), int((same_gaps > 0).sum())))
    print("phase frames outside the match periods: {}".format(outside))

    keep2 = [relabel.controlled(c, FPS) for c in codes]
    rows = [("skillcorner phases", relabel.summary([relabel.stats(c, FPS) for c in codes])),
            ("skillcorner -> keep2", relabel.summary([relabel.stats(c, FPS) for c in keep2]))]
    dead = [("skillcorner phases", codes, FPS)]
    if args.idsse:
        from solution.sources import idsse
        raw = [idsse.load(m, labels="dfl") for m in idsse.DEV_IDS]
        fps = raw[0].fps
        dfl = [m.labels for m in raw]
        k2 = [relabel.controlled(x, fps) for x in dfl]
        rows += [("idsse dev dfl", relabel.summary([relabel.stats(x, fps) for x in dfl])),
                 ("idsse dev keep2", relabel.summary([relabel.stats(x, fps) for x in k2]))]
        dead += [("idsse dev dfl", dfl, fps), ("idsse dev keep2", k2, fps)]
    if args.public:
        gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT)["label"].to_numpy())
        rows.append(("public window (structure)", relabel.summary([relabel.stats(gt, 25)])))
        dead.append(("public window (structure)", [gt], 25))
    print()
    print_rows(rows)
    dead_rows(dead)
    changed = np.mean([np.mean(c[c != pm.X] != k[k != pm.X]) for c, k in zip(codes, keep2)])
    print("\nframes keep2 changes on SkillCorner: {:.4f} of graded frames".format(changed))


if __name__ == "__main__":
    main()
