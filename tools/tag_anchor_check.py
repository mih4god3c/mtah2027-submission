"""E8 Step 0: where do the Smart Tagging SET PIECES windows sit against the restart? Label-free.

    .venv/bin/python tools/tag_anchor_check.py --bar-ath                 # (needs data/) for a human
    .venv/bin/python tools/tag_anchor_check.py --proxy-side              # the same picture on the proxy
    .venv/bin/python tools/tag_anchor_check.py --bar-ath --methods v5_frame_model c3s_dead2

Before any set-piece tag can anchor a stoppage in the decoder (PLAN E8, REVIEW.md item 6), one
thing has to be measured: where the tag's start lies relative to the restart kick. The tags are
automatic 35 s phases; the EDA says every type is one window per set piece, but not whether the
window opens at the kick, at the ball going out, or somewhere between. BAR-ATH has no labels, so
the question is put to two label-free clocks: the submitted method's own decoded restarts (D to
a team) and the touch engine's first ball event after the tag opens.

Per tag type, `--bar-ath` prints:
  - n tags, and the share with a decoded restart within +-2 s of the tag start and within
    -5..+10 s of it;
  - the offset of the nearest decoded restart from the tag start, median (p10..p90), seconds
    (positive: the restart is decoded after the tag opens);
  - decoded D share in the 10 s before the tag start, in the 5 s after it, and over the rest
    of the window;
  - the offset of the touch engine's first touch or control start at or after 3 s before the
    tag start, median (p10..p90);
  - where the tag carries a team: the share of tags whose decoded team right after the nearest
    restart is the tag's team.
`--proxy-side` prints the same rows with the DFL restart events standing in for tag starts (a
"tag" opening exactly at the kick, with the event feed's own timing error of about a second),
plus the truth's own row. Reading BAR-ATH's offsets against the proxy's says whether the tag
opens at the kick (offsets alike), before it (BAR-ATH offsets positive and D before the start
high) or after it (negative). Nothing here is scored or tuned; aggregates only.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from solution import touch  # noqa: E402
from solution.match import AWAY, HOME, pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from tools.deadball_check import TAG_TYPES, inputs, restart_frames  # noqa: E402

WINDOW_S = 35.0            # the tags' typical length (FINDINGS A5: p50 35 s)
NEAR_S = 2.0               # "a restart at the tag start": within this
REACH = (-5.0, 10.0)       # "a restart around the tag start": within this
BEFORE_S, AFTER_S = 10.0, 5.0
TOUCH_FROM_S = -3.0        # the touch engine's first event from this far before the tag start
TEAM_CODES = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY}


def tag_windows(match):
    """[(start, stop, kind, team)] frames of the BAR-ATH SET PIECES tags inside the halves;
    team is H, A or None (the tag's Team is N/A for every throw-in)."""
    import load_smart_tagging
    out = []
    for tag in load_smart_tagging.load():
        if (tag["code"] or "").strip().lower() != "set pieces":
            continue
        kind = TAG_TYPES.get((tag["tags"].get("Type") or "").strip().lower(), "other")
        team = TEAM_CODES.get((tag["tags"].get("Team") or "").strip().lower())
        if any(a <= tag["start_frame"] < b for a, b in match.halves):
            out.append((tag["start_frame"], tag["end_frame"], kind, team))
    return out


def touch_events(match):
    """Sorted frames of the touch engine's touches and control starts."""
    found = touch.detect(match)
    return np.sort(np.concatenate((found.touch_frame, found.control[:, 0]))).astype(np.int64)


def anchor_records(codes, events, windows, halves, fps):
    """One dict per tag window, from `codes` (scorer codes) and the touch `events`."""
    runs = [tuple(h) for h in halves]
    restarts = restart_frames(codes, runs)
    out = []
    for start, stop, kind, team in windows:
        lo, hi = start + int(REACH[0] * fps), start + int(REACH[1] * fps)
        near = restarts[(restarts >= lo) & (restarts < hi)]
        offset = None
        if len(near):
            offset = float(near[np.argmin(np.abs(near - start))] - start) / fps
        before = codes[max(start - int(BEFORE_S * fps), 0):start]
        after = codes[start:start + int(AFTER_S * fps)]
        rest = codes[start + int(AFTER_S * fps):stop]
        first = events[events >= start + int(TOUCH_FROM_S * fps)]
        touch_offset = float(first[0] - start) / fps if len(first) else None
        agree = None
        if team is not None and offset is not None:
            frame = start + int(round(offset * fps))
            decoded = codes[frame:frame + int(2 * fps)]
            decoded = decoded[np.isin(decoded, pm.POSSESSION_LABELS)]
            if len(decoded):
                agree = bool(np.bincount(decoded, minlength=3)[team] * 2 > len(decoded))
        out.append({"kind": kind, "offset": offset, "touch_offset": touch_offset,
                    "d_before": float(np.mean(before == pm.D)) if len(before) else None,
                    "d_after": float(np.mean(after == pm.D)) if len(after) else None,
                    "d_rest": float(np.mean(rest == pm.D)) if len(rest) else None,
                    "team_agree": agree})
    return out


def _pct(values):
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return "{:+.1f} ({:+.1f}..{:+.1f})".format(p50, p10, p90)


def _share(values):
    values = [v for v in values if v is not None]
    return "{:.2f}".format(np.mean(values)) if values else "--"


def summarise(records):
    """{kind: row dict}, plus 'all'."""
    by_kind = defaultdict(list)
    for r in records:
        by_kind[r["kind"]].append(r)
        by_kind["all"].append(r)
    rows = {}
    for kind, rs in by_kind.items():
        offsets = [r["offset"] for r in rs]
        rows[kind] = {
            "n": len(rs),
            "near": np.mean([o is not None and abs(o) <= NEAR_S for o in offsets]),
            "reach": np.mean([o is not None for o in offsets]),
            "offset": _pct(offsets),
            "d_before": _share([r["d_before"] for r in rs]),
            "d_after": _share([r["d_after"] for r in rs]),
            "d_rest": _share([r["d_rest"] for r in rs]),
            "touch": _pct([r["touch_offset"] for r in rs]),
            "team": _share([r["team_agree"] for r in rs]),
        }
    return rows


def print_rows(title, rows):
    print("\n" + title)
    print("{:<10} {:>4} {:>6} {:>6} {:>20} {:>8} {:>8} {:>7} {:>20} {:>6}".format(
        "type", "n", "+-2 s", "reach", "restart offset s", "D before", "D after", "D rest",
        "first touch s", "team"))
    order = ["throw_in", "goal_kick", "corner", "free_kick", "goal", "kickoff", "other", "all"]
    for kind in order:
        if kind not in rows:
            continue
        r = rows[kind]
        print("{:<10} {:>4} {:>6.2f} {:>6.2f} {:>20} {:>8} {:>8} {:>7} {:>20} {:>6}".format(
            kind, r["n"], r["near"], r["reach"], r["offset"], r["d_before"], r["d_after"],
            r["d_rest"], r["touch"], r["team"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["v5_frame_model"],
                        choices=sorted(METHODS))
    parser.add_argument("--bar-ath", action="store_true",
                        help="(needs data/) BAR-ATH, label-free, for a person to run")
    parser.add_argument("--proxy-side", action="store_true",
                        help="DFL restart events as tag starts on the stressed proxy")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    if not (args.bar_ath or args.proxy_side):
        parser.error("pass --bar-ath or --proxy-side")

    if args.bar_ath:
        from solution.sources import atd
        match = atd.load()
        windows = tag_windows(match)
        events = touch_events(match)
        print("BAR-ATH: {} SET PIECES tags inside the halves; offsets are seconds from the tag "
              "start".format(len(windows)))
        for name in args.methods:
            codes = METHODS[name](match)
            print_rows(name, summarise(anchor_records(codes, events, windows, match.halves,
                                                      match.fps)))
    if args.proxy_side:
        from solution.sources import idsse
        records = defaultdict(list)
        for truth, match, restarts, _ in inputs(idsse.DEV_IDS, args.seeds, plain=False):
            windows = [(f, f + int(WINDOW_S * match.fps), kind, None) for f, kind in restarts]
            events = touch_events(match)
            records["truth"] += anchor_records(truth, events, windows, match.halves, match.fps)
            for name in args.methods:
                records[name] += anchor_records(METHODS[name](match), events, windows,
                                                match.halves, match.fps)
        print("IDSSE dev x seeds {}, D12 ball: DFL restart events as tag starts (the event clock "
              "is about a second off)".format(args.seeds))
        for name, rs in records.items():
            print_rows(name, summarise(rs))


if __name__ == "__main__":
    main()
