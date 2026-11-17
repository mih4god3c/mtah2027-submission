"""Where does a method lose on dead ball? Step 0 of C2b (PLAN.md C2b, E2). Aggregates only.

    .venv/bin/python tools/deadball_check.py                       # v3_kickoff, D12 ball stress
    .venv/bin/python tools/deadball_check.py --methods v3_kickoff touch_deadball
    .venv/bin/python tools/deadball_check.py --plain               # simulator v1, clean ball
    .venv/bin/python tools/deadball_check.py --bar-ath --methods v3_kickoff touch_deadball
                                                  # (needs data/) label-free, for a human to run
    .venv/bin/python tools/deadball_check.py --proxy-side --methods ...   # the same on the proxy

On the IDSSE dev matches, degraded as `make proxy` does (simulator v1, then the D12 ball stress
unless --plain), it compares each method's dead-ball stretches with the true ones:
  - every true D stretch gets the restart that ends it, from the DFL event feed
    (idsse.restarts): one-to-one, the event nearest the stretch's end within MATCH_S. The feed
    is timed by hand, a median -2.0 to +0.4 s from the end depending on the match;
  - per restart type: how many stretches are missed (no predicted D frame inside), the share of
    their frames covered, and, for the found ones, the start and end error of the predicted D
    that overlaps them (pred - true, seconds);
  - false stoppages: predicted D stretches that touch no true D frame;
  - oracle ceilings, pooled spellF1 exactly as the proxy league pools it: the prediction with its
    dead ball repaired from the truth, and the gain over the method as it is. "perfect" repairs
    everything; the others repair one kind of error, or one restart type, and leave the rest.
    A repaired frame that was wrongly D takes the team of the nearest frame the method itself
    labelled H or A, so the repairs never fix possession, only dead ball.
  - E10 Step 0, the whistle ceilings: the same repairs for the stoppages a referee whistles
    (WHISTLED: free kicks, which include fouls and offsides, penalties, goals, dropped balls).
    "whistle start" moves their start edges only, "whistle start+missed" also adds their missed
    stretches whole, "whistle all" repairs those types entirely (an upper bound).
--labels picks the truth (default keep2, the proxy's decision truth, D21); D is the same in
every variant, so only the possession part of spellF1 moves.

With --bar-ath (Step 2, no labels): per method, the decoded D share of graded frames (IDSSE
dev truth: 0.41 to 0.48, D9), stoppages per 90 minutes and their median length, how many
frames it turns to D or away from D against the first method, and how often the dead-ball cues
of solution/deadball.py fire. Then agreement with the Smart Tagging SET PIECES windows: per
set-piece type, the share of windows (widened by TAG_PAD_S) holding a decoded restart (D to a
team), and the median offset of the first one from the window start. Set pieces end a dead
ball, so a method that finds stoppages should restart inside most of them; 32 of the 59 have
no team, and throw-ins are under-tagged, so this checks only one direction. --proxy-side prints
the same aggregates on the stressed proxy, with the DFL restart events in place of the tags
(windows of TAG_PAD_S either side of the event). Nothing frame-level is printed.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.simulate import add_ball_noise, degrade, retime, retime_frame  # noqa: E402
from tools.proxy_league import f1_from_flags, spell_flags  # noqa: E402

STRESS = (0.05, 3.4, 0.26)          # D12: sd m, jumps per visible minute, smoothing s
MATCH_S = 6.0                       # a restart event belongs to a stretch ending this close
KINDS = ("throw_in", "goal_kick", "corner", "free_kick", "goal", "penalty", "referee_ball",
         "kickoff", "unknown")
LENGTH_BINS_S = (0.0, 2.0, 5.0, 15.0, 60.0, np.inf)
WHISTLED = ("free_kick", "penalty", "goal", "referee_ball")   # stoppages the referee whistles
TAG_PAD_S = 2.0                     # set-piece windows are widened by this much either side
TAG_TYPES = {"throw-in": "throw_in", "goal kick": "goal_kick", "corner kick": "corner",
             "free kick": "free_kick", "kick off (after goal)": "goal",
             "kick off (start)": "kickoff"}


def dead_stretches(codes, runs):
    """[(start, stop)] maximal D runs inside the graded runs."""
    return [(a, b) for a, b, k in pm.segments(codes, runs) if k == pm.D]


def restart_kinds(stretches, events, fps):
    """Restart type per true D stretch: one-to-one, nearest event to the stretch end."""
    ends = np.array([b for _, b in stretches], dtype=float)
    pairs = sorted((abs(frame - end), i, j) for j, (frame, _) in enumerate(events)
                   for i, end in enumerate(ends) if abs(frame - end) <= MATCH_S * fps)
    kinds, used = ["unknown"] * len(stretches), set()
    for _, i, j in pairs:
        if kinds[i] == "unknown" and j not in used:
            kinds[i] = events[j][1]
            used.add(j)
    return kinds


def overlaps(stretches, others):
    """For each stretch, the indices of `others` it shares a frame with (both sorted)."""
    out, j = [], 0
    for a, b in stretches:
        while j < len(others) and others[j][1] <= a:
            j += 1
        k, hit = j, []
        while k < len(others) and others[k][0] < b:
            hit.append(k)
            k += 1
        out.append(hit)
    return out


def nearest_team(pred, runs):
    """Per frame, the H/A label of the nearest frame `pred` labels H or A in the same run."""
    out = np.full(len(pred), pm.H, dtype=pred.dtype)
    for start, stop in runs:
        window = pred[start:stop]
        alive = np.flatnonzero(np.isin(window, pm.POSSESSION_LABELS))
        if not len(alive):
            continue
        index = np.arange(len(window))
        right = np.clip(np.searchsorted(alive, index), 0, len(alive) - 1)
        left = np.clip(right - 1, 0, len(alive) - 1)
        pick = np.where(np.abs(alive[left] - index) <= np.abs(alive[right] - index),
                        alive[left], alive[right])
        out[start:stop] = window[pick]
    return out


def repair(truth, pred, runs, force, release):
    """pred with frames in `force` set to D, and its D frames in `release` given a team."""
    filled = pm.fill_abstentions(pred, runs)
    out = filled.copy()
    release = release & (filled == pm.D)
    out[release] = nearest_team(filled, runs)[release]
    out[force] = pm.D
    return out


def analyse(truth, pred, events, fps):
    """(records per true stretch, false stretches, {repair name: repaired codes})."""
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    true_d = dead_stretches(truth, runs)
    pred_d = dead_stretches(filled, runs)
    kinds = restart_kinds(true_d, events, fps)
    hits = overlaps(true_d, pred_d)
    touched = set(j for hit in hits for j in hit)
    is_true_d = truth == pm.D
    records = []
    for (a, b), kind, hit in zip(true_d, kinds, hits):
        row = {"kind": kind, "seconds": (b - a) / float(fps), "found": bool(hit),
               "covered": float(np.mean(filled[a:b] == pm.D))}
        if hit:
            row["start_s"] = (pred_d[hit[0]][0] - a) / float(fps)
            row["end_s"] = (pred_d[hit[-1]][1] - b) / float(fps)
        records.append(row)
    false = [pred_d[j] for j in range(len(pred_d)) if j not in touched]

    n = len(truth)
    def mask(spans):
        out = np.zeros(n, dtype=bool)
        for a, b in spans:
            out[a:b] = True
        return out
    false_mask = mask(false)
    repairs = {"perfect": repair(truth, pred, runs, is_true_d, ~is_true_d),
               "missed": repair(truth, pred, runs,
                                mask([s for s, r in zip(true_d, records) if not r["found"]]),
                                np.zeros(n, dtype=bool)),
               "false": repair(truth, pred, runs, np.zeros(n, dtype=bool), false_mask)}
    found = [i for i, r in enumerate(records) if r["found"]]
    near = mask([pred_d[j] for i in found for j in hits[i]]) & ~is_true_d
    repairs["edges"] = repair(truth, pred, runs, mask([true_d[i] for i in found]), near)
    def edge_masks(edge, chosen):
        """(force, release): the predicted D's first (last) frame moved onto the true one."""
        force, release = [], []
        for i in chosen:
            (a, b), first, last = true_d[i], pred_d[hits[i][0]][0], pred_d[hits[i][-1]][1]
            if edge == "start":
                (force if first > a else release).append((min(a, first), max(a, first)))
            else:
                (force if last < b else release).append((min(b, last), max(b, last)))
        return mask(force) & is_true_d, mask(release) & ~is_true_d

    for edge in ("start", "end"):
        repairs["edge " + edge] = repair(truth, pred, runs, *edge_masks(edge, found))

    def type_repair(chosen):
        near = mask([pred_d[j] for i in chosen for j in hits[i]]) & ~is_true_d
        return repair(truth, pred, runs, mask([true_d[i] for i in chosen]), near)

    for kind in set(kinds):
        repairs["type " + kind] = type_repair([i for i, k in enumerate(kinds) if k == kind])
    # E10 Step 0: what a perfect whistle could fix. The referee whistles the stoppage itself
    # only for the WHISTLED restarts; a whistle gives the start, so "whistle start" moves only
    # start edges, "+missed" also adds the whole missed stretches (generous: the whistle does
    # not say when play resumes), "whistle all" repairs those types entirely (upper bound).
    blown = [i for i, k in enumerate(kinds) if k in WHISTLED]
    blown_found = [i for i in blown if records[i]["found"]]
    force, release = edge_masks("start", blown_found)
    repairs["whistle start"] = repair(truth, pred, runs, force, release)
    missed = mask([true_d[i] for i in blown if not records[i]["found"]])
    repairs["whistle start+missed"] = repair(truth, pred, runs, force | missed, release)
    repairs["whistle all"] = type_repair(blown)
    return records, false, repairs


def inputs(match_ids, seeds, plain, labels=None, tempo=1.0):
    """(truth, degraded match, restart events, seed) per match and seed. `labels` picks the
    truth's label variant (idsse.LABELS when None); the simulator never reads labels, so the
    degraded inputs are the same for every variant. `tempo` != 1 retimes the clean match
    first (simulate.retime, the D9 tempo stress), truth and restart events with it."""
    from solution.sources import idsse
    for match_id in match_ids:
        raw = idsse.load(match_id, labels=labels)
        spans, offset = [], 0
        for _, first, last, _ in idsse.half_clocks(match_id):
            spans.append((offset, offset + last - first + 1))
            offset += last - first + 1 + idsse.HALF_GAP
        assert spans == [tuple(h) for h in raw.halves], "event clock disagrees with the match"
        events = idsse.restarts(match_id)
        if tempo != 1:
            raw = retime(raw, tempo)
            events = [(retime_frame(f, tempo), kind) for f, kind in events]
        for seed in seeds:
            match = degrade(raw, seed)
            if not plain:
                match = add_ball_noise(match, *STRESS[:2], seed=seed, smooth_s=STRESS[2])
            yield raw.labels, match, events, seed


def percentiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return "--"
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return "{:+.2f} ({:+.2f}..{:+.2f})".format(p50, p10, p90)


def report(name, rows, n_seeds):
    records = [r for row in rows for r in row["records"]]
    false = [s for row in rows for s in row["false"]]
    fps = rows[0]["fps"]
    print("\n== {} ==  ({} true D stretches over {} seeds)".format(name, len(records), n_seeds))
    print("{:<13} {:>6} {:>8} {:>8} {:>8} {:>21} {:>21} {:>9}".format(
        "restart", "n/seed", "true s", "missed", "covered", "start err s p50 (p10..p90)",
        "end err s p50 (p10..p90)", "|end|>1s"))
    by_kind = defaultdict(list)
    for r in records:
        by_kind[r["kind"]].append(r)
    for kind in KINDS + ("all",):
        group = records if kind == "all" else by_kind.get(kind, [])
        if not group:
            continue
        found = [r for r in group if r["found"]]
        ends = [r["end_s"] for r in found]
        print("{:<13} {:>6.1f} {:>8.1f} {:>8.3f} {:>8.3f} {:>27} {:>25} {:>9}".format(
            kind, len(group) / float(n_seeds), np.median([r["seconds"] for r in group]),
            1.0 - len(found) / float(len(group)), np.mean([r["covered"] for r in group]),
            percentiles([r["start_s"] for r in found]), percentiles(ends),
            "{:.3f}".format(np.mean(np.abs(ends) > 1.0)) if ends else "--"))
    print("missed by true length:", "  ".join(
        "{}-{} s {:.3f} of {:.0f}".format(int(lo), "" if hi == np.inf else int(hi),
                                         1.0 - np.mean([r["found"] for r in group]) if group else 0,
                                         len(group) / float(n_seeds))
        for lo, hi in zip(LENGTH_BINS_S[:-1], LENGTH_BINS_S[1:])
        for group in [[r for r in records if lo <= r["seconds"] < hi]]))
    lengths = np.array([b - a for a, b in false], dtype=float) / fps
    print("false stoppages: {:.1f} per seed, {:.0f} s per seed, length s p50 {:.1f} "
          "(p10 {:.1f}, p90 {:.1f})".format(
              len(false) / float(n_seeds), lengths.sum() / n_seeds,
              *(np.percentile(lengths, [50, 10, 90]) if len(lengths) else [np.nan] * 3)))
    truth_d = sum(row["true_d"] for row in rows)
    pred_d = sum(row["pred_d"] for row in rows)
    both = sum(row["both_d"] for row in rows)
    print("D frames: recall {:.3f}, precision {:.3f}, predicted / true {:.3f}".format(
        both / float(truth_d), both / float(max(pred_d, 1)), pred_d / float(truth_d)))

    names = ["as is"] + sorted(set(k for row in rows for k in row["repairs"]),
                               key=lambda k: (k.startswith("type"), k))
    scores = {}
    for key in names:
        per_seed = []
        for seed in sorted(set(row["seed"] for row in rows)):
            parts, n_pred = [], 0
            for row in rows:
                if row["seed"] != seed:
                    continue
                pred = row["pred"] if key == "as is" else row["repairs"].get(key, row["pred"])
                flags, n = spell_flags(row["truth"], pred)
                parts.append(flags)
                n_pred += n
            per_seed.append(f1_from_flags(np.concatenate(parts, axis=1), n_pred))
        scores[key] = float(np.mean(per_seed))
    print("oracle ceilings, pooled spellF1 (gain over as is):")
    for key in names:
        print("  {:<22} {:.4f}  {:+.4f}".format(key, scores[key], scores[key] - scores["as is"]))
    return scores


# -- Step 2: label-free aggregates, for BAR-ATH ----------------------------------------------

def restart_frames(codes, runs):
    """Frames where a graded run goes from D to a team."""
    return np.array([f for f, before, after in pm.transitions(codes, runs)
                     if before == pm.D and after in pm.POSSESSION_LABELS], dtype=np.int64)


def label_free(match, codes, reference, windows):
    """Aggregates of one method's codes on one match; `windows` [(start, stop, kind)] frames."""
    from solution import deadball
    runs = [tuple(h) for h in match.halves]
    inside = match.in_play_mask()
    minutes = inside.sum() / float(match.fps) / 60.0
    stoppages = dead_stretches(codes, runs)
    restarts = restart_frames(codes, runs)
    pad = int(round(TAG_PAD_S * match.fps))
    tags = defaultdict(list)
    for start, stop, kind in windows:
        hit = restarts[(restarts >= start - pad) & (restarts < stop + pad)]
        tags[kind].append(None if not len(hit) else (hit[0] - start) / float(match.fps))
    seen = match.ball_seen & inside
    cue_params = dict(deadball.PARAMS, cues=("player", "ball"))
    cue = deadball.cues(match, cue_params)
    beyond0 = deadball.ball_beyond(match, dict(cue_params, ball_m=0.0))
    return {"minutes": minutes, "d_frames": int(np.sum(codes[inside] == pm.D)),
            "frames": int(inside.sum()), "stoppages": len(stoppages),
            "stoppage_s": [(b - a) / float(match.fps) for a, b in stoppages],
            "to_d": int(np.sum(inside & (codes == pm.D) & (reference != pm.D))),
            "from_d": int(np.sum(inside & (codes != pm.D) & (reference == pm.D))),
            "player_cue": int(np.sum(cue[inside, 0])),
            "ball_cue": int(np.sum(cue[seen, 1])), "ball_beyond": int(np.sum(beyond0[seen])),
            "seen": int(seen.sum()), "tags": dict(tags)}


def print_label_free(title, rows_by_method):
    print("\n" + title)
    print("{:<20} {:>8} {:>12} {:>10} {:>10} {:>10}".format(
        "method", "D share", "stops / 90", "median s", "to D", "from D"))
    for name, rows in rows_by_method.items():
        frames = sum(r["frames"] for r in rows)
        minutes = sum(r["minutes"] for r in rows)
        lengths = [v for r in rows for v in r["stoppage_s"]]
        print("{:<20} {:>8.3f} {:>12.1f} {:>10.1f} {:>10.4f} {:>10.4f}".format(
            name[:20], sum(r["d_frames"] for r in rows) / float(frames),
            sum(r["stoppages"] for r in rows) * 90.0 / minutes,
            np.median(lengths) if lengths else float("nan"),
            sum(r["to_d"] for r in rows) / float(frames),
            sum(r["from_d"] for r in rows) / float(frames)))
    rows = next(iter(rows_by_method.values()))
    frames, seen = sum(r["frames"] for r in rows), sum(r["seen"] for r in rows)
    print("cues: player beyond a line on {:.4f} of graded frames; ball beyond a line by > 1 m "
          "on {:.4f} of visible graded frames, beyond at all on {:.4f}".format(
              sum(r["player_cue"] for r in rows) / float(frames),
              sum(r["ball_cue"] for r in rows) / float(max(seen, 1)),
              sum(r["ball_beyond"] for r in rows) / float(max(seen, 1))))
    print("set pieces with a decoded restart inside (window +-{:.0f} s): share, median offset of "
          "the first from the window start, s".format(TAG_PAD_S))
    kinds = sorted(set(k for r in rows for k in r["tags"]))
    print("{:<20}".format("method") + "".join("{:>22}".format(k) for k in kinds + ["all"]))
    for name, rows in rows_by_method.items():
        cells = []
        for kind in kinds + ["all"]:
            values = [v for r in rows for k, vs in r["tags"].items() for v in vs
                      if kind in ("all", k)]
            hits = [v for v in values if v is not None]
            cells.append("{:.2f} of {} ({})".format(
                len(hits) / float(len(values)), len(values),
                "{:+.1f}".format(np.median(hits)) if hits else "--") if values else "--")
        print("{:<20}".format(name[:20]) + "".join("{:>22}".format(c) for c in cells))


def smart_tag_windows(match):
    """[(start, stop, kind)] video frames of the BAR-ATH SET PIECES tags inside the halves."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import load_smart_tagging
    out = []
    for tag in load_smart_tagging.load():
        if (tag["code"] or "").strip().lower() != "set pieces":
            continue
        kind = TAG_TYPES.get((tag["tags"].get("Type") or "").strip().lower(), "other")
        if any(a <= tag["start_frame"] < b for a, b in match.halves):
            out.append((tag["start_frame"], tag["end_frame"], kind))
    return out


def bar_ath(methods):
    from solution.sources import atd
    match = atd.load()
    windows = smart_tag_windows(match)
    codes = {name: METHODS[name](match) for name in methods}
    reference = codes[methods[0]]
    print_label_free("BAR-ATH (no labels; IDSSE dev truth D share 0.41 to 0.48)",
                     {name: [label_free(match, codes[name], reference, windows)]
                      for name in methods})


def proxy_side(methods, seeds, plain):
    """The same aggregates on the stressed proxy, DFL restart events standing in for tags."""
    from solution.sources import idsse
    rows = {name: [] for name in methods + ["truth"]}
    for truth, match, events, _ in inputs(idsse.DEV_IDS, seeds, plain):
        windows = [(f, f + 1, kind) for f, kind in events]      # label_free widens them
        codes = {name: METHODS[name](match) for name in methods}
        codes["truth"] = truth
        for name in rows:
            rows[name].append(label_free(match, codes[name], codes[methods[0]], windows))
    print_label_free("IDSSE dev x seeds {} (restart events +-{:.0f} s as the tags)".format(
        seeds, TAG_PAD_S), rows)


def main():
    from solution.sources import idsse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["v3_kickoff"], choices=sorted(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--plain", action="store_true", help="no ball stress (plain v1)")
    parser.add_argument("--labels", default="keep2",
                        help="truth label variant (keep2, the proxy's decision truth, D21; or dfl)")
    parser.add_argument("--bar-ath", action="store_true",
                        help="(needs data/) label-free BAR-ATH aggregates, for a human to run")
    parser.add_argument("--proxy-side", action="store_true",
                        help="the --bar-ath aggregates on the stressed proxy, for comparison")
    args = parser.parse_args()
    if args.bar_ath:
        bar_ath(args.methods)
        return
    if args.proxy_side:
        proxy_side(args.methods, args.seeds, args.plain)
        return
    print("dead-ball check: IDSSE dev x seeds {}, simulator v1{}, {} truth".format(
        args.seeds, "" if args.plain else " + D12 ball stress {}".format(STRESS), args.labels))
    rows = {name: [] for name in args.methods}
    for truth, match, events, seed in inputs(idsse.DEV_IDS, args.seeds, args.plain,
                                             args.labels):
        for name in args.methods:
            pred = METHODS[name](match)
            records, false, repairs = analyse(truth, pred, events, match.fps)
            inside = match.in_play_mask()
            rows[name].append({"records": records, "false": false, "repairs": repairs,
                               "truth": truth, "pred": pred, "seed": seed, "fps": match.fps,
                               "true_d": int(np.sum(truth[inside] == pm.D)),
                               "pred_d": int(np.sum(pred[inside] == pm.D)),
                               "both_d": int(np.sum((truth == pm.D) & (pred == pm.D) & inside))})
    for name in args.methods:
        report(name, rows[name], len(args.seeds))


if __name__ == "__main__":
    main()
