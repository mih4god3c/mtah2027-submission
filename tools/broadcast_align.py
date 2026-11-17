"""E13 Step 0: align the broadcast ATD to the tactical timeline and measure what it could add.
Aggregates only, for a human to run on BAR-ATH (needs data/).

    .venv/bin/python tools/broadcast_align.py                  # full report
    .venv/bin/python tools/broadcast_align.py --method none    # skip the predicted-label split
                                                               # (no IDSSE download needed)
    make broadcast-align

The alignment is solution/broadcast.py (algorithmic, parameters fixed before the first run).
The report, in sections:

  A. The broadcast files: record count and contiguity, the frame rate both metadata files state,
     slots, ball and homography shares.
  B. Orientation: the four mirrorings tried on the longest stretches; the best one is used.
  C. Alignment: how many valid stretches align confidently, with the cost against its rivals
     (metres; a clear gap between them is what makes the offsets trustworthy), the live
     segments and their offsets, drift inside long stretches (non-zero means the frame rates
     differ), replay candidates, and whether both files code home and away the same way.
  D. Coverage of the graded tactical frames: what the TV was showing at each moment.
  E. Ball: tactical, broadcast and union visibility; where the tactical ball is missing, what
     the broadcast had; by gap length; position agreement where both see it. The union share per
     half is the input for the proxy what-if `make proxy BALL_VISIBLE=...`. Note that
     simulate.reveal_gaps reveals whole gaps at random, while the broadcast fills some gaps only
     in part and not at random, so that run is an optimistic upper bound for this coverage.
  F. Cut-aways as a dead-ball cue: the share of time the TV is off the live pitch, split by the
     submitted method's predicted label (--method) and by Smart Tagging set pieces. No labels:
     the split is against our own prediction and the organiser's automatic tags.
  G. Cut-away timing, the inputs for a synthetic cut-away cue on the proxy: run lengths, the
     off-pitch share around each tagged set piece by Type, and against the predicted D runs (by
     length: how often the TV cuts away, the lag after the run starts and before it ends), plus
     off-pitch runs inside predicted live play, next to a predicted D run or isolated.
  H. The ball merge (broadcast.fuse_ball, D27 reopened by miha): what the fill rule keeps, and
     how the method's output changes on the merged input (label-free).

Writes cache/broadcast/BAR-ATH_align.npz (gitignored) for later fusion work. It is derived from
the match data: never commit it, and agents do not read it.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_atd  # noqa: E402
import load_homography  # noqa: E402
import load_smart_tagging  # noqa: E402
from games import GAME, require  # noqa: E402
from solution import broadcast as bc  # noqa: E402
from solution.match import CACHE_DIR, pm  # noqa: E402
from solution.sources import atd  # noqa: E402

GAP_BINS_S = (0, 1, 3, 10, 30, np.inf)
OUT_PATH = os.path.join(CACHE_DIR, "broadcast", GAME + "_align.npz")


def meta_frame_rate(source):
    path = require(["atd_meta"], GAME, None, source)[0]
    for element in ET.parse(path).getroot().iter():
        if element.tag.split("}")[-1].lower() == "framerate" and (element.text or "").strip():
            return element.text.strip()
    return "not stated"


def pct(x):
    return "{:5.1f}%".format(100 * x) if np.isfinite(x) else "   n/a"


def share(mask, within):
    n = int(within.sum())
    return float((mask & within).sum()) / n if n else np.nan


def quantiles(values, qs=(10, 50, 90)):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(values):
        return "n=0"
    return "n={}  ".format(len(values)) + "  ".join(
        "p{}={:.2f}".format(q, np.percentile(values, q)) for q in qs)


def header(title):
    print("\n" + "=" * 94 + "\n" + title + "\n" + "=" * 94)


def section_files(bro, valid):
    header("A  broadcast files")
    check = load_atd.check_alignment_raw(source="broadcast")
    print("  records {}, expected {}, frame_id {}..{}, contiguous {}".format(
        check["n_records"], check["n_video_frames"], *check["frame_id_range"],
        check["contiguous"]))
    print("  frame rate in metadata: tactical {}, broadcast {}".format(
        meta_frame_rate("tactical"), meta_frame_rate("broadcast")))
    print("  slots {} (home {}, away {}); ball seen {}; homography valid {}; ball seen where "
          "valid {}".format(len(bro.team), int((bro.team == 0).sum()), int((bro.team == 1).sum()),
                            pct(bro.ball_seen.mean()), pct(valid.mean()),
                            pct(share(bro.ball_seen, valid))))


def section_orientation(tac, bro, valid):
    header("B  orientation ({} longest valid stretches)".format(bc.N_FLIP_STRETCHES))
    flip, rows = bc.choose_flip(tac, bro, valid)
    for row in rows:
        print("  flip {:>4}: confident {}/{}, median cost {}".format(
            row["flip"], row["confident"], row["stretches"],
            "{:.2f} m".format(row["cost"]) if np.isfinite(row["cost"]) else "n/a"))
    print("  using: {}".format(flip))
    return flip


def section_alignment(tac, bro, valid, out):
    header("C  alignment (tactical frame = broadcast frame + d)")
    results, kinds, chain = out["results"], out["kinds"], out["chain"]
    fps = tac.fps
    valid_frames = int(valid.sum())
    length = np.array([r["stop"] - r["start"] for r in results])
    conf = np.array([r["confident"] for r in results], dtype=bool)
    print("  valid stretches >= {:g} s: {} ({} of valid frames)".format(
        bc.MIN_STRETCH_S, len(results), pct(length.sum() / max(valid_frames, 1))))
    print("  confident: {} stretches ({} of their frames), {} of them in pass 2".format(
        int(conf.sum()), pct(length[conf].sum() / max(length.sum(), 1)),
        sum(r["pass"] == 2 for r in results if r["confident"])))
    print("  cost, confident (m):    " + quantiles([r["cost"] for r in results if r["confident"]]))
    print("  rival, confident (m):   " + quantiles([r["rival"] for r in results if r["confident"]]))
    print("  random offsets (m):     " + quantiles([r["probe"] for r in results]))
    print("  cost, not confident:    " + quantiles(
        [r["cost"] for r in results if not r["confident"]]))
    print("  ball votes per stretch: " + quantiles([r["votes"] for r in results]))

    for kind in ("live", "replay", "ahead", "off-chain", "unaligned"):
        idx = [k for k, x in enumerate(kinds) if x == kind]
        secs = sum(length[k] for k in idx) / fps
        print("  {:>9}: {:3d} stretches, {:7.1f} s, length {}".format(
            kind, len(idx), secs, quantiles([length[k] / fps for k in idx])))

    # live segments: consecutive chain stretches whose offsets agree
    ordered = sorted(chain, key=lambda k: results[k]["start"])
    segments = []
    for k in ordered:
        r = results[k]
        if segments and abs(r["d"] - segments[-1]["d_last"]) <= bc.CHAIN_TOL:
            seg = segments[-1]
            seg.update(d_last=r["d"], stop=r["stop"], n=seg["n"] + 1,
                       d_min=min(seg["d_min"], r["d"]), d_max=max(seg["d_max"], r["d"]))
        else:
            segments.append({"d_first": r["d"], "d_last": r["d"], "d_min": r["d"],
                             "d_max": r["d"], "start": r["start"], "stop": r["stop"], "n": 1})
    print("  live segments: {}".format(len(segments)))
    for seg in segments:
        print("    d {:>7} (range {}..{}), {:3d} stretches, tactical frames {}..{} ({:.0f} s)"
              .format(seg["d_first"], seg["d_min"], seg["d_max"], seg["n"],
                      seg["start"] + seg["d_first"], seg["stop"] + seg["d_last"],
                      (seg["stop"] - seg["start"]) / fps))

    drift = [r["drift"] for r in results if r["drift"] is not None]
    bins = {"0": 0, "1": 0, "2-5": 0, ">5": 0}
    for v in drift:
        a = abs(v)
        bins["0" if a == 0 else "1" if a == 1 else "2-5" if a <= 5 else ">5"] += 1
    print("  drift inside stretches >= {:g} s (frames, last third - first third): {}".format(
        bc.DRIFT_MIN_S, ", ".join("|{}|: {}".format(k, v) for k, v in bins.items())))
    if drift:
        print("    mean drift {:+.2f} frames over a median {:.0f} s".format(
            float(np.mean(drift)),
            float(np.median([2 * (r["stop"] - r["start"]) / 3 / fps for r in results
                             if r["drift"] is not None]))))

    back = []
    for k, kind in enumerate(kinds):
        if kind == "replay":
            before = [j for j in chain if results[j]["start"] < results[k]["start"]]
            ref = results[before[-1] if before else chain[0]]["d"]
            back.append((ref - results[k]["d"]) / fps)
    print("  replays: how far back they show (s): " + quantiles(back))

    team_share, pairs = bc.team_agreement(tac, bro, results, chain)
    print("  team codes agree on {} of {} paired players{}".format(
        pct(team_share), pairs, "  <-- metadata team order differs between the feeds"
        if np.isfinite(team_share) and team_share < 0.5 else ""))


def section_coverage(tac, out, graded):
    header("D  what the broadcast shows, at the moment of each graded tactical frame")
    state = out["state"]
    for code, name in bc.STATE_NAMES.items():
        print("  {:<28} {}".format(name, pct(share(state == code, graded))))


def gap_table(tac, graded, filled):
    fps = tac.fps
    gaps = bc.runs(graded & ~tac.ball_seen)
    total = sum(b - a for a, b in gaps)
    print("  tactical ball gaps, by length: frames share | broadcast fills | gaps >= half filled")
    for lo, hi in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:]):
        picked = [(a, b) for a, b in gaps if lo * fps <= b - a < hi * fps]
        frames = sum(b - a for a, b in picked)
        got = sum(int(filled[a:b].sum()) for a, b in picked)
        half = sum(filled[a:b].mean() >= 0.5 for a, b in picked)
        print("    {:>4}-{:<4} s  {:5d} gaps  {}  |  {}  |  {}".format(
            lo, hi if np.isfinite(hi) else "", len(picked), pct(frames / max(total, 1)),
            pct(got / frames if frames else np.nan),
            pct(half / len(picked) if picked else np.nan)))


def section_ball(tac, bro, out, graded):
    header("E  ball")
    state, bro_frame, replay_frame = out["state"], out["bro_frame"], out["replay_frame"]
    live = state == bc.LIVE
    bro_seen = np.zeros(tac.n_frames, dtype=bool)
    bro_seen[live] = bro.ball_seen[bro_frame[live]]
    has_replay = replay_frame >= 0
    replay_seen = np.zeros(tac.n_frames, dtype=bool)
    replay_seen[has_replay] = bro.ball_seen[replay_frame[has_replay]]
    tac_seen = tac.ball_seen
    union_live = tac_seen | bro_seen
    union_all = union_live | replay_seen

    print("  graded frames: tactical sees {}, broadcast live sees {}, union {}, "
          "union with replays {}".format(pct(share(tac_seen, graded)),
                                         pct(share(bro_seen, graded)),
                                         pct(share(union_live, graded)),
                                         pct(share(union_all, graded))))
    for h, (a, b) in enumerate(tac.halves, 1):
        mask = np.zeros(tac.n_frames, dtype=bool)
        mask[a:b] = True
        mask &= graded
        print("    half {}: tactical {}, union {}, union with replays {}".format(
            h, pct(share(tac_seen, mask)), pct(share(union_live, mask)),
            pct(share(union_all, mask))))

    missing = graded & ~tac_seen
    print("  where the tactical ball is missing ({} graded frames), the broadcast at that moment:"
          .format(int(missing.sum())))
    print("    live pitch, sees the ball   {}".format(pct(share(bro_seen, missing))))
    print("    live pitch, no ball         {}".format(pct(share(live & ~bro_seen, missing))))
    for code in (bc.CUT, bc.REPLAY, bc.OTHER, bc.NOT_COVERED):
        print("    {:<27} {}".format(bc.STATE_NAMES[code], pct(share(state == code, missing))))
    print("    (a replay later shows this moment with the ball: {})".format(
        pct(share(replay_seen, missing))))
    gap_table(tac, graded, bro_seen | replay_seen)

    both = live & graded & tac_seen & bro_seen
    t = np.flatnonzero(both)
    b = bro_frame[t]
    dist = np.hypot(tac.ball_x[t] - bro.ball_x[b], tac.ball_y[t] - bro.ball_y[b])
    print("  both see the ball on {} graded frames; distance (m) {}; > 2 m {}, > 5 m {}".format(
        len(t), quantiles(dist, (50, 90, 99)),
        pct(float(np.mean(dist > 2)) if len(t) else np.nan),
        pct(float(np.mean(dist > 5)) if len(t) else np.nan)))
    return bro_seen, replay_seen


def predict(method, match):
    from solution import run
    from solution.methods import METHODS
    run.require_external()
    return METHODS[method](match)


def set_pieces():
    """[(start_frame, end_frame, Type)] of the Smart Tagging SET PIECES instances."""
    return [(max(0, e["start_frame"]), max(0, e["end_frame"]),
             (e["tags"].get("Type") or "untyped").strip())
            for e in load_smart_tagging.load()
            if (e["code"] or "").strip().upper() == "SET PIECES"]


def section_cutaway(tac, out, graded, method, codes):
    header("F  cut-aways as a dead-ball cue (no labels: our prediction and the automatic tags)")
    state = out["state"]
    covered = graded & (state != bc.NOT_COVERED)
    off = np.isin(state, (bc.CUT, bc.REPLAY))
    rows = []
    if codes is not None:
        rows += [("predicted D ({})".format(method), covered & (codes == pm.D)),
                 ("predicted H or A", covered & np.isin(codes, (pm.H, pm.A)))]
    pieces = np.zeros(tac.n_frames, dtype=bool)
    for a, b, _ in set_pieces():
        pieces[a:b] = True
    rows += [("inside SET PIECES tags", covered & pieces),
             ("outside SET PIECES tags", covered & ~pieces)]
    print("  {:<38} {:>9} {:>9} {:>9} {:>9}".format("graded, covered frames", "frames", "cut",
                                                    "replay", "off pitch"))
    for name, mask in rows:
        print("  {:<38} {:>9d} {:>9} {:>9} {:>9}".format(
            name, int(mask.sum()), pct(share(state == bc.CUT, mask)),
            pct(share(state == bc.REPLAY, mask)), pct(share(off, mask))))
    for a, b in ((0, 1), (2, 3)) if len(rows) == 4 else ((0, 1),):
        pa, pb = share(off, rows[a][1]), share(off, rows[b][1])
        print("  off-pitch lift, {} vs {}: {}".format(
            rows[a][0], rows[b][0], "{:.2f}x".format(pa / pb) if pb else "n/a"))


def section_timing(tac, out, graded, codes):
    header("G  cut-away timing (label-free; for the synthetic cue on the proxy)")
    fps = tac.fps
    state = out["state"]
    covered = state != bc.NOT_COVERED
    off = np.isin(state, (bc.CUT, bc.REPLAY))
    minutes = graded.sum() / fps / 60

    spans = bc.runs(off & graded)
    lengths = np.array([b - a for a, b in spans]) / fps
    print("  off-pitch runs on graded frames: {} ({:.1f} per 90 min); length (s) {}".format(
        len(spans), len(spans) / minutes * 90, quantiles(lengths)))
    if len(lengths):
        print("    share of off-pitch time in runs >= 10 s: {}".format(
            pct(lengths[lengths >= 10].sum() / lengths.sum())))

    windows = ((-30, -20), (-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 15))
    print("  off-pitch share around SET PIECES tag starts, by Type (covered, graded frames)")
    print("    {:<24} {:>4}".format("type", "n") + "".join(
        "{:>10}".format("{}..{}s".format(a, b)) for a, b in windows))
    by_type = {}
    for a, _, kind in set_pieces():
        by_type.setdefault(kind, []).append(a)
    for kind in sorted(by_type, key=lambda k: (-len(by_type[k]), k)):
        cells = []
        for lo, hi in windows:
            mask = np.zeros(tac.n_frames, dtype=bool)
            for start in by_type[kind]:
                mask[max(0, start + lo * fps):max(0, start + hi * fps)] = True
            cells.append(pct(share(off, mask & covered & graded)))
        print("    {:<24} {:>4}".format(kind[:24], len(by_type[kind])) +
              "".join("{:>10}".format(c) for c in cells))

    if codes is None:
        print("  (predicted-label timing skipped: --method none)")
        return
    dead = (codes == pm.D) & graded
    bins = ((0, 5), (5, 15), (15, 30), (30, np.inf))
    print("  predicted D runs >= 2 s, at least 90% covered: with any off-pitch | mean off share | "
          "lag start -> first off (s) | last off -> end (s)")
    runs_d = [(a, b) for a, b in bc.runs(dead, 2 * fps) if covered[a:b].mean() >= 0.9]
    for lo, hi in bins:
        picked = [(a, b) for a, b in runs_d if lo * fps <= b - a < hi * fps]
        hit, frac, lag_in, lag_out = 0, [], [], []
        for a, b in picked:
            idx = np.flatnonzero(off[a:b])
            frac.append(len(idx) / (b - a))
            if len(idx):
                hit += 1
                lag_in.append(idx[0] / fps)
                lag_out.append((b - a - 1 - idx[-1]) / fps)
        print("    {:>3}-{:<4}s {:4d} runs  {}  |  {}  |  {}  |  {}".format(
            lo, hi if np.isfinite(hi) else "", len(picked),
            pct(hit / len(picked) if picked else np.nan),
            pct(float(np.mean(frac)) if frac else np.nan),
            quantiles(lag_in, (25, 50, 75)), quantiles(lag_out, (25, 50, 75))))

    live = np.isin(codes, (pm.H, pm.A)) & graded
    inside = [(a, b) for a, b in bc.runs(off & live, 2 * fps)]
    near = 2 * fps
    touching = [(a, b) for a, b in inside
                if dead[max(0, a - near):a].any() or dead[b:b + near].any()]
    alone = [(a, b) for a, b in inside if (a, b) not in set(touching)]
    for name, group in (("next to a predicted D run (edge timing)", touching),
                        ("isolated (a stoppage we may miss)", alone)):
        secs = np.array([b - a for a, b in group]) / fps
        print("  off-pitch runs >= 2 s inside predicted live, {}: {}, {:.0f} s; length {}".format(
            name, len(group), secs.sum(), quantiles(secs)))


def possessions(codes):
    return len(pm.possession_spells(codes, pm.valid_runs(codes)))


def section_merge(tac, bro, out, graded, method, codes):
    header("H  ball merge on BAR-ATH (broadcast.fuse_ball; label-free)")
    fused, stats = bc.fuse_ball(tac, bro, out["state"], out["bro_frame"])
    print("  candidate frames {}, rejected by the speed gate {}, dropped as short runs {}, "
          "filled {}".format(stats["candidates"], stats["speed_rejected"],
                             stats["short_dropped"], stats["filled"]))
    print("  ball seen on graded frames: tactical {}, merged {}".format(
        pct(share(tac.ball_seen, graded)), pct(share(fused.ball_seen, graded))))
    filled = fused.ball_seen & ~tac.ball_seen
    gap_table(tac, graded, filled)
    if codes is None:
        print("  (method comparison skipped: --method none)")
        return
    merged = predict(method, fused)
    per90 = graded.sum() / tac.fps / 60 / 90
    print("  {} on tactical -> merged input:".format(method))
    print("    D share {:.3f} -> {:.3f}; possessions per 90 {:.1f} -> {:.1f}".format(
        share(codes == pm.D, graded), share(merged == pm.D, graded),
        possessions(codes) / per90, possessions(merged) / per90))
    changed = (codes != merged) & graded
    print("    graded frames relabelled {}; of the filled frames {}; outside them {}".format(
        pct(share(changed, graded)), pct(share(changed, filled & graded)),
        pct(share(changed, graded & ~filled))))
    for a, b in ((pm.H, pm.A), (pm.A, pm.H), (pm.D, pm.H), (pm.D, pm.A), (pm.H, pm.D),
                 (pm.A, pm.D)):
        n = int((graded & (codes == a) & (merged == b)).sum())
        print("      {} -> {}: {:6d} frames".format(pm.LABELS[a], pm.LABELS[b], n))


def save(out, flip):
    folder = os.path.dirname(OUT_PATH)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    results = out["results"]
    np.savez(OUT_PATH, flip=flip, live_t=out["live_t"], bstate=out["bstate"],
             state=out["state"], bro_frame=out["bro_frame"], replay_frame=out["replay_frame"],
             stretch_start=np.array([r["start"] for r in results]),
             stretch_stop=np.array([r["stop"] for r in results]),
             stretch_d=np.array([r["d"] if r["d"] is not None else 0 for r in results]),
             stretch_kind=np.array(out["kinds"]))
    print("\nwrote {} (derived from the match data: gitignored, never commit)".format(
        os.path.relpath(OUT_PATH, ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default=None,
                        help="method whose predicted D splits section F (default: the submitted "
                             "one, solution/run.py METHOD); 'none' skips it")
    parser.add_argument("--refresh", action="store_true", help="re-parse the broadcast ATD")
    args = parser.parse_args()
    if args.method is None:
        from solution import run
        args.method = run.METHOD

    tac = atd.load()
    bro = atd.load_broadcast(args.refresh)
    _, _, valid = load_homography.load(source="broadcast")
    if len(valid) != bro.n_frames:
        raise SystemExit("broadcast homography has {} rows, ATD {} frames".format(
            len(valid), bro.n_frames))
    graded = tac.in_play_mask()

    section_files(bro, valid)
    flip = section_orientation(tac, bro, valid)
    bro = bc.flipped(bro, flip)
    out = bc.align(tac, bro, valid)
    save(out, flip)
    section_alignment(tac, bro, valid, out)
    section_coverage(tac, out, graded)
    section_ball(tac, bro, out, graded)
    codes = None if args.method == "none" else predict(args.method, tac)
    section_cutaway(tac, out, graded, args.method, codes)
    section_timing(tac, out, graded, codes)
    section_merge(tac, bro, out, graded, args.method, codes)


if __name__ == "__main__":
    main()
