"""Each part of a decoder method on its own: what every "voter" says per frame, and how often it is right.

    .venv/bin/python tools/voters.py                       # proxy: the 5 IDSSE dev matches, keep2 + D12
    .venv/bin/python tools/voters.py --bar-ath             # (needs data/) BAR-ATH's public 5 minutes
    .venv/bin/python tools/voters.py --bar-ath --frames-csv   # also the per-frame table, for a person
    .venv/bin/python tools/voters.py --bar-ath --blocks-csv   # one row per 5-frame block, whole match
    make voters [ARGS="--bar-ath --frames-csv"]

The decoder does not take a vote. It adds each part's log-likelihood per 5-frame block and picks
the best segmentation under the duration priors. A part's "own call" here is what it would say
alone, frame by frame:
  - nearest:  team of the tracked player nearest the ball (no call without a ball or a player);
  - holder:   the touch engine's holding team, after the control model drops touches (v7);
  - spread:   gapteam's spread cue, its sign (home spread minus away spread);
  - frame:    the frame model, home if P(home | alive) > 0.5 (the decoder reads it where the ball
              is seen; scored on every frame here, and on seen frames as its own row);
  - deadmodel: the dead-ball model, D if P(dead) > 0.5, else "alive" (it has no team);
  - kickoff:  D before each detected kickoff, no call after it;
  - final:    the method's decoded label.
The keep2 duration priors are part of the method but make no per-frame call.

Team voters are scored where the truth is H or A and they make a call: team accuracy, split by
ball seen or not. The dead-ball voters are scored on dead/alive. Two counts of each voter against
the final label say whether it carries information the decoder throws away: "right where final
is wrong" and "wrong where final is right", as shares of the frames both are scored on.

Prints aggregates only. `--frames-csv` (BAR-ATH) writes evaluation/reports/voters_<method>.csv,
gitignored: one row per graded frame of the public window, for a person to read against the video.
`--blocks-csv` writes evaluation/reports/voters_blocks_<method>.csv: one row per 5-frame block of
the whole match, on the decoder's own blocks (from each half's kickoff span), with each voter's
majority call over the block ("-" none speaks, "?" a tie) and truth X outside the public window.
Methods with the whistle cue (v7_whistle, v7_whistle_end, ...) get its columns at the end.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution import deadmodel, framemodel, gapteam  # noqa: E402
from solution.methods import METHODS  # noqa: E402

FPS = 25
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
NONE = -1
TEAM_VOTERS = ("nearest", "holder", "spread", "frame", "frame_seen")
DEAD_VOTERS = ("deadmodel", "kickoff")
LETTER = {pm.H: "H", pm.A: "A", pm.D: "D", NONE: "-"}


def own_calls(method, match):
    """{voter: (T,) int8 codes (pm.H / pm.A / pm.D, NONE for no call)}, the final codes, the
    ball-seen mask, and {name: (T,) raw number behind the calls} plus "spans" (decoded spans)."""
    inp = method.prepare(match)
    n = match.n_frames
    calls = {}
    calls["nearest"] = np.where(inp["nearest"] == 0, pm.H,
                                np.where(inp["nearest"] == 1, pm.A, NONE)).astype(np.int8)
    holder = inp["holder"]
    calls["holder"] = (np.full(n, NONE, np.int8) if holder is None else
                       np.where(holder == 0, pm.H, np.where(holder == 1, pm.A, NONE)).astype(np.int8))
    spread = gapteam.spread(match)
    calls["spread"] = np.where(spread > 0, pm.H, np.where(spread < 0, pm.A, NONE)).astype(np.int8)
    odds = inp["frame"]
    frame = np.full(n, NONE, np.int8)
    if odds is not None:
        frame = np.where(odds > 0, pm.H, np.where(odds < 0, pm.A, NONE)).astype(np.int8)
    calls["frame"] = frame
    calls["frame_seen"] = np.where(match.ball_seen, frame, NONE).astype(np.int8)
    p_dead = deadmodel.dead_probability(match)
    calls["deadmodel"] = np.where(np.isfinite(p_dead),
                                  np.where(p_dead > 0.5, pm.D, pm.H), NONE).astype(np.int8)
    # "alive" is written as H above only as a placeholder; scoring reads dead/alive.
    kick = np.full(n, NONE, np.int8)
    for (start, _), (span_start, _) in zip(match.halves, inp["spans"]):
        kick[start:span_start] = pm.D
    calls["kickoff"] = kick
    final = method.decode(match, inp)
    final = np.array([pm.H, pm.A, pm.D], dtype=np.int8)[final]
    raw = {"nearest": inp["nearest"], "holder": (np.full(n, -1, np.int8) if holder is None else holder),
           "spread": spread, "frame": (np.full(n, np.nan) if odds is None else odds),
           "p_dead": p_dead, "spans": inp["spans"]}
    keeper_ll = inp.get("keeper")
    if keeper_ll is not None:
        # the keeper model (keeper_w > 0): its best state where the frame is in the keeper zone
        zone = np.isfinite(keeper_ll).all(axis=1)
        best = np.argmax(np.where(zone[:, None], keeper_ll, -np.inf), axis=1)
        order = np.array([pm.H, pm.A, pm.D], dtype=np.int8)
        calls["keeper"] = np.where(zone, order[best], NONE).astype(np.int8)
        raw["keeper"] = keeper_ll
    whistle_ll = inp.get("whistle")
    if whistle_ll is not None:
        # the whistle cue (whistle_w > 0): D where it leans dead, "alive" (H placeholder) where
        # it leans live; its log ratio and the whistle onsets for the block columns
        ratio = whistle_ll[:, 2] - whistle_ll[:, 0]
        calls["whistle"] = np.where(np.isfinite(ratio), np.where(ratio > 0, pm.D, pm.H),
                                    NONE).astype(np.int8)
        raw["whistle_ratio"] = ratio
    if inp.get("whistle_onsets") is not None:
        raw["whistle_onsets"] = np.sort(np.asarray(inp["whistle_onsets"], dtype=np.int64))
    return calls, final, match.ball_seen.copy(), raw


class Tally:
    """Counts summed over matches, so the shares are pooled over frames."""

    def __init__(self):
        self.c = {}

    def add(self, key, value):
        self.c[key] = self.c.get(key, 0) + int(value)

    def share(self, num, den):
        d = self.c.get(den, 0)
        return float("nan") if d == 0 else self.c.get(num, 0) / d


def score(tally, calls, final, seen, truth, graded):
    alive_truth = graded & np.isin(truth, (pm.H, pm.A))
    final_team_right = final == truth
    final_alive = np.isin(final, (pm.H, pm.A))
    tally.add("graded", graded.sum())
    tally.add("final_right", (graded & (final == truth)).sum())
    tally.add("alive", alive_truth.sum())
    tally.add("final_team_right", (alive_truth & final_team_right).sum())
    tally.add("final_alive_on_alive", (alive_truth & final_alive).sum())
    tally.add("final_team_right_both_alive", (alive_truth & final_alive & final_team_right).sum())
    for tag, mask in (("seen", seen), ("gap", ~seen)):
        tally.add("final:speaks_" + tag, (alive_truth & mask).sum())
        tally.add("final:right_" + tag, (alive_truth & mask & final_team_right).sum())
    final_other_team = alive_truth & final_alive & ~final_team_right
    for name in TEAM_VOTERS:
        v = calls[name]
        speaks = alive_truth & (v != NONE)
        right = speaks & (v == truth)
        tally.add(name + ":speaks_graded", (graded & (v != NONE)).sum())
        tally.add(name + ":speaks", speaks.sum())
        tally.add(name + ":right", right.sum())
        for tag, mask in (("seen", seen), ("gap", ~seen)):
            tally.add(name + ":speaks_" + tag, (speaks & mask).sum())
            tally.add(name + ":right_" + tag, (right & mask).sum())
        tally.add(name + ":rescue", (right & ~final_team_right).sum())
        tally.add(name + ":rescue_team", (right & final_other_team).sum())
        tally.add(name + ":mislead", (speaks & ~right & final_team_right).sum())
    dead_truth = graded & (truth == pm.D)
    final_dead_right = (final == pm.D) == (truth == pm.D)
    tally.add("dead", dead_truth.sum())
    tally.add("final_dead_hit", (dead_truth & (final == pm.D)).sum())
    tally.add("final_dead_calls", (graded & (final == pm.D)).sum())
    tally.add("final_dead_right", (graded & final_dead_right).sum())
    for name in DEAD_VOTERS:
        v = calls[name]
        speaks = graded & (v != NONE)
        says_dead = speaks & (v == pm.D)
        right = speaks & ((v == pm.D) == (truth == pm.D))
        tally.add(name + ":speaks", speaks.sum())
        tally.add(name + ":right", right.sum())
        tally.add(name + ":dead_calls", says_dead.sum())
        tally.add(name + ":dead_hit", (says_dead & (truth == pm.D)).sum())
        tally.add(name + ":dead_seen", (speaks & dead_truth).sum())
        tally.add(name + ":dead_found", (says_dead & dead_truth).sum())
        tally.add(name + ":rescue", (right & ~final_dead_right).sum())
        tally.add(name + ":mislead", (speaks & ~right & final_dead_right).sum())


def pct(x):
    return "   --" if not np.isfinite(x) else "{:5.1f}".format(100.0 * x)


def report(t, title):
    print("\n" + title)
    print("{} graded frames; final label right on {}%, team right on {}% of the frames where the "
          "truth is H or A (on {}% where it also says H or A); dead ball found {}%, its D calls "
          "right {}%".format(t.c.get("graded", 0), pct(t.share("final_right", "graded")).strip(),
                             pct(t.share("final_team_right", "alive")).strip(),
                             pct(t.share("final_team_right_both_alive",
                                         "final_alive_on_alive")).strip(),
                             pct(t.share("final_dead_hit", "dead")).strip(),
                             pct(t.share("final_dead_hit", "final_dead_calls")).strip()))
    print("\nTeam voters, on frames where the truth is H or A (%):")
    row = "{:<11} {:>7} {:>8} {:>9} {:>8} {:>9} {:>10} {:>9}"
    print(row.format("voter", "speaks", "right", "right", "right", "rescues", "of which", "misleads"))
    print(row.format("", "", "", "ball seen", "in gaps", "final", "wrong team", "final"))
    for name in TEAM_VOTERS:
        print(row.format(
            name, pct(t.share(name + ":speaks", "alive")), pct(t.share(name + ":right", name + ":speaks")),
            pct(t.share(name + ":right_seen", name + ":speaks_seen")),
            pct(t.share(name + ":right_gap", name + ":speaks_gap")),
            pct(t.share(name + ":rescue", name + ":speaks")),
            pct(t.share(name + ":rescue_team", name + ":speaks")),
            pct(t.share(name + ":mislead", name + ":speaks"))))
    print(row.format("final", pct(1.0), pct(t.share("final_team_right", "alive")),
                     pct(t.share("final:right_seen", "final:speaks_seen")),
                     pct(t.share("final:right_gap", "final:speaks_gap")), "", "", ""))
    print("(final counts a D call on an alive frame as wrong; 'of which wrong team' leaves those out)")
    print("\nDead-ball voters, on every graded frame they speak on (%):")
    print("{:<11} {:>7} {:>9} {:>10} {:>10} {:>9} {:>9}".format(
        "voter", "speaks", "dead/alive", "D found", "D calls", "rescues", "misleads"))
    print("{:<11} {:>7} {:>9} {:>10} {:>10} {:>9} {:>9}".format(
        "", "", "right", "(recall)", "right", "final", "final"))
    for name in DEAD_VOTERS:
        print("{:<11} {:>7} {:>9} {:>10} {:>10} {:>9} {:>9}".format(
            name, pct(t.share(name + ":speaks", "graded")), pct(t.share(name + ":right", name + ":speaks")),
            pct(t.share(name + ":dead_found", name + ":dead_seen")),
            pct(t.share(name + ":dead_hit", name + ":dead_calls")),
            pct(t.share(name + ":rescue", name + ":speaks")),
            pct(t.share(name + ":mislead", name + ":speaks"))))
    print("{:<11} {:>7} {:>9} {:>10} {:>10}".format(
        "final", pct(1.0), pct(t.share("final_dead_right", "graded")),
        pct(t.share("final_dead_hit", "dead")), pct(t.share("final_dead_hit", "final_dead_calls"))))
    print("\nrescues = voter right where the final label is wrong; misleads = voter wrong where the "
          "final label is right; both as a share of the frames the voter speaks on.")


def write_frames(path, calls, raw, final, seen, truth, graded):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = np.flatnonzero(graded)

    def num(x, digits=3):
        return "" if not np.isfinite(x) else round(float(x), digits)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "video_time", "ball_seen", "truth", "final"] + list(TEAM_VOTERS[:4])
                   + ["deadmodel", "kickoff", "spread_m", "frame_log_odds", "p_dead"])
        for f in frames:
            m, s = divmod(f / float(FPS), 60.0)
            dead = calls["deadmodel"][f]
            w.writerow([int(f), "{:d}:{:05.2f}".format(int(m), s), int(seen[f]),
                        LETTER[int(truth[f])], LETTER[int(final[f])]]
                       + [LETTER[int(calls[k][f])] for k in TEAM_VOTERS[:4]]
                       + ["D" if dead == pm.D else "alive" if dead != NONE else "-",
                          LETTER[int(calls["kickoff"][f])],
                          num(raw["spread"][f], 2), num(raw["frame"][f]), num(raw["p_dead"][f])])


BLOCK_VOTERS = TEAM_VOTERS + DEAD_VOTERS        # the seven voters, one column each


def block_starts(n, spans, block=5):
    """Start frame of every block over frames [0, n): the decoder's blocks of `block` frames from
    each span's start, and blocks of the same size from 0 or a span's end over the frames
    outside the spans (they are D before the kickoff and not decoded)."""
    starts, edges = [], sorted({0, n} | {int(s) for span in spans for s in span})
    for a, b in zip(edges[:-1], edges[1:]):
        starts.extend(range(a, b, block))
    return np.array(starts, dtype=int)


def majority(values):
    """Letter of the most frequent call in a block, ignoring "-" (no call); "-" if none speaks,
    "?" on a tie."""
    letters = [LETTER[int(v)] for v in values if v != NONE]
    if not letters:
        return "-"
    counts = {k: letters.count(k) for k in set(letters)}
    best = max(counts.values())
    top = [k for k, c in counts.items() if c == best]
    return top[0] if len(top) == 1 else "?"


def mean_of(values, digits=3):
    """Mean of the finite values, rounded; "" when there are none."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return "" if values.size == 0 else round(float(values.mean()), digits)


def block_rows(calls, raw, final, seen, truth, spans, fps=FPS, reference=None):
    """One dict per block: frames, time, ball seen count, truth, final, and per voter its
    majority call plus the numbers behind it, as the decoder reads them per block:
      nearest_home / nearest_away, holder_home / holder_away: frames of the block on each team;
      spread_m: mean home spread minus away spread (metres); frame_log_odds: mean log P(home) /
      P(away), frame_p_home: mean P(home); frame_seen_log_odds: the same over ball-seen frames;
      p_dead: mean P(dead). kickoff has no number (D before the detected kickoff).
    With the keeper model (keeper_w > 0): keeper, its majority call in the keeper zone ("-"
    outside), keeper_frames (zone frames in the block), keeper_ll_H / _A / _D (mean log P(x |
    state) up to a constant; the gaps between them are what the decoder adds). With the whistle
    (whistle_w > 0 or whistle_end), after everything else: whistle, the cue's majority call ("D"
    leans dead, "alive" leans live, "-" unknown), whistle_log_ratio (mean log P(cue | dead) /
    P(cue | alive), what the decoder adds to D), whistle_onset (1 if a whistle starts in the
    block) and since_whistle_s (seconds from the last whistle to the block's start). With `reference`
    ((T,) codes of another method): ref_final, its majority, and changed (1 where it differs)."""
    n = len(final)
    starts = block_starts(n, spans)
    stops = np.append(starts[1:], n)
    inside = np.zeros(n, bool)
    for a, b in spans:
        inside[a:b] = True
    odds = np.asarray(raw["frame"], dtype=float)
    with np.errstate(over="ignore"):
        p_home = 1.0 / (1.0 + np.exp(-odds))
    rows = []
    for a, b in zip(starts, stops):
        m, s = divmod(a / float(fps), 60.0)
        cut = slice(a, b)
        sighted = seen[cut]
        row = {"block": len(rows), "frame_from": int(a), "frame_to": int(b - 1),
               "video_time": "{:d}:{:05.2f}".format(int(m), s), "decoded": int(inside[a]),
               "ball_seen": int(sighted.sum()),
               "truth": (majority(truth[cut][truth[cut] != pm.X]).replace("-", "X")
                         if truth is not None else ""),
               "final": majority(final[cut])}
        for name in BLOCK_VOTERS:
            call = majority(calls[name][cut])
            row[name] = ("alive" if call == "H" else call) if name == "deadmodel" else call
            if name in ("nearest", "holder"):
                row[name + "_home"] = int((raw[name][cut] == 0).sum())
                row[name + "_away"] = int((raw[name][cut] == 1).sum())
            elif name == "spread":
                row["spread_m"] = mean_of(raw["spread"][cut], 2)
            elif name == "frame":
                row["frame_log_odds"] = mean_of(odds[cut])
                row["frame_p_home"] = mean_of(p_home[cut])
            elif name == "frame_seen":
                row["frame_seen_log_odds"] = mean_of(odds[cut][sighted])
            elif name == "deadmodel":
                row["p_dead"] = mean_of(raw["p_dead"][cut])
        if "keeper" in calls:
            ll = raw["keeper"][cut]
            zone = np.isfinite(ll).all(axis=1)
            row["keeper"] = majority(calls["keeper"][cut])
            row["keeper_frames"] = int(zone.sum())
            for k, letter in enumerate("HAD"):
                row["keeper_ll_" + letter] = mean_of(ll[zone, k], 2) if zone.any() else ""
        if reference is not None:
            row["ref_final"] = majority(reference[cut])
            row["changed"] = int(row["ref_final"] != row["final"])
        if "whistle" in calls:
            call = majority(calls["whistle"][cut])
            row["whistle"] = "alive" if call == "H" else call
            row["whistle_log_ratio"] = mean_of(raw["whistle_ratio"][cut], 2)
        if "whistle_onsets" in raw:
            onsets = raw["whistle_onsets"]
            row["whistle_onset"] = int(((onsets >= a) & (onsets < b)).any())
            last = onsets[onsets <= a]
            row["since_whistle_s"] = ("{:.1f}".format((a - last[-1]) / float(fps))
                                      if len(last) else "")
        rows.append(row)
    return rows


def write_blocks(path, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="v7_keep2_control", choices=sorted(METHODS))
    parser.add_argument("--bar-ath", action="store_true",
                        help="score on BAR-ATH's public 5 minutes (needs data/); default: proxy")
    parser.add_argument("--frames-csv", action="store_true",
                        help="with --bar-ath: write the per-frame table to evaluation/reports/")
    parser.add_argument("--blocks-csv", action="store_true",
                        help="with --bar-ath: write one row per 5-frame block of the whole match "
                             "(final + the seven voters) to evaluation/reports/")
    parser.add_argument("--compare", default=None, choices=sorted(METHODS),
                        help="with --blocks-csv: add another method's label per block (ref_final) "
                             "and whether it differs (changed)")
    parser.add_argument("--seed", type=int, default=0, help="proxy: simulator seed")
    args = parser.parse_args(argv)
    method = METHODS[args.method]
    tally = Tally()

    if args.bar_ath:
        from solution.sources import atd
        match = atd.load()
        truth = pm.label_codes(pm.read_label_csv(GT_PUBLIC))
        graded = truth != pm.X
        calls, final, seen, raw = own_calls(method, match)
        score(tally, calls, final, seen, truth, graded)
        report(tally, "{} on BAR-ATH, public window (17 possessions: a description, never a "
                      "tuning signal)".format(args.method))
        if args.frames_csv:
            path = os.path.join(ROOT, "evaluation", "reports", "voters_{}.csv".format(args.method))
            write_frames(path, calls, raw, final, seen, truth, graded)
            print("\nwrote {}".format(os.path.relpath(path, ROOT)))
        if args.blocks_csv:
            reference = None
            if args.compare:
                ref = np.asarray(METHODS[args.compare](match))
                reference = np.where(np.isin(ref, (pm.H, pm.A, pm.D)), ref, NONE).astype(np.int8)
            rows = block_rows(calls, raw, final, seen, truth, raw["spans"], reference=reference)
            path = os.path.join(ROOT, "evaluation", "reports",
                                "voters_blocks_{}.csv".format(args.method))
            write_blocks(path, rows)
            print("\nwrote {} ({} blocks)".format(os.path.relpath(path, ROOT), len(rows)))
            if reference is not None:
                changed = sum(r["changed"] for r in rows)
                print("blocks whose label differs from {}: {} of {} ({:.1%})".format(
                    args.compare, changed, len(rows), changed / float(len(rows))))
        return

    from solution.simulate import add_ball_noise, degrade
    from solution.sources import idsse
    for match_id in idsse.DEV_IDS:
        clean = idsse.load(match_id, labels="keep2")
        match = add_ball_noise(degrade(clean, args.seed), 0.05, 3.4, args.seed, smooth_s=0.26)
        calls, final, seen, raw = own_calls(method, match)
        score(tally, calls, final, seen, clean.labels, clean.in_play_mask())
    report(tally, "{} on the proxy: {} IDSSE dev matches, simulator seed {}, keep2 truth, D12 "
                  "ball stress".format(args.method, len(idsse.DEV_IDS), args.seed))


if __name__ == "__main__":
    main()
