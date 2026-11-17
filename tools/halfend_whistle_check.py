"""D14's checks for `v7_end`: the end-of-half rule on the detected whistles, alone (D48).

    .venv/bin/python tools/halfend_whistle_check.py              # proxy: plain halves + walk-off stress
    .venv/bin/python tools/halfend_whistle_check.py --bar-ath    # (needs data/) label-free aggregates
    make halfend-whistle [ARGS="--bar-ath"]

`v7_end` is `v7_keep2_control` with whistlecue.end_of_half and its tracking guard on top, and no
in-play cue, so its decode is v7's and only the end of a half can change. D14 adopts such a rule
without D4's 0.02 when it fixes a documented failure the proxy cannot represent (the walk-off
after the whistle, EXPERIMENTS section 20) and passes the checks below. Settings and pass
criteria fixed in D48 before the first run.

Proxy (IDSSE dev x SEEDS, simulator v1 + D12 ball stress, keep2 truth; whistles imitated by
whistlecue.synthetic for each VARIANT; v7_end's rule applied to v7_keep2_control's decode, which
is what the method computes -- checked once against the method itself):
  P1. Without whistles (the plain proxy: IDSSE has no audio) nothing changes.
  P2. Plain halves with imitated whistles: no false fire anywhere, i.e. no half where the rule
      turns FALSE_FIRE_S or more of truly live frames into D.
  P3. Walk-off stress (simulate.add_walkoff, WALKOFF_S): not WORSE for any variant (mean gain
      under 0 with at most 1 seed of 5 winning), better for at least one (mean gain over 0 with at
      least 4 of 5 winning).
  Also reported: the same without the guard.
BAR-ATH (--bar-ath; whistles from the self-calibrating detector, D47; aggregates only):
  B1. The rule fires in at least one half.
  B2. In each half where it fires: v7_keep2_control decodes at least one possession after the
      pick, and, where at least 10 s follow the pick, fewer players are inside the pitch after it
      than in the 30 s before it (medians).
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import whistlecue  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.simulate import add_ball_noise, add_walkoff, degrade  # noqa: E402
from tools.proxy_league import f1_from_flags, spell_flags  # noqa: E402

STRESS = (0.05, 3.4, 0.26)                 # D12: sd m, jumps per visible minute, smoothing s
SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ("prior", "matched", "pessimistic")
WALKOFF_S = (30.0, 90.0)
FALSE_FIRE_S = 1.0
REFERENCE = "v7_keep2_control"


def build(match_id, seed, walkoff):
    from solution.sources import idsse
    raw = idsse.load(match_id, labels="keep2")
    clean = add_walkoff(raw, walkoff, seed) if walkoff else raw
    truth = (add_walkoff(raw, walkoff, 0) if walkoff else raw).labels
    assert np.array_equal(clean.labels, truth)
    match = add_ball_noise(degrade(clean, seed), *STRESS[:2], seed=seed, smooth_s=STRESS[2])
    return match, truth


def pooled(rows, key):
    """Per-seed pooled spellF1 of rows[...][key], as the proxy league pools."""
    out = []
    for seed in sorted(set(r["seed"] for r in rows)):
        parts, n_pred = [], 0
        for r in rows:
            if r["seed"] == seed:
                flags, n = spell_flags(r["truth"], r[key])
                parts.append(flags)
                n_pred += n
        out.append(f1_from_flags(np.concatenate(parts, axis=1), n_pred))
    return np.array(out)


def false_fires(match, truth, pred, ended):
    """Halves where the rule turned FALSE_FIRE_S or more of truly live frames into D."""
    count = 0
    for a, b in match.halves:
        changed = (ended[a:b] != pred[a:b]) & np.isin(truth[a:b], (pm.H, pm.A))
        count += changed.sum() >= FALSE_FIRE_S * match.fps
    return int(count)


def run_proxy(seeds):
    from solution.sources import idsse
    checked = False
    verdict = {"P1": True, "P2": True}
    p3 = []
    for walkoff in (0.0,) + WALKOFF_S:
        rows = []
        for match_id in idsse.DEV_IDS:
            for seed in seeds:
                match, truth = build(match_id, seed, walkoff)
                pred = METHODS[REFERENCE](match)
                row = {"seed": seed, "truth": truth, "as is": pred, "fires": {}, "false": {}}
                none = whistlecue.end_of_half(pred, match, [], pm.D, guard=True)
                verdict["P1"] &= bool(np.array_equal(none, pred))
                for variant in VARIANTS:
                    onsets = whistlecue.synthetic(match, seed, variant)
                    for guard in (True, False):
                        key = (variant, guard)
                        ended = whistlecue.end_of_half(pred, match, onsets, pm.D, guard=guard)
                        row[key] = ended
                        row["fires"][key] = sum(
                            r["fires"] for r in whistlecue.end_picks(match, onsets, guard))
                        row["false"][key] = false_fires(match, truth, pred, ended)
                        if not checked and variant == "matched" and guard:
                            method = METHODS["v7_end"](match)
                            assert np.array_equal(method, ended), "v7_end differs from the rule"
                            checked = True
                rows.append(row)
        base = pooled(rows, "as is")
        title = "plain halves" if not walkoff else "walk-off stress, {:.0f} s".format(walkoff)
        print("\n== {} ==  IDSSE dev x seeds {}, v1 + D12, keep2; {} as is {:.4f}".format(
            title, list(seeds), REFERENCE, base.mean()))
        print("  {:<12} {:<6} | gain    | wins | halves fired | false fires".format(
            "whistles", "guard"))
        for variant in VARIANTS:
            for guard in (True, False):
                key = (variant, guard)
                score = pooled(rows, key)
                gain = float((score - base).mean())
                wins = int(np.sum(score > base))
                fired = sum(r["fires"][key] for r in rows)
                false = sum(r["false"][key] for r in rows)
                print("  {:<12} {:<6} | {:+.4f} | {}/{}  | {:12d} | {:11d}".format(
                    variant, "on" if guard else "off", gain, wins, len(seeds), fired, false))
                if guard and not walkoff:
                    verdict["P2"] &= false == 0
                if guard and walkoff:
                    p3.append((gain, wins))
    n = len(seeds)
    worse = any(g < 0 and w <= n // 5 for g, w in p3)
    better = any(g > 0 and w >= n - n // 5 for g, w in p3)
    verdict["P3"] = not worse and better
    print("\nP1 (no whistles, no change): {}".format("pass" if verdict["P1"] else "FAIL"))
    print("P2 (no false fire on plain halves, guard on): {}".format(
        "pass" if verdict["P2"] else "FAIL"))
    print("P3 (walk-off: never WORSE, better for at least one variant): {}".format(
        "pass" if verdict["P3"] else "FAIL"))
    return verdict


def run_bar_ath():
    import games
    from solution import whistle
    from solution.sources import atd
    from tools.halfend_check import inside_counts
    match = atd.load()
    fps = match.fps
    features = whistle.peak_video_features(games.video_path(games.GAME), match.n_frames, fps)
    _, found = whistle.match_whistles(features, match)
    onsets = whistlecue.group_onsets(found, fps)
    pred = METHODS[REFERENCE](match)
    _, inside = inside_counts(match)
    picks = whistlecue.end_picks(match, onsets, guard=True)
    print("\n== BAR-ATH (label-free; aggregates only) ==")
    fired_any, b2 = False, True
    for h, ((a, b), row) in enumerate(zip(match.halves, picks)):
        if row["pick"] is None:
            print("  half {}: no whistle after 45:00".format(h + 1))
            continue
        p = row["pick"]
        alive = np.isin(pred[p:b], (pm.H, pm.A)).astype(np.int8)
        starts = int(alive[0] + np.sum(np.diff(alive) == 1)) if len(alive) else 0
        line = ("  half {}: pick {:.1f} s before the half's end; median collective speed after it "
                "{} m/s (guard {}); fires: {}".format(
                    h + 1, row["before_s"],
                    "n/a" if row["speed"] is None else "{:.2f}".format(row["speed"]),
                    whistlecue.END_GUARD_MS, "yes" if row["fires"] else "no"))
        print(line)
        print("     v7 after the pick: {} possessions started, H/A share {:.2f}".format(
            starts, float(alive.mean()) if len(alive) else 0.0))
        before = inside[max(a, p - 30 * fps):p]
        after = inside[p:b]
        if b - p >= 10 * fps:
            fewer = np.median(after) < np.median(before)
            print("     players inside the pitch: {:.0f} in the 30 s before, {:.0f} after "
                  "(median)".format(np.median(before), np.median(after)))
        else:
            fewer = None
            print("     players inside the pitch: under 10 s after the pick, not tested")
        if row["fires"]:
            fired_any = True
            b2 &= starts >= 1 and fewer is not False
    print("\nB1 (fires in at least one half): {}".format("pass" if fired_any else "FAIL"))
    print("B2 (where it fires: possessions removed, players leaving): {}".format(
        "pass" if b2 else "FAIL"))
    return {"B1": fired_any, "B2": b2}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar-ath", action="store_true", help="BAR-ATH aggregates (needs data/)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    verdict = run_bar_ath() if args.bar_ath else run_proxy(args.seeds)
    print("Verdict (D48): {}".format("all pass" if all(verdict.values()) else "FAILS"))


if __name__ == "__main__":
    main()
