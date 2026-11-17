"""The proxy league: score methods on external matches with the official scorer.

    .venv/bin/python tools/proxy_league.py                                  # baseline_smooth50
    .venv/bin/python tools/proxy_league.py --methods baseline_smooth50 baseline_smooth125
    .venv/bin/python tools/proxy_league.py --methods oracle --clean         # sanity: must be 1.0
    make proxy METHODS="baseline_smooth50 baseline_smooth125"
    make proxy METHODS="baseline_smooth50" SIM=v0                         # the older simulator
    make proxy METHODS="v1_decoder" BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26  # BAR-ATH-like ball
    make proxy METHODS="v5_frame_model" BALL_VISIBLE=0.9                 # better-tracker what-if
    make proxy METHODS="v1_decoder" BOUNDARIES=1                          # tools/boundaries.py
    make proxy METHODS="v2_touch v3_kickoff" PREKICKOFF=117 [PREKICKOFF_PROFILE=busy]
    make proxy METHODS="..." BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26 BEYOND_KEEP=0.82 BEYOND_MARGIN=1
    make proxy METHODS="v4_spread new" LABELS=dfl         # DFL first-touch truth (D21 check)

Why it exists (PLAN.md, section 6): the public 5 minutes hold 17 possessions and may not be tuned
on, and the full-match truth is secret. External matches with corrected possession labels give
about 2,200 possessions (five development matches) to decide on instead; two more are held out. Each match is degraded by solution/simulate.py so
the ball goes missing the way it does in the BAR-ATH ATD, then scored by the unmodified
evaluation/possession_metrics.evaluate -- the function that produces the official result.

What it prints
--------------
- per match: spellF1 and macroF1, averaged over the simulator seeds, plus possession counts;
- pooled: all matches scored as one long ground truth, by summing each match's matched and
  predicted possessions (identical to laying the matches end to end, since no possession crosses
  a match). This is the number to decide on, with a bootstrap interval over true possessions;
- with two or more methods: each one's pooled gain over the FIRST method listed, with a paired
  bootstrap interval. Paired means both methods are judged on the same resampled possessions,
  which removes the shared noise; an interval that excludes 0 is a real gap.

Decision rule (PLAN.md): adopt a change only if the pooled gain is at least 0.02, its paired
interval excludes 0, and it wins on at least two thirds of the matches.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import relabel  # noqa: E402
from solution.match import CACHE_DIR, pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.simulate import (DEFAULT_VERSION, TRAIN_SIM_ENV, VERSIONS, add_ball_noise,
                               train_version,  # noqa: E402
                               PREKICKOFF_PROFILES, add_position_noise, add_prekickoff,
                               degrade, retime, scale_speeds, thin_beyond)
from solution.sources import idsse  # noqa: E402
from tools import boundaries  # noqa: E402

OUT_DIR = os.path.join(CACHE_DIR, "proxy_league")


def spell_flags(gt, pred):
    """(flags[iou, true_spell], n_pred) for one match, mirroring possession_metrics exactly.

    flags says which true possessions were matched at each IoU threshold. Pooling sums these
    counts across matches. That equals scoring all matches laid end to end, because a possession
    never crosses a match boundary, but it avoids comparing every possession of one match with
    every possession of another.
    """
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), t) for t in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def f1_from_flags(flags, n_pred, picks=None):
    """Mean F1 over the IoU thresholds; `picks` resamples true possessions for a bootstrap."""
    if picks is not None:
        flags = flags[:, picks]
    n_gt = flags.shape[1]
    scores = []
    for row in flags:
        tp = int(row.sum())
        fp = max(0, n_pred - tp)   # same clamp as possession_metrics.bootstrap_spell_f1
        scores.append(2.0 * tp / (2 * tp + fp + (n_gt - tp)) if tp else 0.0)
    return float(np.mean(scores))


def bootstrap(flags, n_pred, draws, seed=0, other=None):
    """5-95% interval of pooled spellF1, or of other - self when `other` = (flags, n_pred)."""
    rng = np.random.RandomState(seed)
    n_gt = flags.shape[1]
    values = np.empty(draws)
    for draw in range(draws):
        picks = rng.randint(0, n_gt, n_gt)
        value = f1_from_flags(flags, n_pred, picks)
        if other is not None:
            value = f1_from_flags(other[0], other[1], picks) - value
        values[draw] = value
    low, high = np.percentile(values, pm.BOOTSTRAP_PERCENTILES)
    return float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["baseline_smooth50"],
                        choices=sorted(METHODS), help="the first one is the reference")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev",
                        help="dev = the five matches we decide on (default); holdout = the two "
                             "locked away until Phase 4 -- do not tune on them")
    parser.add_argument("--matches", nargs="+", help="explicit IDSSE match ids, overrides --split")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="simulator seeds; scores are averaged over them")
    parser.add_argument("--simulator", choices=VERSIONS, default=DEFAULT_VERSION,
                        help="simulator version (solution/simulate.py); numbers from different "
                             "versions are not comparable")
    parser.add_argument("--train-simulator", choices=VERSIONS, default=None,
                        help="simulator for the IDSSE matches the trained models (frame, "
                             "dead-ball, control) learn from; default {} "
                             "(simulate.train_version)".format(DEFAULT_VERSION))
    parser.add_argument("--position-noise", type=float, default=0.0, metavar="SD",
                        help="stress test: add white noise of SD metres to player positions "
                             "after degrading (default 0, off); not part of any simulator "
                             "version, so never mix these numbers with plain ones")
    parser.add_argument("--speed-factor", type=float, default=1.0, metavar="F",
                        help="stress test: shrink all positions towards the centre so every "
                             "speed is F times slower or faster (default 1, off)")
    parser.add_argument("--tempo", type=float, default=1.0, metavar="F",
                        help="stress test: replay each clean match F times as fast before "
                             "degrading (simulate.retime): every speed x F, positions "
                             "unchanged, truth retimed with it (default 1, off). 0.9 is the D9 "
                             "tempo check for methods that read positions (EXPERIMENTS 16)")
    parser.add_argument("--ball-noise", type=float, default=0.0, metavar="SD",
                        help="stress test: white jitter of SD metres on the ball "
                             "(simulate.add_ball_noise; default 0, off). 0.05 matches the "
                             "BAR-ATH median ball speed")
    parser.add_argument("--ball-jumps", type=float, default=0.0, metavar="PER_MIN",
                        help="stress test: 1.6-4.4 m ball hops, counted as jumps per minute of "
                             "visible ball (default 0, off); BAR-ATH has about 3.4")
    parser.add_argument("--ball-smooth", type=float, default=0.0, metavar="S",
                        help="stress test: Gaussian smoothing of the ball over S seconds (sd), "
                             "after the jitter (default 0, off); 0.26 matches the BAR-ATH "
                             "ball's velocity changes (tools/touch_check.py --diagnose)")
    parser.add_argument("--beyond-keep", type=float, default=1.0, metavar="P",
                        help="stress test: keep each stretch of ball beyond a line with "
                             "probability P, else hide it (simulate.thin_beyond; default 1, "
                             "off). 0.26 gives BAR-ATH's 4.2%% beyond a line, 0.82 with "
                             "--beyond-margin 1 its 2.7%% beyond by more than 1 m")
    parser.add_argument("--hide-keeper", type=float, default=1.0, metavar="P",
                        help="stress test: keep the ball over each stretch where the clean "
                             "match has a keeper on it in his own area with probability P, "
                             "else hide it (keeper.hide_keeper_ball; default 1, off)")
    parser.add_argument("--lose-near-goal", type=float, default=1.0, metavar="P",
                        help="stress test: keep the ball over each out-behind-the-goal and "
                             "keeper's-ball episode with probability P, else lose it "
                             "(keeper.lose_ball_near_goal; default 1, off). 0.5 and 0 bracket "
                             "BAR-ATH's keeper-zone aggregates (tools/keeper_check.py)")
    parser.add_argument("--beyond-margin", type=float, default=0.0, metavar="M",
                        help="only stretches more than M metres beyond a line are thinned")
    parser.add_argument("--prekickoff", type=float, default=0.0, metavar="S",
                        help="stress test: S seconds of synthetic pre-kickoff dead ball in "
                             "front of each second half (simulate.add_prekickoff; default 0, "
                             "off). BAR-ATH's second half has about 117")
    parser.add_argument("--prekickoff-profile", choices=sorted(PREKICKOFF_PROFILES),
                        default="calm", help="how players and ball behave before the kickoff")
    parser.add_argument("--labels", choices=relabel.VARIANTS, default="keep2",
                        help="the truth: controlled possession, solution/relabel.py's keep2 "
                             "(default, D21), or DFL's first-touch possession (dfl; every "
                             "change must also not be WORSE there). Numbers on different "
                             "labels are not comparable")
    parser.add_argument("--prior-labels", choices=relabel.VARIANTS, default="dfl",
                        help="default labels the priors are fitted on (decoder, touch, gapteam, "
                             "deadball), for methods that do not set prior_labels: dfl, as "
                             "solution/run.py submits (D21)")
    parser.add_argument("--boundaries", action="store_true",
                        help="also print tools/boundaries.py's report for every method")
    parser.add_argument("--clean-players", action="store_true",
                        help="degrade the ball only; players stay as tracked in the clean match "
                             "(the perfect player tracker what-if, simulator v1)")
    parser.add_argument("--ball-visible", type=float, default=0.0, metavar="SHARE",
                        help="what-if for a better tracker: reveal whole ball gaps at random "
                             "until about SHARE of each half shows the ball (e.g. 0.76, 0.90; "
                             "simulate.reveal_gaps); 0 = off")
    parser.add_argument("--clean", action="store_true",
                        help="score on undegraded data (ball always visible) for reference")
    parser.add_argument("--bootstrap", type=int, default=1000, help="draws; 0 = off")
    args = parser.parse_args()
    started = time.time()

    if not args.matches:
        args.matches = list({"dev": idsse.DEV_IDS, "holdout": idsse.HOLDOUT_IDS,
                             "all": idsse.MATCH_IDS}[args.split])
    # Priors are fitted lazily from idsse.load(), so this must be set before any method runs.
    idsse.LABELS = args.prior_labels
    # Likewise the trained models' simulator: their training sets are cached on first use.
    if args.train_simulator:
        os.environ[TRAIN_SIM_ENV] = args.train_simulator
    used_holdout = sorted(set(args.matches) & set(idsse.HOLDOUT_IDS))
    if used_holdout:
        print("WARNING: scoring held-out matches {}. Read the number, do not tune on it "
              "(PLAN.md section 6).".format(", ".join(used_holdout)))
    raw = [retime(idsse.load(match_id, labels=args.labels), args.tempo)
           for match_id in args.matches]
    # The pre-kickoff stretch differs by seed, its labels do not: truth from seed 0.
    matches = [add_prekickoff(m, args.prekickoff, 0, profile=args.prekickoff_profile)
               for m in raw]
    seeds = [None] if args.clean else args.seeds
    print("proxy league: {} matches x {} {} -- {} input{}".format(
        len(matches), len(seeds), "seed" if len(seeds) == 1 else "seeds",
        "clean" if args.clean else "degraded (simulator {})".format(args.simulator),
        ", position noise sd {} m".format(args.position_noise) if args.position_noise else "")
        + (", speeds x {}".format(args.speed_factor) if args.speed_factor != 1 else "")
        + (", tempo x {} (retimed)".format(args.tempo) if args.tempo != 1 else "")
        + (", ball noise sd {} m + {} jumps/min + smoothing {} s".format(
            args.ball_noise, args.ball_jumps, args.ball_smooth)
           if args.ball_noise or args.ball_jumps or args.ball_smooth else "")
        + (", ball visible raised to {}".format(args.ball_visible) if args.ball_visible else "")
        + (", players clean" if args.clean_players else "")
        + ", models trained on simulator {}".format(train_version())
        + (", {} s {} pre-kickoff".format(args.prekickoff, args.prekickoff_profile)
           if args.prekickoff else "")
        + (", ball beyond {} m kept with p {}".format(args.beyond_margin, args.beyond_keep)
           if args.beyond_keep < 1 else "")
        + (", keeper's ball kept with p {}".format(args.hide_keeper)
           if args.hide_keeper < 1 else "")
        + (", ball near goal kept with p {}".format(args.lose_near_goal)
           if args.lose_near_goal < 1 else "")
        + ", truth labels {}".format(args.labels)
        + ("" if idsse.LABELS == args.labels else ", priors on {} labels".format(idsse.LABELS)))

    # predictions[method][seed_index] = list of per-match code arrays
    predictions = {name: [] for name in args.methods}
    seen = []                  # seen[seed_index][match]: the degraded ball, for --boundaries
    for seed in seeds:
        base = [add_prekickoff(m, args.prekickoff, seed or 0, profile=args.prekickoff_profile)
                for m in raw]
        for clean, truth in zip(base, matches):
            assert np.array_equal(clean.labels, truth.labels)
        inputs = [m if seed is None else
                  degrade(m, seed, args.simulator, ball_visible=args.ball_visible or None,
                          players=not args.clean_players)
                  for m in base]
        if args.position_noise:
            inputs = [add_position_noise(m, args.position_noise, seed or 0) for m in inputs]
        if args.speed_factor != 1:
            inputs = [scale_speeds(m, args.speed_factor) for m in inputs]
        if args.ball_noise or args.ball_jumps or args.ball_smooth:
            inputs = [add_ball_noise(m, args.ball_noise, args.ball_jumps, seed or 0,
                                     smooth_s=args.ball_smooth) for m in inputs]
        if args.beyond_keep < 1:
            inputs = [thin_beyond(m, args.beyond_keep, seed or 0, args.beyond_margin)
                      for m in inputs]
        if args.hide_keeper < 1:
            from solution.keeper import hide_keeper_ball
            inputs = [hide_keeper_ball(m, clean, args.hide_keeper, seed or 0)
                      for m, clean in zip(inputs, base)]
        if args.lose_near_goal < 1:
            from solution.keeper import lose_ball_near_goal
            inputs = [lose_ball_near_goal(m, clean, args.lose_near_goal, seed or 0)
                      for m, clean in zip(inputs, base)]
        seen.append([m.ball_seen for m in inputs])
        for name in args.methods:
            predictions[name].append([METHODS[name](m) for m in inputs])
    truths = [m.labels for m in matches]

    report = {"split": args.split, "matches": args.matches, "seeds": seeds, "clean": args.clean,
              "simulator": None if args.clean else args.simulator,
              "position_noise": args.position_noise, "speed_factor": args.speed_factor,
              "tempo": args.tempo,
              "ball_noise": args.ball_noise, "ball_jumps": args.ball_jumps,
              "ball_smooth": args.ball_smooth, "ball_visible": args.ball_visible,
              "clean_players": args.clean_players,
              "prekickoff": args.prekickoff,
              "prekickoff_profile": args.prekickoff_profile, "beyond_keep": args.beyond_keep,
              "beyond_margin": args.beyond_margin, "hide_keeper": args.hide_keeper,
              "lose_near_goal": args.lose_near_goal,
              "labels": args.labels,
              "prior_labels": idsse.LABELS, "methods": {}}
    pooled_flags = {}          # pooled_flags[name] = (flags, n_pred) for seed index 0
    for name in args.methods:
        per_match, by_seed = [], []
        for s in range(len(seeds)):
            parts, n_pred, confusion = [], 0, 0
            for i in range(len(matches)):
                result = pm.evaluate(truths[i], predictions[name][s][i])
                flags, n = spell_flags(truths[i], predictions[name][s][i])
                # The pooled numbers are only trustworthy if this reproduces the official score.
                assert abs(f1_from_flags(flags, n) - result["spell_f1_mean"]) < 1e-12
                parts.append(flags)
                n_pred += n
                confusion = confusion + np.array(result["confusion"])[:pm.N_SCORED]
                if s == 0:
                    per_match.append({"match": matches[i].name, "n_gt": result["n_gt_spells"],
                                      "spell_f1": [], "macro_f1": [], "n_pred": []})
                per_match[i]["spell_f1"].append(result["spell_f1_mean"])
                per_match[i]["macro_f1"].append(result["macro_f1"])
                per_match[i]["n_pred"].append(result["n_pred_spells"])
            flags = np.concatenate(parts, axis=1)
            by_seed.append({"spell_f1": f1_from_flags(flags, n_pred),
                            "macro_f1": pm.macro_f1(confusion), "n_pred": n_pred})
            if s == 0:
                pooled_flags[name] = (flags, n_pred)
        for row in per_match:
            for key in ("spell_f1", "macro_f1", "n_pred"):
                row[key] = float(np.mean(row[key]))
        flags, n_pred = pooled_flags[name]
        report["methods"][name] = {
            "per_match": per_match,
            "pooled_spell_f1": float(np.mean([r["spell_f1"] for r in by_seed])),
            "pooled_spell_f1_by_seed": [r["spell_f1"] for r in by_seed],
            "pooled_macro_f1": float(np.mean([r["macro_f1"] for r in by_seed])),
            "pooled_ci": list(bootstrap(flags, n_pred, args.bootstrap)) if args.bootstrap
            else [None, None],
            "n_gt": int(flags.shape[1]),
            "n_pred": float(np.mean([r["n_pred"] for r in by_seed])),
        }

    # -- per match table --------------------------------------------------------------------
    names = args.methods
    print("\nspellF1 per match (mean over seeds)        true possessions in brackets")
    print("{:<16}".format("match") + "".join("{:>20}".format(n[:19]) for n in names))
    for i, match in enumerate(matches):
        row = report["methods"][names[0]]["per_match"][i]
        print("{:<16}".format("{} [{}]".format(match.name.replace("idsse_", ""), row["n_gt"])) +
              "".join("{:>20.4f}".format(report["methods"][n]["per_match"][i]["spell_f1"])
                      for n in names))

    # -- pooled table -----------------------------------------------------------------------
    first = report["methods"][names[0]]
    print("\npooled over {} true possessions".format(first["n_gt"]))
    print("{:<20} {:>8} {:>17} {:>8} {:>7}   {}".format(
        "method", "spellF1", "5-95% interval", "macroF1", "poss", "seed spread"))
    for name in names:
        r = report["methods"][name]
        ci = r["pooled_ci"]
        print("{:<20} {:>8.4f} {:>17} {:>8.4f} {:>7.0f}   {}".format(
            name[:20], r["pooled_spell_f1"],
            "{:.4f}..{:.4f}".format(*ci) if ci[0] is not None else "--",
            r["pooled_macro_f1"], r["n_pred"],
            "{:.4f}".format(np.ptp(r["pooled_spell_f1_by_seed"])) if len(seeds) > 1 else "--"))

    # -- paired comparison against the first method ----------------------------------------
    if len(names) > 1 and args.bootstrap:
        print("\ngain over {} (paired bootstrap over the same possessions, first seed)".format(
            names[0]))
        reference = pooled_flags[names[0]]
        for name in names[1:]:
            low, high = bootstrap(reference[0], reference[1], args.bootstrap,
                                  other=pooled_flags[name])
            gain = report["methods"][name]["pooled_spell_f1"] - first["pooled_spell_f1"]
            wins = sum(report["methods"][name]["per_match"][i]["spell_f1"] >
                       first["per_match"][i]["spell_f1"] for i in range(len(matches)))
            if high < 0:
                verdict = "WORSE"
            elif gain >= 0.02 and low > 0 and wins * 3 >= 2 * len(matches):
                verdict = "ADOPT"
            elif low > 0:
                verdict = "better, below the adoption bar"
            else:
                verdict = "within noise"
            print("  {:<20} {:+.4f}   interval {:+.4f}..{:+.4f}   wins {}/{}   {}".format(
                name[:20], gain, low, high, wins, len(matches), verdict))
            report["methods"][name]["gain_vs_first"] = {"gain": gain, "interval": [low, high],
                                                        "wins": wins, "verdict": verdict}

    if args.boundaries:
        for name in names:
            boundaries.report(name, truths, predictions[name], seen)

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    out = os.path.join(OUT_DIR, "latest.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    print("\nwrote {}  ({:.0f} s)".format(os.path.relpath(out, ROOT), time.time() - started))


if __name__ == "__main__":
    main()
