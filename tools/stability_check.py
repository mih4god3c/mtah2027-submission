"""L3 stability check (PLAN.md section 6): does the submitted configuration sit on a plateau?

    .venv/bin/python tools/stability_check.py                       # proxy: dev matches, seed 0, D12 ball
    .venv/bin/python tools/stability_check.py --seeds 0 1 2         # more seeds (slower)
    .venv/bin/python tools/stability_check.py --bar-ath             # label-free, for a human (needs data/)
    .venv/bin/python tools/stability_check.py --method c3s_dead2    # another method of the c3s family

Each setting of the method is moved by a fixed factor, one at a time, and the match is
re-decoded. PLAN L3's rule: a robust configuration does not swing its possession count by 30%
under small parameter changes. Every row reports the possession count against the reference's,
the stoppage count, the dead-ball and home shares, and, on the proxy, pooled spellF1 with its
change. On BAR-ATH (`--bar-ath`) there are no labels, so only the counts are printed, which is
what a person compares with the proxy's rows. The perturbation list is fixed in PERTURBATIONS
and nothing here is a tuning grid: the row that scores best is not a candidate.

Perturbations: the four evidence weights (`w`, `touch_w`, `team_w`, `frame_w`) at 0.8 and 1.2
times their value (frame_w 1.2 goes above D16's limit on purpose, as a probe only); the fitted
median possession and stoppage length at 0.9 and 1.1 times; the touch engine's velocity-change
threshold and player radius at 0.8 and 1.2 times. Decoder and prior perturbations re-decode the
prepared inputs; touch-engine ones rebuild the method.

Only the `v5_frame_model` family (`v5_frame_model*`, `c3s_*`; solution/methods.py) is
supported: the tool rebuilds touch-engine variants from the method's decoder settings plus
`methods._GAPTEAM`'s non-decoder ones, so a member that also sets `touch_params`, `merge_ball`
or `control` is not rebuilt faithfully for the touch rows.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder, methods, touch  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.simulate import add_ball_noise, degrade  # noqa: E402
from solution.sources import idsse  # noqa: E402
from tools.proxy_league import f1_from_flags, spell_flags  # noqa: E402

D12 = {"ball_noise": 0.05, "ball_jumps": 3.4, "ball_smooth": 0.26}
WEIGHTS = ("w", "touch_w", "team_w", "frame_w")
PERTURBATIONS = (
    [{"name": "{} x{}".format(k, f), "decoder": {k: f}} for k in WEIGHTS for f in (0.8, 1.2)]
    + [{"name": "possession length x{}".format(f), "prior": ("possession", f)} for f in (0.9, 1.1)]
    + [{"name": "stoppage length x{}".format(f), "prior": ("dead", f)} for f in (0.9, 1.1)]
    + [{"name": "touch_dv x{}".format(f), "touch": {"touch_dv": f}} for f in (0.8, 1.2)]
    + [{"name": "touch_m x{}".format(f), "touch": {"touch_m": f}} for f in (0.8, 1.2)]
)
FRAGILE, WATCH = 0.30, 0.10     # PLAN L3's 30% rule, and a softer line worth a look


def family_kwargs(predict):
    """The semi_markov keyword arguments that rebuild `predict` (c3s family only)."""
    changed = {k: v for k, v in predict.settings.items() if v != decoder.DEFAULTS[k]}
    return dict(changed, kickoff_params=methods._GAPTEAM["kickoff_params"],
                team_params=methods._GAPTEAM["team_params"])


def perturbed_priors(priors, key, factor):
    """Copy of the priors with the fitted median length of `key` ('possession' or 'dead')
    multiplied by `factor` (mu is the log of the median)."""
    out = dict(priors)
    out["durations"] = {k: dict(v) for k, v in priors["durations"].items()}
    out["durations"][key]["mu"] = priors["durations"][key]["mu"] + math.log(factor)
    return out


def decode_variants(predict, match, perturbations=PERTURBATIONS):
    """{name: scorer codes} for the reference ('reference') and every perturbation."""
    inp = predict.prepare(match)
    out = {"reference": methods.to_scorer_codes(_letters(predict.decode(match, inp)), match)}
    base = family_kwargs(predict)
    for p in perturbations:
        if "decoder" in p:
            overrides = {k: predict.settings[k] * f for k, f in p["decoder"].items()}
            states = predict.decode(match, inp, overrides)
        elif "prior" in p:
            inp2 = dict(inp, priors=perturbed_priors(inp["priors"], *p["prior"]))
            states = predict.decode(match, inp2)
        else:
            touch_params = {k: touch.PARAMS[k] * f for k, f in p["touch"].items()}
            variant = methods.semi_markov("stability_variant", touch_params=touch_params, **base)
            states = variant.decode(match, variant.prepare(match))
        out[p["name"]] = methods.to_scorer_codes(_letters(states), match)
    return out


def _letters(states):
    return np.array(["H", "A", "D"])[states]


def counts(codes, halves, fps):
    """Possessions, stoppages, dead-ball share, home share of alive frames, inside the halves."""
    runs = list(halves)
    spells = pm.possession_spells(codes, runs)
    dead = [(a, b) for a, b, k in pm.segments(codes, runs) if k == pm.D]
    graded = sum(b - a for a, b in halves)
    n_dead = sum(int((codes[a:b] == pm.D).sum()) for a, b in halves)
    n_home = sum(int((codes[a:b] == pm.H).sum()) for a, b in halves)
    alive = graded - n_dead
    return {"poss": len(spells), "stops": len(dead), "d_share": n_dead / float(graded),
            "home": n_home / float(alive) if alive else float("nan"),
            "stop_med_s": float(np.median([b - a for a, b in dead])) / fps if dead else 0.0}


def proxy_inputs(match_ids, seeds, stress=D12, labels="keep2", prior_labels="dfl"):
    """[(truth labels, degraded match)] as the proxy league's decision run builds them."""
    idsse.LABELS = prior_labels
    out = []
    for match_id in match_ids:
        clean = idsse.load(match_id, labels=labels)
        for seed in seeds:
            m = degrade(clean, seed, "v1")
            if stress:
                m = add_ball_noise(m, stress["ball_noise"], stress["ball_jumps"], seed,
                                   smooth_s=stress["ball_smooth"])
            out.append((clean.labels, m))
    return out


def summarise(rows, names):
    """Per perturbation: summed counts over the matches, and pooled spellF1 where truth exists."""
    table = []
    for name in names:
        poss = sum(r[name]["poss"] for r in rows)
        stops = sum(r[name]["stops"] for r in rows)
        d_share = float(np.mean([r[name]["d_share"] for r in rows]))
        home = float(np.mean([r[name]["home"] for r in rows]))
        stop_med = float(np.mean([r[name]["stop_med_s"] for r in rows]))
        f1 = None
        if all("flags" in r[name] for r in rows):
            flags = np.concatenate([r[name]["flags"] for r in rows], axis=1)
            n_pred = sum(r[name]["n_pred"] for r in rows)
            f1 = f1_from_flags(flags, n_pred)
        table.append({"name": name, "poss": poss, "stops": stops, "d_share": d_share,
                      "home": home, "stop_med_s": stop_med, "spell_f1": f1})
    return table


def verdict(table):
    """(worst relative possession-count change among the weight rows, label)."""
    ref = table[0]["poss"]
    worst = 0.0
    for row in table[1:]:
        if any(row["name"].startswith(w + " x") for w in WEIGHTS):
            worst = max(worst, abs(row["poss"] - ref) / float(ref))
    label = "FRAGILE (over {:.0%})".format(FRAGILE) if worst > FRAGILE else (
        "watch (over {:.0%})".format(WATCH) if worst > WATCH else "plateau")
    return worst, label


def print_table(table, with_scores):
    ref = table[0]
    head = "{:<26} {:>6} {:>7} {:>6} {:>8} {:>7} {:>8}".format(
        "perturbation", "poss", "ratio", "stops", "stop_med", "D share", "home")
    if with_scores:
        head += " {:>8} {:>8}".format("spellF1", "change")
    print(head)
    for row in table:
        line = "{:<26} {:>6} {:>7.3f} {:>6} {:>8.1f} {:>7.3f} {:>8.3f}".format(
            row["name"], row["poss"], row["poss"] / float(ref["poss"]), row["stops"],
            row["stop_med_s"], row["d_share"], row["home"])
        if with_scores:
            line += " {:>8.4f} {:>+8.4f}".format(row["spell_f1"], row["spell_f1"] - ref["spell_f1"])
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="v5_frame_model",
                        choices=sorted(n for n in methods.METHODS
                                       if n.startswith("v5_frame_model") or n.startswith("c3s_")))
    parser.add_argument("--bar-ath", action="store_true",
                        help="BAR-ATH, label-free counts only (needs data/); for a person to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--no-stress", action="store_true", help="plain simulator v1 ball")
    args = parser.parse_args()
    started = time.time()
    predict = methods.METHODS[args.method]
    names = ["reference"] + [p["name"] for p in PERTURBATIONS]

    if args.bar_ath:
        from solution.sources import atd
        match = atd.load()
        variants = decode_variants(predict, match)
        rows = [{n: counts(v, match.halves, match.fps) for n, v in variants.items()}]
        print("BAR-ATH, {}: label-free counts, whole match ({} perturbations)".format(
            args.method, len(PERTURBATIONS)))
    else:
        pairs = proxy_inputs(idsse.DEV_IDS, args.seeds, None if args.no_stress else D12)
        rows = []
        for truth, match in pairs:
            variants = decode_variants(predict, match)
            row = {}
            for n, v in variants.items():
                row[n] = counts(v, match.halves, match.fps)
                flags, n_pred = spell_flags(truth, v)
                row[n]["flags"], row[n]["n_pred"] = flags, n_pred
            rows.append(row)
        print("proxy, {}: {} dev matches x seeds {}, {} ball, keep2 truth ({} perturbations)"
              .format(args.method, len(idsse.DEV_IDS), args.seeds,
                      "plain v1" if args.no_stress else "D12", len(PERTURBATIONS)))
    table = summarise(rows, names)
    print_table(table, with_scores=not args.bar_ath)
    worst, label = verdict(table)
    print("\nL3: worst possession-count change over the +-20% weight rows {:.1%}: {}".format(
        worst, label))
    print("({:.0f} s)".format(time.time() - started))


if __name__ == "__main__":
    main()
