"""#3 control model, Step 0: what a perfect control judgement on the touch engine's events is worth.

    .venv/bin/python tools/control_check.py                    # keep2 and DFL, smoothing 0.26 and 0.40 s
    .venv/bin/python tools/control_check.py --smooth 0.26 --labels keep2 --seeds 0
    .venv/bin/python tools/control_check.py --model            # Step 1 bar 1: the learned score

Proxy only (it reads the truth); nothing runs on BAR-ATH. Method and pass bar: docs/EXPERIMENTS.md
section 21, fixed before the first run.

Per event of solution/touch.py (a touch, or a control episode's start) at frame f by team k, the
truth over [f, f + controlmodel.WINDOW_S) decides: majority k "agreed", majority the other team
"contradicted", anything else "neutral" (never removed). The holder is re-swept without the
removed events (touch.sweep_holder's keep masks) and the reference re-decoded with it:
  - as is: no event removed; must reproduce the reference exactly (asserted);
  - perfect: every contradicted event removed; "touches only" keeps the control episodes;
  - imitations of a real model: a seeded random share r of contradicted events removed and a
    share f of agreed events removed by mistake, for each (r, f) in IMITATIONS.
Scores are pooled spellF1 over the dev matches, mean over seeds, as the proxy league pools them.
Deterministic: the imitation draws use their own seeded stream per match and seed.

With --model (Step 1, bar 1): solution/controlmodel.py's score on each held-out dev match's
touches (keep2 verdicts): recall of contradicted touches and the share of agreed touches dropped
at the model's own threshold, and the AUC, pooled over matches and seeds, per smoothing.

With --diagnose (Step 1 diagnostic, section 21): rows A to F, the model's drops taken apart.
"""

import argparse
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import controlmodel, touch  # noqa: E402
from solution.controlmodel import AGREED, CONTRADICTED, verdicts as event_verdicts  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import METHODS, to_scorer_codes  # noqa: E402
from solution.simulate import add_ball_noise, degrade  # noqa: E402
from tools.proxy_league import f1_from_flags, spell_flags  # noqa: E402

REFERENCE = "v5_frame_model"      # D23, submitted
BALL_NOISE, BALL_JUMPS = 0.05, 3.4  # D12's ball noise; smoothing is the --smooth argument
IMITATIONS = ((0.7, 0.1), (0.5, 0.1), (0.7, 0.2))


def resweep(match, touches, keep_touch=None, keep_control=None):
    return touch.sweep_holder(match.n_frames, touches.touch_frame, touches.touch_team,
                              touches.control, touches.usable, match.halves, match.fps,
                              touch.PARAMS, keep_touch, keep_control)


def imitation(verdicts, recall, false_drop, rng):
    """Keep mask: contradicted events dropped with p recall, agreed ones with p false_drop."""
    draw = rng.random_sample(len(verdicts))
    drop = ((verdicts == CONTRADICTED) & (draw < recall)) | \
           ((verdicts == AGREED) & (draw < false_drop))
    return ~drop


def rows_for(truth, match, seed):
    """{row name: scorer codes} plus event counts for one degraded match."""
    method = METHODS[REFERENCE]
    inp = method.prepare(match)
    touches = inp["touches"]
    reference = to_scorer_codes(np.array(["H", "A", "D"])[method.decode(match, inp)], match)

    def decoded(holder):
        states = method.decode(match, dict(inp, holder=holder))
        return to_scorer_codes(np.array(["H", "A", "D"])[states], match)

    vt = event_verdicts(truth, touches.touch_frame, touches.touch_team, match.fps)
    vc = event_verdicts(truth, touches.control[:, 0], touches.control[:, 2], match.fps)
    holder = resweep(match, touches)
    assert np.array_equal(holder, touches.holder), "re-sweep does not reproduce the engine"
    out = {"as is": decoded(holder)}
    assert np.array_equal(out["as is"], reference), "re-decode does not reproduce the reference"
    out["perfect"] = decoded(resweep(match, touches, vt != CONTRADICTED, vc != CONTRADICTED))
    out["perfect, touches only"] = decoded(resweep(match, touches, vt != CONTRADICTED, None))
    rng = np.random.RandomState(zlib.crc32(match.name.encode()) % (2 ** 31) + 7919 * seed)
    for r, f in IMITATIONS:
        out["imitation r {} f {}".format(r, f)] = decoded(
            resweep(match, touches, imitation(vt, r, f, rng), imitation(vc, r, f, rng)))
    counts = {"touches": len(vt), "controls": len(vc),
              "touch contradicted": int(np.sum(vt == CONTRADICTED)),
              "touch agreed": int(np.sum(vt == AGREED)),
              "control contradicted": int(np.sum(vc == CONTRADICTED)),
              "control agreed": int(np.sum(vc == AGREED))}
    return out, counts


def pooled(results, name, seeds):
    per_seed = []
    for seed in seeds:
        parts, n_pred = [], 0
        for truth, rows, s in results:
            if s == seed:
                flags, n = spell_flags(truth, rows[name])
                parts.append(flags)
                n_pred += n
        per_seed.append(f1_from_flags(np.concatenate(parts, axis=1), n_pred))
    return float(np.mean(per_seed))


def run(smooth, labels, seeds):
    from solution.sources import idsse
    print("\n== truth {}, simulator v1 + ball noise {} m, {} jumps/min, smoothing {} s; "
          "{} dev matches x seeds {} ==".format(labels, BALL_NOISE, BALL_JUMPS, smooth,
                                                len(idsse.DEV_IDS), seeds))
    results, counts = [], []
    for match_id in idsse.DEV_IDS:
        raw = idsse.load(match_id, labels=labels)
        for seed in seeds:
            match = add_ball_noise(degrade(raw, seed), BALL_NOISE, BALL_JUMPS, seed=seed,
                                   smooth_s=smooth)
            rows, c = rows_for(raw.labels, match, seed)
            results.append((raw.labels, rows, seed))
            counts.append(c)
    base = pooled(results, "as is", seeds)
    print("{:<24} {:>8} {:>9}".format("row", "spellF1", "gain"))
    for name in results[0][1]:
        score = pooled(results, name, seeds)
        print("{:<24} {:>8.4f} {:>+9.4f}".format(name, score, score - base))
    n = float(len(seeds))
    total = {k: sum(c[k] for c in counts) for k in counts[0]}
    print("events per seed: touches {:.0f} (contradicted {:.3f}, agreed {:.3f}), control "
          "episodes {:.0f} (contradicted {:.3f}, agreed {:.3f})".format(
              total["touches"] / n, total["touch contradicted"] / max(total["touches"], 1),
              total["touch agreed"] / max(total["touches"], 1), total["controls"] / n,
              total["control contradicted"] / max(total["controls"], 1),
              total["control agreed"] / max(total["controls"], 1)))


def model_check(smooth, seeds):
    """Bar 1: touch-level recall and false drops of the learned score on held-out matches."""
    from sklearn.metrics import roc_auc_score
    from solution.sources import idsse
    labels = controlmodel.PARAMS["labels"]
    drops, verdict_all, score_all = [], [], []
    thresholds = []
    for match_id in idsse.DEV_IDS:
        raw = idsse.load(match_id, labels=labels)
        for seed in seeds:
            match = add_ball_noise(degrade(raw, seed), BALL_NOISE, BALL_JUMPS, seed=seed,
                                   smooth_s=smooth)
            touches = touch.detect(match)
            v = event_verdicts(raw.labels, touches.touch_frame, touches.touch_team, match.fps)
            p = controlmodel.scores(match, touches)
            keep = controlmodel.keep_mask(match, touches)
            thresholds.append(controlmodel.idsse_model(
                match.name, tuple(sorted(controlmodel.PARAMS.items())))[1])
            drops.append(~keep)
            verdict_all.append(v)
            score_all.append(p)
    drop, v, p = (np.concatenate(x) for x in (drops, verdict_all, score_all))
    judged = v != 0
    print("\n== control model, truth {}, v1 + ball noise {} m, {} jumps/min, smoothing {} s; "
          "{} dev matches x seeds {} (held out) ==".format(labels, BALL_NOISE, BALL_JUMPS, smooth,
                                                          len(idsse.DEV_IDS), seeds))
    print("touches {} (contradicted {}, agreed {}, neutral {}); thresholds {:.3f}..{:.3f}".format(
        len(v), int(np.sum(v == CONTRADICTED)), int(np.sum(v == AGREED)), int(np.sum(~judged)),
        min(thresholds), max(thresholds)))
    print("recall of contradicted {:.3f}; agreed dropped {:.3f}; neutral dropped {:.3f}; "
          "AUC {:.3f}".format(np.mean(drop[v == CONTRADICTED]), np.mean(drop[v == AGREED]),
                              np.mean(drop[~judged]) if np.any(~judged) else float("nan"),
                              roc_auc_score(v[judged] == CONTRADICTED, p[judged])))


def diagnose_rows(truth, match, seed):
    """{row: scorer codes} for the Step 1 diagnostic (section 21: rows A to F)."""
    method = METHODS[REFERENCE]
    inp = method.prepare(match)
    touches = inp["touches"]

    def decoded(keep_touch):
        holder = resweep(match, touches, keep_touch, None)
        states = method.decode(match, dict(inp, holder=holder))
        return to_scorer_codes(np.array(["H", "A", "D"])[states], match)

    v = event_verdicts(truth, touches.touch_frame, touches.touch_team, match.fps)
    model_drop = ~controlmodel.keep_mask(match, touches)
    contradicted, agreed, neutral = v == CONTRADICTED, v == AGREED, v == 0
    rng = np.random.RandomState(zlib.crc32(match.name.encode()) % (2 ** 31) + 7919 * seed + 1)
    false_rate = float(np.mean(model_drop[agreed])) if agreed.any() else 0.0
    recall = float(np.mean(model_drop[contradicted])) if contradicted.any() else 0.0
    random_false = agreed & (rng.random_sample(len(v)) < false_rate)
    random_true = contradicted & (rng.random_sample(len(v)) < recall)
    drops = {
        "A as is": np.zeros(len(v), dtype=bool),
        "B model": model_drop,
        "C model, neutral kept": model_drop & ~neutral,
        "D model, contradicted only": model_drop & contradicted,
        "E D + random false drops": (model_drop & contradicted) | random_false,
        "F random at model rates": random_true | random_false,
    }
    out = {name: decoded(~drop) for name, drop in drops.items()}
    counts = {"touches": len(v), "contradicted": int(contradicted.sum()),
              "agreed": int(agreed.sum()), "neutral": int(neutral.sum()),
              "model drops": int(model_drop.sum()),
              "model contradicted": int((model_drop & contradicted).sum()),
              "model agreed": int((model_drop & agreed).sum()),
              "model neutral": int((model_drop & neutral).sum())}
    return out, counts


def diagnose(smooth, seeds):
    from solution.sources import idsse
    labels = controlmodel.PARAMS["labels"]
    print("\n== diagnostic, truth {}, v1 + ball noise {} m, {} jumps/min, smoothing {} s; "
          "{} dev matches x seeds {} ==".format(labels, BALL_NOISE, BALL_JUMPS, smooth,
                                                len(idsse.DEV_IDS), seeds))
    results, counts = [], []
    for match_id in idsse.DEV_IDS:
        raw = idsse.load(match_id, labels=labels)
        for seed in seeds:
            match = add_ball_noise(degrade(raw, seed), BALL_NOISE, BALL_JUMPS, seed=seed,
                                   smooth_s=smooth)
            rows, c = diagnose_rows(raw.labels, match, seed)
            results.append((raw.labels, rows, seed))
            counts.append(c)
    base = pooled(results, "A as is", seeds)
    print("{:<30} {:>8} {:>9} {:>8}".format("row", "spellF1", "gain", "poss"))
    for name in results[0][1]:
        score = pooled(results, name, seeds)
        n_poss = sum(len(pm.possession_spells(pm.fill_abstentions(r[name], pm.valid_runs(t)),
                                              pm.valid_runs(t)))
                     for t, r, _ in results) / float(len(seeds))
        print("{:<30} {:>8.4f} {:>+9.4f} {:>8.0f}".format(name, score, score - base, n_poss))
    total = {k: sum(c[k] for c in counts) for k in counts[0]}
    print("per seed: touches {:.0f}; contradicted {:.0f}, agreed {:.0f}, neutral {:.0f}; model "
          "drops {:.0f} = {:.0f} contradicted + {:.0f} agreed + {:.0f} neutral".format(
              *(total[k] / float(len(seeds)) for k in (
                  "touches", "contradicted", "agreed", "neutral", "model drops",
                  "model contradicted", "model agreed", "model neutral"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smooth", nargs="+", type=float, default=[0.26, 0.40])
    parser.add_argument("--labels", nargs="+", default=["keep2", "dfl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--model", action="store_true",
                        help="Step 1 bar 1: the learned control score, touch level")
    parser.add_argument("--diagnose", action="store_true",
                        help="Step 1 diagnostic: the model's drops taken apart (rows A-F)")
    args = parser.parse_args()
    if args.diagnose:
        for smooth in args.smooth:
            diagnose(smooth, args.seeds)
        return
    if args.model:
        for smooth in args.smooth:
            model_check(smooth, args.seeds)
        return
    for labels in args.labels:
        for smooth in args.smooth:
            run(smooth, labels, args.seeds)


if __name__ == "__main__":
    main()
