"""Adversarial validation: can a classifier tell BAR-ATH's frames from the proxy's? Aggregates only.

    .venv/bin/python tools/adversarial_check.py --proxy-only   # odd-one-out baseline, no data/
    .venv/bin/python tools/adversarial_check.py --sanity       # v0 against v1: must find the gap
    .venv/bin/python tools/adversarial_check.py                # (needs data/) for a human to run

Why (D26 update 3): solution/deadmodel.py lowers seen-ball D on BAR-ATH and raises it on the
proxy. Either BAR-ATH's seen-ball frames are a different population from the proxy's (the
simulator is wrong there), or they are not and the model reads the same frames differently. This
asks the first question without labels on BAR-ATH.

What it does, per subset of graded frames ("seen": the ball is seen, the subset D26 is about;
"gaps": it is not) and per feature set ("deadmodel": deadmodel.features, exactly what the model
reads; "context": those plus the ball and its visibility, which the model does not read but the
simulator sets, so they test the simulator's known gaps, simulate.py's docstring):
  - rows: every STEP-th frame of the subset. The proxy is the IDSSE dev matches (never the
    hold-out, D3) degraded as `make proxy` does (simulator v1 + the D12 ball stress); its seeds
    take turns (seed k reads frames k*STEP, k*STEP + n*STEP, ...), so the proxy side is about
    one seed's worth of rows (default one seed). Each proxy match weighs the same; the two classes weigh the same;
  - a gradient-boosted classifier (deadmodel's learner, half the rounds at twice the rate) of
    "BAR-ATH or proxy",
    cross-fitted over FOLDS folds of BLOCK_S blocks (a block never sits in two folds), scored by
    weighted out-of-fold AUC;
  - the baseline: the same classifier for each IDSSE dev match against the other four (all on
    the proxy). It says how different one match is from others under the same simulator, in
    all and per feature group (the same shuffle as below);
  - drivers, per feature group: the AUC lost when the group is shuffled within each test fold,
    and (--alone, slower) the AUC of a classifier that reads the group alone. Per feature: medians each side,
    missing shares, and the feature's own AUC (above 0.5: higher on BAR-ATH);
  - covariate shift (description only): with w = p / (1 - p) from the out-of-fold P(BAR-ATH),
    the proxy's truth D share and the model's mean P(dead) reweighted so the proxy frames look
    like BAR-ATH's, next to BAR-ATH's own mean P(dead), and the proxy's D share by quintile of
    P(BAR-ATH). It assumes P(D | features) is the same in both, which is not tested here.

Reading rules, fixed before the first run on BAR-ATH (amended after the first --sanity run, still
before any BAR-ATH run: the odd-one-out AUC came out at 0.87 to 0.90, so the overall AUC alone
says little, and R2 now compares each group with its own odd-one-out value):
  R1. Per subset and feature set, the baseline is the largest odd-one-out AUC. BAR-ATH at most
      MARGIN above it: no difference beyond one match against others overall.
  R2. Whatever R1 says, a feature group is a DRIVER when its shuffle drop for BAR-ATH exceeds the
      largest odd-one-out drop for that group by at least DRIVER_MARGIN. A driver the simulator
      sets ("tracked", "player beyond line", "ball", "visibility") is a simulator target for
      PLAN #1. A driver it does not set ("shape", "lines and box", the speed ranks and spread)
      points at play style, league or tracking quality, not directly at the simulator. No
      driver on the seen frames: the proxy result stands there and the simulator is not the
      explanation of D26's seen-D sign.
  R2b. (added with R2's amendment) Per feature, SHIFT when |its own AUC - 0.5| exceeds the
      largest odd-one-out |AUC - 0.5| of that feature by at least DRIVER_MARGIN. The first
      --sanity run showed why: v1's gaps in slow play make seen frames faster (speed_rank AUC
      0.374), but the speed ranks also tell one IDSSE match from another, so their group drop
      hid it. A SHIFT is read like a DRIVER.
  R4. (added before any BAR-ATH run) The reweighted proxy numbers are read only when the
      reweighted mean P(dead) is within 0.02 of the target's own and the effective rows are at
      least 0.20; otherwise the weights are wrong, and the line says so.
  R3. Nothing here adopts or submits anything. A simulator change is decided on the proxy first
      (D26 update 3), with the dead-ball model re-run on the corrected proxy.
--sanity runs the same with the dev matches degraded by simulator v0 (no player losses or
ghosts, gaps not tied to play) as the target. It must give a high AUC led by "tracked": a check
that the tool finds a simulator difference it is told about. Nothing frame-level is printed.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import deadball, deadmodel  # noqa: E402
from solution.framemodel import _window_mean  # noqa: E402
from solution.match import pm  # noqa: E402

STEP = 5                 # one row per STEP frames of a subset
BLOCK_S = 60.0           # cross-fitting blocks
FOLDS = 5
MARGIN = 0.03            # R1
DRIVER_MARGIN = 0.02     # R2
CLIP = 0.02              # P(BAR-ATH) clip for the covariate-shift weights
SHUFFLE_SEED = 0
SUBSETS = ("seen", "gaps")
FEATURE_SETS = ("deadmodel", "context")
CONTEXT_NAMES = ["ball_beyond", "ball_still", "ball_nearest_m",
                 "run_s", "run_pos", "seen_past3", "seen_future3"]
GROUPS = {
    "speed rank now": ["speed_rank", "slow_share_rank", "speed_max_rank", "speed_p25_rank"],
    "speed rank context": ["rank_{}{:g}".format(k, s) for s in deadmodel.CONTEXT_S
                           for k in ("past", "future", "change")],
    "speed spread": ["speed_cv", "cv_past3", "cv_future3"],
    "shape": ["gyration", "tilt_abs"],
    "lines and box": ["touchline_min", "goalline_min", "in_box"],
    "player beyond line": ["beyond_past3", "beyond_future3"],
    "tracked": ["tracked"],
    "ball": ["ball_beyond", "ball_still", "ball_nearest_m"],
    "visibility": ["run_s", "run_pos", "seen_past3", "seen_future3"],
}
SIMULATOR_GROUPS = ("tracked", "player beyond line", "ball", "visibility")


# -- features -------------------------------------------------------------------------------------

def run_lengths(flag):
    """(length of the run each frame sits in, position inside it from 0 to 1)."""
    n = len(flag)
    change = np.flatnonzero(flag[1:] != flag[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [n]))
    length = np.repeat(stops - starts, stops - starts).astype(float)
    pos = (np.arange(n) - np.repeat(starts, stops - starts)) / np.maximum(length - 1, 1)
    return length, pos


def context_features(match):
    """(T, 7): the ball and its visibility; ball columns are NaN where the ball is unseen."""
    fps = match.fps
    seen = match.ball_seen
    beyond = np.where(seen, deadball.ball_beyond(match).astype(float), np.nan)
    still = np.where(seen, deadball.ball_still(match).astype(float), np.nan)
    with np.errstate(invalid="ignore"):
        dist = np.hypot(match.player_x - match.ball_x[:, None], match.player_y - match.ball_y[:, None])
    nearest = np.full(len(seen), np.nan)
    rows = seen & np.isfinite(dist).any(axis=1)
    nearest[rows] = np.nanmin(dist[rows], axis=1)
    length, pos = run_lengths(seen)
    side = int(round(3.0 * fps))
    seen_f = seen.astype(float)
    return np.column_stack([beyond, still, nearest, length / fps, pos,
                            _window_mean(seen_f, -side, 0), _window_mean(seen_f, 0, side)])


def match_features(match, feature_set):
    X, names = deadmodel.features(match)
    if feature_set == "context":
        X = np.column_stack([X, context_features(match)]).astype(np.float32)
        names = list(names) + CONTEXT_NAMES
    return X, list(names)


# -- rows -----------------------------------------------------------------------------------------

class Rows:
    """Sampled frames of several matches: features, source match, block, truth and P(dead)."""

    def __init__(self):
        self.parts = []

    def add(self, match, X, graded, subset, step, offset, dead, p_dead, source):
        mask = graded & (match.ball_seen if subset == "seen" else ~match.ball_seen)
        pick = np.flatnonzero(mask)[offset::step]
        block = (pick // int(round(BLOCK_S * match.fps))).astype(np.int64)
        self.parts.append(dict(X=X[pick], match=np.array([match.name] * len(pick), dtype=object),
                               group=np.array(["{}:{}".format(match.name, b) for b in block],
                                              dtype=object),
                               dead=(dead[pick] if dead is not None
                                     else np.full(len(pick), -1, dtype=np.int8)),
                               p_dead=p_dead[pick], source=np.array([source] * len(pick))))

    def table(self):
        return {key: np.concatenate([p[key] for p in self.parts]) for key in self.parts[0]}


def balanced_weights(target, match):
    """Each class weighs the same; inside each class each match weighs the same."""
    w = np.zeros(len(target))
    for cls in (True, False):
        names = np.unique(match[target == cls])
        for name in names:
            sel = (target == cls) & (match == name)
            w[sel] = 1.0 / (len(names) * sel.sum())
    return w * len(target) / w.sum()


# -- classifier -----------------------------------------------------------------------------------

def learner():
    """deadmodel's learner at half the rounds and twice the rate: it only tells sources apart."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = deadmodel.PARAMS
    return HistGradientBoostingClassifier(
        max_iter=p["max_iter"] // 2, learning_rate=p["learning_rate"] * 2,
        max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p["min_samples_leaf"],
        l2_regularization=p["l2"], early_stopping=False, random_state=0)


def wauc(y, score, w):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(score)
    if ok.sum() < 2 or y[ok].all() or not y[ok].any():
        return float("nan")
    return float(roc_auc_score(y[ok], score[ok], sample_weight=w[ok]))


def cross_fit(X, y, w, groups):
    """(out-of-fold P(target), [(model, test index)]) over FOLDS group folds."""
    from sklearn.model_selection import GroupKFold
    oof = np.full(len(y), np.nan)
    folds = []
    for train, test in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        model = learner().fit(X[train], y[train], sample_weight=w[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
        folds.append((model, test))
    return oof, folds


def shuffled_auc(X, y, w, folds, cols):
    """AUC with the columns `cols` shuffled together within each test fold."""
    rng = np.random.RandomState(SHUFFLE_SEED)
    score = np.full(len(y), np.nan)
    for model, test in folds:
        Xt = X[test].copy()
        Xt[:, cols] = Xt[rng.permutation(len(test))][:, cols]
        score[test] = model.predict_proba(Xt)[:, 1]
    return wauc(y, score, w)


# -- report ---------------------------------------------------------------------------------------

def group_columns(names):
    """{group: column indices} for the groups with a column in `names`."""
    out = {}
    for group, members in GROUPS.items():
        cols = [names.index(m) for m in members if m in names]
        if cols:
            out[group] = cols
    return out


def odd_one_out(table, names):
    """({match: AUC of that proxy match against the other proxy matches},
    {group: [shuffle drop per match]}, [max over matches of |AUC - 0.5| per feature alone])."""
    proxy = table["source"] == "proxy"
    X, match, groups = table["X"][proxy], table["match"][proxy], table["group"][proxy]
    aucs, drops = {}, {g: [] for g in group_columns(names)}
    alone = np.zeros(X.shape[1])
    for name in sorted(np.unique(match)):
        y = match == name
        w = balanced_weights(y, match)
        oof, folds = cross_fit(X, y, w, groups)
        aucs[name] = wauc(y, oof, w)
        for group, cols in group_columns(names).items():
            drops[group].append(aucs[name] - shuffled_auc(X, y, w, folds, cols))
        for j in range(X.shape[1]):
            alone[j] = max(alone[j], abs(wauc(y, X[:, j].astype(float), w) - 0.5))
    return aucs, drops, alone


def drivers(table, names, y, w, folds, base, alone, odd_drops):
    X, groups = table["X"], table["group"]
    print("  {:<20} {:>9} {:>9} {:>9} {:>9}   {:<9}  {}".format(
        "feature group", "drop", "odd max", "excess", "alone", "set by", "R2"))
    rows = []
    for group, cols in group_columns(names).items():
        drop = base - shuffled_auc(X, y, w, folds, cols)
        own = (wauc(y, cross_fit(X[:, cols], y, w, groups)[0], w) if alone
               else float("nan"))
        rows.append((drop, group, own, max(odd_drops[group])))
    for drop, group, own, odd in sorted(rows, reverse=True):
        print("  {:<20} {:>9.3f} {:>9.3f} {:>+9.3f} {:>9.3f}   {:<9}  {}".format(
            group, drop, odd, drop - odd, own,
            "simulator" if group in SIMULATOR_GROUPS else "play",
            "DRIVER" if drop - odd >= DRIVER_MARGIN else ""))


def per_feature(table, names, y, w, target_name, odd_alone):
    X = table["X"]
    print("  {:<16} {:>10} {:>10} {:>7} {:>7} {:>7} {:>8}  {}".format(
        "feature", "p50 " + target_name[:5], "p50 proxy", "nan T", "nan P", "AUC", "odd max",
        "R2b"))
    for j, name in enumerate(names):
        v = X[:, j].astype(float)
        ok = np.isfinite(v)
        med_t = np.median(v[y & ok]) if (y & ok).any() else float("nan")
        med_p = np.median(v[~y & ok]) if (~y & ok).any() else float("nan")
        a = wauc(y, v, w)
        print("  {:<16} {:>10.3f} {:>10.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>8.3f}  {}".format(
            name, med_t, med_p, 1 - ok[y].mean(), 1 - ok[~y].mean(), a, 0.5 + odd_alone[j],
            "SHIFT" if abs(a - 0.5) - odd_alone[j] >= DRIVER_MARGIN else ""))


def covariate_shift(table, y, oof, w):
    proxy = ~y
    p = np.clip(oof[proxy], CLIP, 1 - CLIP)
    base_w = w[proxy]
    shift_w = base_w * p / (1 - p)
    dead = table["dead"][proxy].astype(float)
    p_dead = table["p_dead"][proxy]
    ess = shift_w.sum() ** 2 / np.sum(shift_w ** 2) / len(shift_w)
    target_p_dead = np.nanmean(table["p_dead"][y])

    def avg(values, weights):
        ok = np.isfinite(values)
        return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok]))

    print("  proxy truth D share:  as is {:.3f}   reweighted {:.3f}   (effective rows {:.2f} of "
          "the proxy's)".format(avg(dead, base_w), avg(dead, shift_w), ess))
    print("  model mean P(dead):   proxy as is {:.3f}   proxy reweighted {:.3f}   target {:.3f}"
          .format(avg(p_dead, base_w), avg(p_dead, shift_w), target_p_dead))
    q = np.argsort(np.argsort(oof[proxy], kind="stable"), kind="stable") * 5 // len(p)
    shift_p_dead = avg(p_dead, shift_w)
    ok = abs(shift_p_dead - target_p_dead) <= 0.02 and ess >= 0.20
    print("  R4: reweighting {}".format(
        "usable" if ok else "NOT usable (P(dead) off by {:+.3f}, effective rows {:.2f})".format(
            shift_p_dead - target_p_dead, ess)))
    if (table["dead"][y] >= 0).all():
        print("  (sanity only) target truth D share: {:.3f}".format(np.mean(table["dead"][y])))
    print("  proxy D share by quintile of P(target), least to most target-like: " +
          "  ".join("{:.3f}".format(avg(dead[q == k], base_w[q == k])) for k in range(5)))


def analyse(table, names, subset, feature_set, target_name, baseline, odd_drops, odd_alone,
            alone=False):
    y = table["source"] == "target"
    w = balanced_weights(y, table["match"])
    oof, folds = cross_fit(table["X"], y, w, table["group"])
    base = wauc(y, oof, w)
    top = max(baseline.values())
    print("\n== {} frames, {} features: {} against the proxy ==".format(subset, feature_set,
                                                                        target_name))
    print("  rows: target {}, proxy {}".format(int(y.sum()), int((~y).sum())))
    print("  AUC {:.3f}   odd-one-out baseline: max {:.3f} (range {:.3f}..{:.3f})   R1: {}".format(
        base, top, min(baseline.values()), top,
        "within the baseline" if base <= top + MARGIN else "ABOVE the baseline by {:+.3f}".format(
            base - top)))
    drivers(table, names, y, w, folds, base, alone, odd_drops)
    per_feature(table, names, y, w, target_name, odd_alone)
    covariate_shift(table, y, oof, w)


def build_tables(target, seeds):
    """({subset: table}, names) with the proxy and, unless target is None, the target; the
    "context" columns, of which the "deadmodel" set is the first len(FEATURE_NAMES)."""
    from solution.sources import idsse
    from solution.simulate import add_ball_noise, degrade
    from tools.deadball_check import STRESS, inputs
    tables = {s: Rows() for s in SUBSETS}
    names = None
    n = len(seeds)
    for truth, match, _, seed in inputs(idsse.DEV_IDS, seeds, False, "keep2"):
        X, names = match_features(match, "context")
        graded = match.in_play_mask() & np.isin(truth, (pm.H, pm.A, pm.D))
        p_dead = deadmodel.dead_probability(match)
        for s in SUBSETS:
            tables[s].add(match, X, graded, s, STEP * n, STEP * seeds.index(seed),
                          (truth == pm.D).astype(np.int8), p_dead, "proxy")
    if target == "bar-ath":
        from solution.sources import atd
        match = atd.load()
        X, names = match_features(match, "context")
        p_dead = deadmodel.dead_probability(match)
        for s in SUBSETS:
            tables[s].add(match, X, match.in_play_mask(), s, STEP, 0, None, p_dead, "target")
    elif target == "v0":
        for match_id in idsse.DEV_IDS:
            raw = idsse.load(match_id, labels="keep2")
            match = degrade(raw, 7, version="v0")
            match = add_ball_noise(match, *STRESS[:2], seed=7, smooth_s=STRESS[2])
            X, names = match_features(match, "context")
            graded = match.in_play_mask() & np.isin(raw.labels, (pm.H, pm.A, pm.D))
            p_dead = deadmodel.dead_probability(match)
            for s in SUBSETS:
                tables[s].add(match, X, graded, s, STEP, 0, (raw.labels == pm.D).astype(np.int8),
                              p_dead, "target")
    return {s: t.table() for s, t in tables.items()}, names


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--proxy-only", action="store_true",
                        help="the odd-one-out baseline only; needs no data/")
    parser.add_argument("--sanity", action="store_true",
                        help="target: the dev matches degraded by simulator v0 (no data/)")
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS),
                        choices=FEATURE_SETS)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--alone", action="store_true",
                        help="also fit a classifier per feature group alone (slow)")
    args = parser.parse_args()
    target = None if args.proxy_only else ("v0" if args.sanity else "bar-ath")
    target_name = {"bar-ath": "BAR-ATH", "v0": "simulator v0", None: ""}[target]
    print("Adversarial check: IDSSE dev x seeds {} (simulator v1 + D12 ball stress) as the proxy; "
          "target: {}".format(args.seeds, target_name or "none (baseline only)"))
    started = time.time()
    tables, names = build_tables(target, args.seeds)
    print("(rows built in {:.0f} s)".format(time.time() - started))
    for feature_set in args.feature_sets:
        width = len(deadmodel.FEATURE_NAMES) if feature_set == "deadmodel" else len(names)
        for subset in args.subsets:
            began = time.time()
            full = tables[subset]
            cols = [j for j in range(width) if np.isfinite(full["X"][:, j]).any()]
            table = dict(full, X=full["X"][:, cols])
            used = [names[j] for j in cols]
            baseline, odd_drops, odd_alone = odd_one_out(table, used)
            print("\n-- {} frames, {} features: odd-one-out AUC per dev match: {}".format(
                subset, feature_set, "  ".join("{} {:.3f}".format(k.replace("idsse_", ""), v)
                                               for k, v in baseline.items())))
            if target is not None:
                analyse(table, used, subset, feature_set, target_name, baseline, odd_drops,
                        odd_alone, args.alone)
            print("  ({:.0f} s)".format(time.time() - began))
    print("\n(total {:.0f} s)".format(time.time() - started))


if __name__ == "__main__":
    main()
