"""The TV cut-away cue (E13, D27): is the broadcast showing the live pitch at this moment?

Per tactical frame, `tv` is 1 when the broadcast is off the live pitch (cut away, or airing a
replay), 0 when it shows the live pitch, -1 when unknown (no broadcast mapped there, or outside
the halves). The decoder reads it with tv_w > 0 (decoder.block_emissions), through a 2 x 2 table
of log P(on / off | alive / dead) fitted here.

BAR-ATH: bar_ath_tv() reads the alignment `make broadcast-align` writes.

Proxy: the IDSSE matches have no TV feed, so synthetic() imitates one from the truth, with the
rates and timing BAR-ATH's broadcast shows against our predicted dead ball (EXPERIMENTS section
17, tools/broadcast_align.py section G). Fixed on 2026-11-06, before the first proxy run:
  - each true stoppage (a D run) gets a cut-away with probability P_CUT by its length;
  - the cut-away is one chunk covering COVER of the stoppage, the uncovered rest split with a
    front share drawn from FRONT (the TV cuts away soon after the stoppage starts and returns
    before the restart: G's lags);
  - "pessimistic" (the decision variant): with probability SPILL_P the cut-away runs on past
    the restart for SPILL_S seconds (all of G's off-pitch runs next to a predicted D run taken
    as TV lag, none as our error), and FALSE_SHARE of live frames get other cut-aways (all of
    G's isolated off-pitch runs inside predicted live taken as close-ups during play);
  - "optimistic" (reported): no spill, and the implied P(off | live) of 0.01 for false ones.
G measured these against our *predicted* dead ball, which includes false stoppages without a
cut-away, so P_CUT for short stoppages is if anything low. Not imitated: BAR-ATH's 5.2% of
graded frames with no broadcast mapped (the proxy cue is known everywhere).
"""

import functools

import numpy as np

from solution.match import pm

# (shortest s, longest s, P(any cut-away), share of the stoppage covered when cut away)
PROFILE = ((0.0, 5.0, 0.171, 0.649), (5.0, 15.0, 0.520, 0.775), (15.0, 30.0, 0.838, 0.665),
           (30.0, np.inf, 1.000, 0.759))
FRONT = (0.0, 0.5)
VARIANTS = {
    "pessimistic": {"spill_p": 0.494, "spill_s": (2.1, 8.7), "false_share": 0.0229,
                    "false_s": (2.39, 11.13)},
    "optimistic": {"spill_p": 0.0, "spill_s": (2.1, 8.7), "false_share": 0.010,
                   "false_s": (2.39, 11.13)},
    # D31 (decides): spill_p and false_share from tools/cutaway_calibrate.py, the grid pair
    # whose cue splits 0.610 / 0.096 against the reference's prediction (BAR-ATH: 0.612 / 0.096)
    "calibrated": {"spill_p": 0.15, "spill_s": (2.1, 8.7), "false_share": 0.0200,
                   "false_s": (2.39, 11.13)},
}
STREAM = 12               # simulate._rng stream; no other stream uses it
TABLE_SEEDS = (0, 1, 2)   # synthetic cues the table is fitted on, per dev match
EPS = 1e-4                # probability floor in the table


def synthetic(match, labels, seed, variant="pessimistic"):
    """(T,) int8 tv cue imitated from `labels` (truth) for one match; -1 outside the halves.
    `variant` names a VARIANTS entry, or is a dict of the same keys (calibration only)."""
    from solution.simulate import _rng
    spec = VARIANTS[variant] if isinstance(variant, str) else variant
    rng = _rng(match, seed, STREAM)
    fps = match.fps
    n = len(labels)
    graded = np.zeros(n, dtype=bool)
    for start, stop in match.halves:
        graded[start:stop] = True
    off = np.zeros(n, dtype=bool)
    for start, stop, code in pm.segments(labels, pm.valid_runs(labels)):
        if code != pm.D:
            continue
        length = stop - start
        p_cut, cover = next((p, c) for lo, hi, p, c in PROFILE
                            if lo * fps <= length < hi * fps)
        if rng.rand() >= p_cut:
            continue
        k = int(round(cover * length))
        front = int(round(rng.uniform(*FRONT) * (length - k)))
        end = start + front + k
        if rng.rand() < spec["spill_p"]:
            end = stop + int(round(rng.uniform(*spec["spill_s"]) * fps))
        off[start + front:min(end, n)] = True
    live = np.flatnonzero(graded & np.isin(labels, (pm.H, pm.A)))
    if len(live):
        mean_len = np.mean(spec["false_s"]) * fps
        for _ in range(int(round(spec["false_share"] * len(live) / mean_len))):
            a = int(live[rng.randint(len(live))])
            off[a:a + int(round(rng.uniform(*spec["false_s"]) * fps))] = True
    tv = np.where(off, 1, 0).astype(np.int8)
    tv[~graded] = -1
    return tv


def fit_table(pairs):
    """(2, 2) log P(tv | state) from [(labels, tv)]: row 0 alive, 1 dead; column 0 on pitch,
    1 off. Frames with tv -1 or a label outside H / A / D are skipped."""
    counts = np.zeros((2, 2))
    for labels, tv in pairs:
        known = (tv >= 0) & np.isin(labels, (pm.H, pm.A, pm.D))
        dead = labels == pm.D
        for row, mask in ((0, known & ~dead), (1, known & dead)):
            counts[row, 1] += (mask & (tv == 1)).sum()
            counts[row, 0] += (mask & (tv == 0)).sum()
    p_off = np.clip(counts[:, 1] / np.maximum(counts.sum(axis=1), 1), EPS, 1 - EPS)
    return np.log(np.column_stack([1 - p_off, p_off]))


@functools.lru_cache(maxsize=None)
def idsse_table(exclude="", variant="pessimistic", labels="keep2"):
    """fit_table on the synthetic cues of the IDSSE dev matches other than `exclude` (a Match
    name): leave-one-match-out on the proxy, all five on BAR-ATH. Never the hold-out (D3)."""
    from solution.sources import idsse
    ids = [m for m in idsse.DEV_IDS if "idsse_" + m != exclude]
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    pairs = []
    for m in ids:
        match = idsse.load(m, labels=labels)
        pairs += [(match.labels, synthetic(match, match.labels, seed, variant))
                  for seed in TABLE_SEEDS]
    return fit_table(pairs)


def from_alignment(state):
    """tv from solution/broadcast.align's per-tactical-frame `state`."""
    from solution import broadcast as bc
    tv = np.full(len(state), -1, dtype=np.int8)
    tv[np.isin(state, (bc.LIVE, bc.OTHER))] = 0
    tv[np.isin(state, (bc.CUT, bc.REPLAY))] = 1
    return tv


def bar_ath_tv(align_path=None):
    """BAR-ATH's tv cue from cache/broadcast/BAR-ATH_align.npz (`make broadcast-align`)."""
    import os
    from solution.sources.fused import ALIGN_PATH
    path = align_path or ALIGN_PATH
    if not os.path.isfile(path):
        raise IOError("no broadcast alignment at {}: run `make broadcast-align` first".format(path))
    return from_alignment(np.load(path, allow_pickle=False)["state"])


def cue_for(match, variant):
    """The tv cue a method reads for `match`:
      - match.notes["tv"] if attached (tools/cutaway_check.py);
      - on BAR-ATH, the real cue from the broadcast alignment (bar_ath_tv);
      - on a degraded proxy match, the synthetic cue of `variant` made from its truth and its
        simulator seed (notes["degraded"]["seed"]) -- the same one cutaway_check attaches. This
        reads the truth by construction (it imitates a TV feed the proxy does not have); BAR-ATH
        has no labels, so it can never take this path there.
    """
    from games import GAME
    if "tv" in match.notes:
        return np.asarray(match.notes["tv"], dtype=np.int8)
    if match.name == GAME:
        return bar_ath_tv()
    seed = (match.notes.get("degraded") or {}).get("seed")
    if match.labels is not None and seed is not None:
        return synthetic(match, match.labels, seed, variant)
    raise ValueError("{}: no TV cut-away cue (attach notes['tv'], or run `make broadcast-align` "
                     "for BAR-ATH)".format(match.name))
