"""The referee's whistle as a dead-ball cue (denis, 2026-11-12; reopens D26 and D45 at his call).

Per tactical frame, a category from two things:
  - time since the last whistle onset: [0, 3 s), [3, 15 s), [15, 45 s), 45 s or more / none;
  - where the ball is: at least MID_M from every line ("mid"), nearer ("line"), or unknown (not
    seen in the last PLACE_S).
The decoder reads it with whistle_w > 0 (decoder.block_emissions), through a 2 x N_CAT table of
log P(category | alive / dead) fitted on the IDSSE dev matches here. So "a whistle just went"
argues for dead ball, and "no whistle for a while, ball in mid-pitch" argues for live play
(denis's idea, D45) -- each by as much as the fitted table says, not by a hand-set rule. With
`whistle_end` the method also applies D33's end-of-half rule on the same whistles
(end_of_half): dead ball from the last whistle after 45:00 to the half's end, when it is within
END_MAX_S of the end.

BAR-ATH: the onsets are the whistles of the self-calibrating detector (whistle.match_whistles,
D47) on the tactical audio: it finds the referee's whistle from the kickoffs the tracking shows,
with no person in the loop. Blasts within GROUP_S are grouped into one onset. Nothing is fitted
on BAR-ATH beyond that calibration, and no frame or time of the match is written anywhere.

Proxy: the IDSSE matches have no audio, so synthetic() imitates a detector from the truth and
the DFL event feed. Fixed on 2026-11-12, before the first proxy run:
  - the referee whistles at the start of every true D stretch whose restart is a free kick
    (fouls, offsides), penalty, goal or dropped ball (tools/deadball_check.WHISTLED), at the
    kickoff ending a goal's stoppage (KICKOFF_LEAD_S before it), at a share RESTART_P of
    free-kick restarts, and at each half's last frame (half-time, full time);
  - the detector misses each with probability `miss`, times the rest with normal jitter
    `jitter_s`, and adds `false_per_min` false whistles (fans) at random graded frames.
  - "prior" (the decision variant, D47): miss 0.3, false 0.1/min, jitter 0.3 s, set a priori on
    2026-11-16 (a detector that misses about a third and adds a false whistle every ten
    minutes), not from anything heard or measured on BAR-ATH. "pessimistic" (reported): miss
    0.5, false 0.2/min, jitter 0.5 s. "matched" (D47 update): false 0.5/min, the upper bound
    D47's Step 0 gives on BAR-ATH without labels. "none" (the control): no whistles, so only the place is
    read.
Not imitated: multiple blasts per stoppage (grouped away on BAR-ATH), whistles for stoppages
the proxy truth does not end in a WHISTLED restart, the audio delay to the stand (~0.2 s).
"""

import functools

import numpy as np

from solution.match import pm

SINCE_S = (3.0, 15.0, 45.0)     # bin edges of the time since the last whistle onset
MID_M = 10.0                    # the ball this far from every line is in mid-pitch
PLACE_S = 10.0                  # the ball's place: its last seen position up to this long ago
N_SINCE, N_PLACE = len(SINCE_S) + 1, 3
N_CAT = N_SINCE * N_PLACE
GROUP_S = 3.0                   # BAR-ATH: blasts this close are one whistle (whistle_check)
KICKOFF_LEAD_S = 1.5            # the kickoff whistle comes this long before the kick (D44)
RESTART_P = 0.25                # share of free-kick restarts the referee whistles (a guess)
LAW7_S = 45 * 60.0              # end_of_half looks for whistles this long after a half starts
END_MAX_S = 120.0               # ... and uses one only this close to the half's end
END_GUARD_MS = 1.7              # D48 guard: after the pick, median collective speed at most this
                                # (halfway between IDSSE's dead-ball 1.02 and open-play 2.33 m/s,
                                # solution/kickoff.py): the players are not playing
VARIANTS = {
    "prior": {"miss": 0.3, "false_per_min": 0.1, "jitter_s": 0.3},
    "pessimistic": {"miss": 0.5, "false_per_min": 0.2, "jitter_s": 0.5},
    # D47 Step 0 (aggregates, no labels): up to ~0.5 whistle onsets per minute on BAR-ATH have
    # no decoded stoppage starting near them, so 0.5/min is the false rate's upper bound
    "matched": {"miss": 0.3, "false_per_min": 0.5, "jitter_s": 0.3},
    # the control: no whistles anywhere (BAR-ATH too), so only the ball's place is read
    "none": {"miss": 1.0, "false_per_min": 0.0, "jitter_s": 0.0},
}
STREAM = 13                     # simulate._rng stream; no other stream uses it
TABLE_SEEDS = (0, 1, 2)         # synthetic cues the table is fitted on, per dev match
EPS = 1e-4                      # probability floor in the table


# -- the category per frame ----------------------------------------------------------------------

def categories(match, onsets):
    """(T,) int8: since-bin * N_PLACE + place for each graded frame, -1 outside the halves.
    Place: 0 mid-pitch, 1 near a line, 2 unknown."""
    n, fps = match.n_frames, match.fps
    onsets = np.sort(np.asarray(onsets, dtype=np.int64))
    t = np.arange(n)
    last = np.searchsorted(onsets, t, side="right") - 1
    since = np.full(n, np.inf)
    has = last >= 0
    since[has] = (t[has] - onsets[last[has]]) / float(fps)
    since_bin = np.searchsorted(np.asarray(SINCE_S), since, side="right")

    seen = match.ball_seen
    idx = np.maximum.accumulate(np.where(seen, t, -1))
    known = (idx >= 0) & (t - idx <= PLACE_S * fps)
    k = np.clip(idx, 0, n - 1)
    length, width = match.pitch
    x, y = match.ball_x[k].astype(float), match.ball_y[k].astype(float)
    margin = np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))
    place = np.where(~known, 2, np.where(margin >= MID_M, 0, 1))

    out = np.full(n, -1, dtype=np.int8)
    graded = match.in_play_mask()
    out[graded] = (since_bin * N_PLACE + place)[graded]
    return out


def fit_table(pairs):
    """(2, N_CAT) log P(category | state) from [(labels, categories)]: row 0 alive, 1 dead."""
    counts = np.zeros((2, N_CAT))
    for labels, cats in pairs:
        known = (cats >= 0) & np.isin(labels, (pm.H, pm.A, pm.D))
        dead = labels == pm.D
        for row, mask in ((0, known & ~dead), (1, known & dead)):
            counts[row] += np.bincount(cats[mask], minlength=N_CAT)
    p = np.clip(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1), EPS, 1.0)
    return np.log(p / p.sum(axis=1, keepdims=True))


# -- whistles: BAR-ATH's from the audio, the proxy's imitated ------------------------------------

def group_onsets(events, fps, gap_s=GROUP_S):
    """First frame of each group of events (dicts with start, stop) whose gaps are <= gap_s."""
    onsets, last_stop = [], None
    for e in sorted(events, key=lambda e: e["start"]):
        if last_stop is None or e["start"] - last_stop > gap_s * fps:
            onsets.append(int(e["start"]))
        last_stop = e["stop"] if last_stop is None else max(last_stop, e["stop"])
    return np.array(onsets, dtype=np.int64)


_BAR_ATH = {}


def bar_ath_onsets(match):
    """BAR-ATH's whistle onsets: whistle.match_whistles on the tactical audio and this match's
    tracking, grouped (needs data/; the per-frame peaks are cached in cache/whistle/). Empty when
    the detector finds no calibration."""
    key = (match.name, match.n_frames)
    if key not in _BAR_ATH:
        import os
        import games
        from solution import whistle
        path = games.video_path(games.GAME)
        if not os.path.isfile(path):
            raise IOError("no tactical video at {} for the whistle cue".format(path))
        features = whistle.peak_video_features(path, match.n_frames, match.fps)
        _, found = whistle.match_whistles(features, match)
        _BAR_ATH[key] = group_onsets(found, match.fps)
    return _BAR_ATH[key]


def referee_whistles(match):
    """[(frame, kind)] where the referee whistles on a proxy match, from its truth and the DFL
    event feed (mapped through the proxy league's tempo and pre-kickoff stresses)."""
    from solution.simulate import retime_frame
    from solution.sources import idsse
    from tools.deadball_check import WHISTLED, dead_stretches, restart_kinds
    match_id = match.name.replace("idsse_", "")
    events = idsse.restarts(match_id)
    if "tempo" in match.notes:
        events = [(retime_frame(f, match.notes["tempo"]), k) for f, k in events]
    if "prekickoff" in match.notes:
        info = match.notes["prekickoff"]
        at = match.halves[info["half"]][0]
        events = [(f + info["frames"] if f >= at else f, k) for f, k in events]
    events = [(walkoff_frame(f, match), k) for f, k in events]
    labels = match.labels
    runs = pm.valid_runs(labels)
    stretches = dead_stretches(labels, runs)
    kinds = restart_kinds(stretches, events, match.fps)
    out = []
    for (a, b), kind in zip(stretches, kinds):
        if kind in WHISTLED:
            out.append((a, "stop"))
        if kind == "goal":
            out.append((b - int(round(KICKOFF_LEAD_S * match.fps)), "kickoff"))
        elif kind == "free_kick":
            out.append((b, "restart"))
    for _, stop in half_ends(match):
        out.append((stop - 1, "end"))
    return out


def walkoff_frame(frame, match):
    """A frame of the match before simulate.add_walkoff, in the match after it."""
    info = match.notes.get("walkoff")
    if not info:
        return frame
    return frame + info["frames"] * sum(frame >= at for at in info["at"])


def half_ends(match):
    """[(start, end of play)] per half: the half itself, or up to its walk-off when
    simulate.add_walkoff added one (the referee's final whistle is at the end of play)."""
    info = match.notes.get("walkoff")
    if not info:
        return list(match.halves)
    return [(a, end) for (a, _), end in zip(match.halves, info["ends"])]


def synthetic(match, seed, variant="prior"):
    """Whistle onsets imitated for a proxy match (its labels are its truth); see the module doc."""
    from solution.simulate import _rng
    spec = VARIANTS[variant] if isinstance(variant, str) else variant
    rng = _rng(match, seed, STREAM)
    fps = match.fps
    onsets = []
    for frame, kind in referee_whistles(match):
        if kind == "restart" and rng.rand() >= RESTART_P:
            continue
        if rng.rand() < spec["miss"]:
            continue
        onsets.append(int(round(frame + rng.normal(0.0, spec["jitter_s"]) * fps)))
    graded = np.flatnonzero(match.in_play_mask())
    n_false = rng.poisson(spec["false_per_min"] * len(graded) / float(fps) / 60.0)
    if n_false:
        onsets += [int(f) for f in rng.choice(graded, n_false, replace=False)]
    return np.clip(np.sort(np.array(onsets, dtype=np.int64)), 0, match.n_frames - 1)


def onsets_for(match, variant="prior"):
    """The whistle onsets a method reads: notes["whistles"] if attached; BAR-ATH's from the
    audio; on a degraded proxy match, synthetic() with its simulator seed (reads the truth by
    construction; BAR-ATH has no labels, so it can never take that path)."""
    from games import GAME
    if "whistles" in match.notes:
        return np.asarray(match.notes["whistles"], dtype=np.int64)
    if variant == "none":
        return np.array([], dtype=np.int64)
    if match.name == GAME:
        return bar_ath_onsets(match)
    seed = (match.notes.get("degraded") or {}).get("seed")
    if match.labels is not None and seed is not None:
        return synthetic(match, seed, variant)
    raise ValueError("{}: no whistles (attach notes['whistles'])".format(match.name))


@functools.lru_cache(maxsize=None)
def idsse_table(exclude="", variant="prior"):
    """fit_table on synthetic whistles on the degraded IDSSE dev matches other than `exclude` (a
    Match name), TABLE_SEEDS each: leave-one-match-out on the proxy, all five on BAR-ATH. Never
    the hold-out (D3). D is the same in every label variant, so keep2 serves all."""
    from solution.simulate import degrade
    from solution.sources import idsse
    ids = [m for m in idsse.DEV_IDS if "idsse_" + m != exclude]
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    pairs = []
    for m in ids:
        raw = idsse.load(m, labels="keep2")
        for seed in TABLE_SEEDS:
            match = degrade(raw, seed)
            pairs.append((match.labels, categories(match, synthetic(match, seed, variant))))
    return fit_table(pairs)


def log_lik(match, variant="prior"):
    """(T, 3) log P(category | H, A, D) per frame (alive row for H and A), NaN where unknown."""
    from solution.decoder import A, D, H
    cats = categories(match, onsets_for(match, variant))
    table = idsse_table(match.name, variant)
    out = np.full((match.n_frames, 3), np.nan)
    known = cats >= 0
    out[known, H] = out[known, A] = table[0, cats[known]]
    out[known, D] = table[1, cats[known]]
    return out


def end_picks(match, onsets, guard=False):
    """Per half, D33's end-of-half pick and whether the rule fires there: a dict with `pick` (the
    first onset of the last whistle group after LAW7_S, or None), `before_s` (seconds from it to
    the half's end), `speed` (median collective speed from it to the end, m/s; with `guard`) and
    `fires` (within END_MAX_S of the end and, with `guard` (D48), the players are not playing
    after it: that median at most END_GUARD_MS; a missed final whistle would otherwise let a late
    foul's whistle end the half early)."""
    speed = None
    if guard:
        from solution.simulate import mean_player_speed
        speed = mean_player_speed(match)
    fps = match.fps
    onsets = np.sort(np.asarray(onsets, dtype=np.int64))
    out = []
    for start, stop in match.halves:
        row = {"pick": None, "before_s": None, "speed": None, "fires": False}
        window = onsets[(onsets >= start + LAW7_S * fps) & (onsets < stop)]
        if len(window):
            pick = int(window[0])
            for a, b in zip(window[:-1], window[1:]):
                if b - a > GROUP_S * fps:
                    pick = int(b)
            row.update(pick=pick, before_s=(stop - pick) / float(fps))
            fires = stop - pick <= END_MAX_S * fps
            if speed is not None:
                after = speed[pick:stop]
                after = after[np.isfinite(after)]
                row["speed"] = float(np.median(after)) if len(after) else None
                fires = fires and row["speed"] is not None and row["speed"] <= END_GUARD_MS
            row["fires"] = bool(fires)
        out.append(row)
    return out


def end_of_half(states, match, onsets, dead, guard=False):
    """D33's end-of-half rule: in each half where end_picks() fires, `dead` from the pick to the
    half's end. `guard` adds D48's tracking guard (see end_picks)."""
    out = states.copy()
    for (_, stop), row in zip(match.halves, end_picks(match, onsets, guard)):
        if row["fires"]:
            out[row["pick"]:stop] = dead
    return out
