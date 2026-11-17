"""C4, first version: a semi-Markov segment decoder over {H, A, D} (PLAN.md C4, backlog E1).

It replaces baseline_smooth50's majority vote. The evidence is what the baseline already uses,
plus one dead-ball cue:
  - the team of the tracked player nearest the ball (methods.nearest_player_codes);
  - whether the ball is seen at all;
  - collective player speed (speed_feature: simulate.mean_player_speed, optionally as a rank
    within the match). Slow play is the strongest dead-ball
    cue (notebooks/eda/FINDINGS.md, point 1). Ball position beyond a line is NOT used, because
    the proxy league overstates it (DECISIONS D7);
  - optionally (touch_w > 0, the touch_markov method) the team holding the ball by last touch,
    from solution/touch.py (E3, D10). It enters like the nearest player, with its own agreement;
  - optionally (deadball_w > 0) a dead-ball cue from solution/deadball.py (C2b): a player
    standing beyond a line. It enters like speed, with fitted P(cue | alive) and P(cue | dead);
  - optionally (team_w > 0) team cues from the players inside ball gaps, solution/gapteam.py
    (E5): fitted P(cue | team in possession) and P(cue | dead), per cue.

Decoding runs at 5 Hz, on blocks of 5 frames, and each block's state is copied back to its
frames. A segment of state k and length d blocks scores
    w * (sum of the block emissions for k) + log P_k(d) + the transition score into it,
and the decoder returns the best-scoring segmentation of each half, exactly. Transition costs are
separate for H<->A (`switch`), alive->D (`dead`) and D->team (`restart`). Same-state segments
may follow each other at no cost, which only matters for runs longer than the duration table;
the scorer merges them into one spell anyway.

Fitted on labelled external matches (fit_priors), leave-one-match-out on the IDSSE dev split
(PLAN.md L1), never on the held-out matches (D3) or the public window:
  - duration priors: a log-normal per state class (possession, shared by H and A; dead ball),
    discretised to blocks, with a minimum duration at the 1st percentile;
  - p(nearest player is on the team in possession), on clean positions;
  - the speed log-likelihood under alive and dead play, as a histogram, on clean positions.

numpy only, no randomness: the same inputs always give the same output.
"""

import math
from functools import lru_cache

import numpy as np

from solution.match import AWAY, HOME, pm
from solution.simulate import mean_player_speed

H, A, D = 0, 1, 2                       # decoder states, same order as the scorer's codes
assert (H, A, D) == (pm.H, pm.A, pm.D) and (H, A) == (HOME, AWAY)
BLOCK = 5                               # frames per decoding step: 5 Hz at 25 fps
MAX_SECONDS = 120.0                     # longest single segment; longer runs chain same-state
SPEED_EDGES = np.arange(0.0, 6.01, 0.25)   # m/s; the last bin takes everything faster
RANK_EDGES = np.linspace(0.0, 1.0, 21)[:-1]   # speed as its quantile within the match
MIN_QUANTILE = 0.01                     # minimum duration = this quantile of the fitted spells

DEFAULTS = {
    "w": 1.0,           # weight of the block evidence against the duration priors
    "switch": 0.0,      # cost of H <-> A
    "dead": 0.0,        # cost of alive -> D
    "restart": 0.0,     # cost of D -> H or A
    "no_ball": 0.0,     # log p(no ball | D) / p(no ball | alive), per block; see below
    "speed_lag": 2,     # frames either side for speed from positions (mean_player_speed)
    "speed_scale": "rank",  # "rank": speed as its quantile among the match's frames; "ms": m/s
    "speed_lead_s": 0.0,  # read collective speed this far ahead: players react to the whistle
    "touch_w": 0.0,     # weight of the touch engine's holder term (solution/touch.py); 0 = off
    "nearest_fallback": False,  # True: the nearest player counts only where no team holds
    "deadball_w": 0.0,  # weight of solution/deadball.py's cue (a player beyond a line); 0 = off
    "team_w": 0.0,      # weight of solution/gapteam.py's team cues; 0 = off
    "team_where": "gaps",  # read them on frames without a seen ball ("gaps") or on all frames
    "team_dead": "fit",  # D's likelihood: fitted ("fit"), or the better team's ("max": no say)
    "bonus_w": 0.0,     # nats added to an H <-> A switch into a team at its events; 0 = off
    "bonus_shift_s": 0.0,   # events are moved this far (negative: earlier) before use
    "bonus_spread_s": 0.4,  # ... and count for switches within this of them
    "keeper_w": 0.0,    # weight of solution/keepermodel.py's per-state log-likelihoods; 0 = off
    "whistle_w": 0.0,   # weight of solution/whistlecue.py's per-state log-likelihoods (the
                        # referee's whistle; denis 2026-11-12); 0 = off
    "frame_w": 0.0,     # weight of solution/framemodel.py's P(home | alive) (C3); 0 = off
    "frame_where": "all",   # read it on every frame ("all"), only without a seen ball ("gaps"),
                            # or only with one ("seen")
    "deadm_w": 0.0,     # weight of solution/deadmodel.py's log P(x | dead) / P(x | alive), which
                        # then takes the place of the speed likelihood (it reads the speed rank
                        # itself); 0 = off (EXPERIMENTS section 16)
    "deadm_where": "all",   # read it on every frame ("all") or only with a seen ball ("seen");
                            # elsewhere the speed likelihood stays
    "tv_w": 0.0,        # weight of the TV cut-away cue (solution/cutaway.py, E13, D27), added
                        # to every state through priors["tv_log"]; 0 = off
    "dead_min_s": None, # minimum stoppage length in seconds, where above the fitted 1st
                        # percentile; None = the fitted minimum (EXPERIMENTS section 22)
}
# speed_scale is "rank" because BAR-ATH plays about 10% slower than the IDSSE matches. In m/s
# the decoder then reads slow open play as dead ball. The rank is immune to that (D9).
# no_ball stays 0. In BAR-ATH the ball is missing on about a third of dead-ball frames, close
# to its overall 41% (FINDINGS), so a missing ball says little about a stoppage there; the
# simulator ties gaps to slow play instead. Neutral is the honest setting: a missing ball then
# only means "no team evidence", and speed carries the dead-ball signal.


# -- priors, fitted on labelled matches --------------------------------------------------------

def spell_lengths(labels):
    """{'possession': [...], 'dead': [...]} lengths in frames of the labelled runs, per half."""
    out = {"possession": [], "dead": []}
    for start, stop, code in pm.segments(labels, pm.valid_runs(labels)):
        out["dead" if code == pm.D else "possession"].append(stop - start)
    return out


def speed_feature(match, lag=2, scale="ms", lead_s=0.0):
    """Per-frame collective speed as the decoder reads it: m/s, or its quantile in the match.

    The quantile ("rank") is computed over the graded frames, so any monotone distortion of
    speed (tracking noise inflating every value, say) leaves it unchanged. It assumes the
    share of slow play is similar from match to match. With `lead_s`, frame t reads the speed
    of frame t + lead_s (NaN past the end): players slow down after the whistle and speed up
    after the restart, so speed trails the ball's state.
    """
    speed = mean_player_speed(match, lag)
    lead = int(round(lead_s * match.fps))
    if lead > 0:
        speed = np.concatenate((speed[lead:], np.full(lead, np.nan)))
    if scale == "ms":
        return speed
    if scale != "rank":
        raise ValueError("unknown speed scale {!r}".format(scale))
    out = np.full(len(speed), np.nan)
    use = match.in_play_mask() & np.isfinite(speed)
    ordered = np.sort(speed[use])
    out[use] = np.searchsorted(ordered, speed[use], side="right") / float(len(ordered))
    return out


def fit_priors(matches, nearest_fn, speed_fn, edges=SPEED_EDGES):
    """Duration, agreement and speed priors from clean, labelled matches.

    `nearest_fn(match)` is methods.nearest_player_codes, passed in so this module stays free of
    import cycles; `speed_fn(match)` gives the speed feature, binned on `edges`.
    """
    lengths = {"possession": [], "dead": []}
    agree = seen = 0
    speed_counts = np.zeros((2, len(edges)))                # rows: alive, dead
    for match in matches:
        labels = match.labels
        for key, values in spell_lengths(labels).items():
            lengths[key].extend(values)
        nearest = nearest_fn(match)
        alive = (labels == pm.H) | (labels == pm.A)
        ok = alive & (nearest >= 0)
        agree += int(np.sum(nearest[ok] == labels[ok]))
        seen += int(np.sum(ok))
        speed = speed_fn(match)
        for row, mask in enumerate((alive, labels == pm.D)):
            values = speed[mask & np.isfinite(speed)]
            speed_counts[row] += np.bincount(speed_bins(values, edges), minlength=len(edges))
    fps = matches[0].fps
    durations = {}
    for key, values in lengths.items():
        seconds = np.asarray(values, dtype=float) / fps
        logs = np.log(seconds)
        durations[key] = {"mu": float(logs.mean()), "sigma": float(logs.std()),
                          "min_s": float(np.quantile(seconds, MIN_QUANTILE)),
                          "n": len(values)}
    speed_counts += 1.0                                     # add-one smoothing
    speed_log = np.log(speed_counts / speed_counts.sum(axis=1, keepdims=True))
    return {"durations": durations, "agree": agree / float(seen),
            "speed_log": speed_log, "speed_edges": edges, "matches": [m.name for m in matches]}


def speed_bins(speed, edges=SPEED_EDGES):
    return np.clip(np.searchsorted(edges, speed, side="right") - 1, 0, len(edges) - 1)


def duration_log_prior(mu, sigma, min_s, block_s, max_blocks):
    """log P(d) for d = 1..max_blocks blocks: a discretised log-normal, -inf below min_s.

    Bin d covers [d - 0.5, d + 0.5) blocks; the first bin starts at 0 and the last takes the
    whole tail, so a segment of max_blocks stands for "this long or longer".
    """
    def cdf(seconds):
        if seconds <= 0:
            return 0.0
        return 0.5 * (1.0 + math.erf((math.log(seconds) - mu) / (sigma * math.sqrt(2.0))))
    edges = [cdf((d - 0.5) * block_s) for d in range(1, max_blocks + 1)] + [1.0]
    edges[0] = 0.0
    p = np.diff(edges)
    min_blocks = max(1, int(math.floor(min_s / block_s)))
    p[:min_blocks - 1] = 0.0
    with np.errstate(divide="ignore"):
        return np.log(p / p.sum())


# -- decoding ----------------------------------------------------------------------------------

def block_emissions(nearest, ball_seen, speed, priors, params, holder=None, cue=None,
                    team=None, frame=None, deadm=None, tv=None, keeper=None, whistle=None):
    """(n_blocks, 3) log-likelihoods of H, A, D for one half's frames, BLOCK frames per row.

    Each block counts as one observation: the fractions of its frames whose nearest player is
    home, away, or where the ball is missing, and its mean player speed. With `holder` (the
    touch engine's holding team per frame, -1 if none) and touch_w > 0, the fractions of
    frames held by home and away add a term of the same form as the nearest player's, with
    its own fitted agreement priors["holder_agree"]. With `cue` ((T, K) bool, deadball.cues)
    and deadball_w > 0, the share c of each cue's frames adds c log P(cue | s) +
    (1 - c) log P(no cue | s) to each state s, from priors["cue_log"] (deadball.fit). With
    `team` ((T, K) float, gapteam.cues) and team_w > 0, each cue's mean over the block's frames
    where it is known (and the ball is not seen, with team_where "gaps") adds its log-likelihood
    per state from priors["team_log"] (gapteam.fit), times the share of frames it was read on.
    With `frame` ((T,) log P(home) / P(away), framemodel.log_odds) and frame_w > 0, the block
    means of log P(home) and log P(away) add to H and A, and their maximum to D, times frame_w
    and the share of frames it was read on (C3). With `deadm` ((T,) log P(x | dead) -
    log P(x | alive), deadmodel.log_ratio) and deadm_w > 0, its block mean over finite frames
    (with a seen ball only, for deadm_where "seen") adds to D, times deadm_w and the share s of frames it was read on, and the speed
    likelihood keeps weight 1 - s: the model reads the speed rank itself (section 16). With
    `tv` ((T,) int8: 1 TV off the live pitch, 0 on it, -1 unknown; solution/cutaway.py) and
    tv_w > 0, the off share f of the block's known frames adds f log P(off | s) +
    (1 - f) log P(on | s) from priors["tv_log"] (row 0 alive, 1 dead), times tv_w and the
    share of the block's frames that are known. With `keeper` ((T, 3) log P(x | H, A, D) up to
    a constant per frame, NaN off the keeper zone; keepermodel.log_lik) and keeper_w > 0, each
    column's block mean over the zone frames adds to its state, times keeper_w and the share of
    the block's frames in the zone. `whistle` ((T, 3), solution/whistlecue.log_lik, NaN where
    unknown) is read the same way with whistle_w.
    """
    n = len(nearest)
    use_holder = holder is not None and params["touch_w"] > 0
    if use_holder and params["nearest_fallback"]:
        nearest = np.where(holder >= 0, -1, nearest)
    n_blocks = -(-n // BLOCK)
    index = np.arange(n) // BLOCK
    size = np.bincount(index, minlength=n_blocks).astype(float)

    def fraction(mask):
        return np.bincount(index, weights=mask.astype(float), minlength=n_blocks) / size
    home, away, missing = fraction(nearest == HOME), fraction(nearest == AWAY), fraction(~ball_seen)
    finite = np.isfinite(speed)
    count = np.bincount(index, weights=finite.astype(float), minlength=n_blocks)
    total = np.bincount(index, weights=np.where(finite, speed, 0.0), minlength=n_blocks)
    with np.errstate(invalid="ignore"):
        block_speed = total / count
    speed_log = np.zeros((n_blocks, 2))
    known = count > 0
    speed_log[known] = priors["speed_log"][:, speed_bins(block_speed[known],
                                                         priors["speed_edges"])].T

    dead_read = None
    if deadm is not None and params["deadm_w"] > 0:
        if params["deadm_where"] not in ("all", "seen"):
            raise ValueError("unknown deadm_where {!r}".format(params["deadm_where"]))
        use = np.isfinite(deadm)
        if params["deadm_where"] == "seen":
            use &= ball_seen
        dead_count = np.bincount(index, weights=use.astype(float), minlength=n_blocks)
        dead_read = dead_count > 0
        dead_total = np.bincount(index, weights=np.where(use, deadm, 0.0), minlength=n_blocks)
        dead_share = dead_count / size
        speed_log = speed_log * (1.0 - dead_share)[:, None]

    agree = priors["agree"]
    same, other, half = math.log(agree), math.log(1.0 - agree), math.log(0.5)
    out = np.empty((n_blocks, 3))
    out[:, H] = home * same + away * other + speed_log[:, 0]
    out[:, A] = away * same + home * other + speed_log[:, 0]
    out[:, D] = (home + away) * half + missing * params["no_ball"] + speed_log[:, 1]
    if use_holder:
        held_home, held_away = fraction(holder == HOME), fraction(holder == AWAY)
        q = priors["holder_agree"]
        same, other, weight = math.log(q), math.log(1.0 - q), params["touch_w"]
        out[:, H] += weight * (held_home * same + held_away * other)
        out[:, A] += weight * (held_away * same + held_home * other)
        out[:, D] += weight * (held_home + held_away) * half
    if cue is not None and params["deadball_w"] > 0:
        for k, table in enumerate(priors["cue_log"]):
            share = fraction(cue[:, k])
            for state, row in ((H, 0), (A, 0), (D, 1)):
                out[:, state] += params["deadball_w"] * (share * table[row, 1] +
                                                         (1.0 - share) * table[row, 0])
    if team is not None and params["team_w"] > 0:
        from solution import gapteam
        where = params["team_where"]
        if where not in ("gaps", "all"):
            raise ValueError("unknown team_where {!r}".format(where))
        for k, (name, table) in enumerate(zip(priors["team_names"], priors["team_log"])):
            value = team[:, k]
            use = np.isfinite(value) & (~ball_seen if where == "gaps" else True)
            count = np.bincount(index, weights=use.astype(float), minlength=n_blocks)
            total = np.bincount(index, weights=np.where(use, value, 0.0), minlength=n_blocks)
            read = count > 0
            if not read.any():
                continue
            log = gapteam.state_log(total[read] / count[read], table, name, params["team_dead"])
            out[read] += params["team_w"] * (count[read] / size[read])[:, None] * log
    if frame is not None and params["frame_w"] > 0:
        where = params["frame_where"]
        if where not in ("gaps", "all", "seen"):
            raise ValueError("unknown frame_where {!r}".format(where))
        use = np.isfinite(frame)
        if where != "all":
            use &= ~ball_seen if where == "gaps" else ball_seen
        count = np.bincount(index, weights=use.astype(float), minlength=n_blocks)
        read = count > 0
        if read.any():
            # log P(home) and log P(away) per frame from the log-odds; dead ball takes the
            # better of the two, so the model chooses between the teams only (as D16's "max").
            odds = np.where(use, frame, 0.0)
            log_home = np.bincount(index, weights=np.where(use, -np.logaddexp(0.0, -odds), 0.0),
                                   minlength=n_blocks)[read] / count[read]
            log_away = np.bincount(index, weights=np.where(use, -np.logaddexp(0.0, odds), 0.0),
                                   minlength=n_blocks)[read] / count[read]
            share = params["frame_w"] * count[read] / size[read]
            out[read, H] += share * log_home
            out[read, A] += share * log_away
            out[read, D] += share * np.maximum(log_home, log_away)
    if dead_read is not None and dead_read.any():
        out[dead_read, D] += (params["deadm_w"] * dead_share[dead_read] *
                              dead_total[dead_read] / dead_count[dead_read])
    if tv is not None and params.get("tv_w", 0.0) > 0:
        known = tv >= 0
        count = np.bincount(index, weights=known.astype(float), minlength=n_blocks)
        read = count > 0
        if read.any():
            off = np.bincount(index, weights=(tv == 1).astype(float), minlength=n_blocks)
            f = off[read] / count[read]
            share = params["tv_w"] * count[read] / size[read]
            table = priors["tv_log"]
            for state, row in ((H, 0), (A, 0), (D, 1)):
                out[read, state] += share * (f * table[row, 1] + (1.0 - f) * table[row, 0])
    for values, weight in ((keeper, params.get("keeper_w", 0.0)),
                           (whistle, params.get("whistle_w", 0.0))):
        if values is None or weight <= 0:
            continue
        use = np.isfinite(values).all(axis=1)
        count = np.bincount(index, weights=use.astype(float), minlength=n_blocks)
        read = count > 0
        if read.any():
            share = weight * count[read] / size[read]
            for state in (H, A, D):
                total = np.bincount(index, weights=np.where(use, values[:, state], 0.0),
                                    minlength=n_blocks)
                out[read, state] += share * total[read] / count[read]
    return params["w"] * out


def transition_scores(params):
    """trans[j, k]: log score of a segment of state k following one of state j."""
    trans = np.zeros((3, 3))
    trans[H, A] = trans[A, H] = -params["switch"]
    trans[H, D] = trans[A, D] = -params["dead"]
    trans[D, H] = trans[D, A] = -params["restart"]
    return trans


def switch_bonus(frames, teams, n_frames, params, fps):
    """(n_blocks, 2) bonus for an H <-> A switch into H (column 0) or A (column 1) at a block,
    from one half's events (frames relative to the half's start, teams H or A; others ignored):
    bonus_w at every block within bonus_spread_s of an event moved by bonus_shift_s."""
    n_blocks = -(-n_frames // BLOCK)
    out = np.zeros((n_blocks, 2))
    shift = int(round(params["bonus_shift_s"] * fps))
    reach = int(round(params["bonus_spread_s"] * fps / BLOCK))
    for f, k in zip(frames, teams):
        if k not in (H, A):
            continue
        b = (int(f) + shift) // BLOCK
        lo, hi = max(0, b - reach), min(n_blocks, b + reach + 1)
        if lo < hi:
            out[lo:hi, k] = params["bonus_w"]
    return out


def viterbi(emissions, log_duration, trans, bonus=None):
    """Exact semi-Markov Viterbi. Returns the best state per block.

    emissions: (N, K) per-block log-likelihoods; log_duration: (K, Dmax), entry d-1 is log P(d),
    -inf where d is not allowed; trans: (K, K). `bonus` (N, 2), optional, is added to an A -> H
    (column 0) or H -> A (column 1) switch whose new segment starts at that block. Ties go to
    the shorter segment, then the lower state, so the result is deterministic.
    """
    n, k_states = emissions.shape
    d_max = log_duration.shape[1]
    cumulative = np.vstack((np.zeros(k_states), np.cumsum(emissions, axis=0)))
    # entry[t, k]: best score of everything before block t, given a segment of k starts at t.
    entry = np.full((n + 1, k_states), -np.inf)
    entry[0] = 0.0
    came_from = np.zeros((n + 1, k_states), dtype=np.int64)
    length = np.zeros((n + 1, k_states), dtype=np.int64)
    best = np.full((n + 1, k_states), -np.inf)
    states = np.arange(k_states)
    for t in range(1, n + 1):
        reach = min(d_max, t)
        starts = t - np.arange(1, reach + 1)
        score = entry[starts] - cumulative[starts] + log_duration[:, :reach].T
        pick = np.argmax(score, axis=0)
        best[t] = score[pick, states] + cumulative[t]
        length[t] = pick + 1
        if t < n:
            onward = best[t][:, None] + trans
            if bonus is not None:
                onward[A, H] += bonus[t, 0]
                onward[H, A] += bonus[t, 1]
            came_from[t] = np.argmax(onward, axis=0)
            entry[t] = onward[came_from[t], states]
    if not np.isfinite(best[n]).any():
        raise ValueError("no segmentation satisfies the minimum durations")
    out = np.empty(n, dtype=np.int8)
    state, t = int(np.argmax(best[n])), n
    while t > 0:
        start = t - int(length[t, state])
        out[start:t] = state
        state, t = int(came_from[start, state]), start
    return out


def _logsumexp(values, axis=0):
    """log(sum(exp(values))) along `axis`; -inf where every entry is -inf."""
    top = np.max(values, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(values - top), axis=axis, keepdims=True)) + top
    return np.squeeze(out, axis=axis)


def forward_sums(emissions, log_duration, trans, bonus=None):
    """The semi-Markov forward pass: viterbi() with every max replaced by a log-sum.

    Same model and arguments as viterbi(). Returns (entry, total, cumulative): entry[t, k] is the
    log total score of every segmentation of blocks [0, t) followed by a segment of k starting
    at t; total[t, k] the same for segmentations of [0, t) whose last segment is k;
    cumulative[t] the summed emissions of blocks [0, t). Read by backward_sample (#6, MBR).
    """
    n, k_states = emissions.shape
    d_max = log_duration.shape[1]
    cumulative = np.vstack((np.zeros(k_states), np.cumsum(emissions, axis=0)))
    entry = np.full((n + 1, k_states), -np.inf)
    entry[0] = 0.0
    total = np.full((n + 1, k_states), -np.inf)
    for t in range(1, n + 1):
        reach = min(d_max, t)
        starts = t - np.arange(1, reach + 1)
        score = entry[starts] - cumulative[starts] + log_duration[:, :reach].T
        total[t] = _logsumexp(score, axis=0) + cumulative[t]
        if t < n:
            onward = total[t][:, None] + trans
            if bonus is not None:
                onward[A, H] += bonus[t, 0]
                onward[H, A] += bonus[t, 1]
            entry[t] = _logsumexp(onward, axis=0)
    if not np.isfinite(total[n]).any():
        raise ValueError("no segmentation satisfies the minimum durations")
    return entry, total, cumulative


def _draw(log_weights, rng):
    """Index drawn with probability proportional to exp(log_weights), by inverse CDF."""
    weights = np.exp(log_weights - np.max(log_weights))
    cdf = np.cumsum(weights)
    return int(min(np.searchsorted(cdf, rng.random_sample() * cdf[-1], side="right"),
                   len(cdf) - 1))


def backward_sample(sums, log_duration, trans, bonus, rng):
    """One state sequence per block drawn from the posterior of forward_sums' model.

    Forward filtering, backward sampling: the last segment's state, then its length, then the
    state before it, and so on back to block 0. `rng` is a numpy RandomState.
    """
    entry, total, cumulative = sums
    n = total.shape[0] - 1
    d_max = log_duration.shape[1]
    out = np.empty(n, dtype=np.int8)
    state, t = _draw(total[n], rng), n
    while t > 0:
        reach = min(d_max, t)
        starts = t - np.arange(1, reach + 1)
        start = int(starts[_draw(entry[starts, state] - cumulative[starts, state] +
                                 log_duration[state, :reach], rng)])
        out[start:t] = state
        if start > 0:
            onward = total[start] + trans[:, state]
            if bonus is not None:
                if state == H:
                    onward[A] += bonus[start, 0]
                elif state == A:
                    onward[H] += bonus[start, 1]
            state = _draw(onward, rng)
        t = start
    return out


def half_model(nearest, ball_seen, speed, priors, params, fps, holder=None, cue=None,
               team=None, events=None, frame=None, deadm=None, tv=None, keeper=None,
               whistle=None):
    """(emissions, log_duration, trans, bonus) for one half: what viterbi() decodes."""
    block_s = BLOCK / float(fps)
    max_blocks = int(round(MAX_SECONDS / block_s))
    poss, dead = priors["durations"]["possession"], priors["durations"]["dead"]
    dead_min_s = dead["min_s"]
    if params.get("dead_min_s") is not None:
        dead_min_s = max(dead_min_s, float(params["dead_min_s"]))
    log_duration = np.vstack([
        duration_log_prior(poss["mu"], poss["sigma"], poss["min_s"], block_s, max_blocks),
        duration_log_prior(poss["mu"], poss["sigma"], poss["min_s"], block_s, max_blocks),
        duration_log_prior(dead["mu"], dead["sigma"], dead_min_s, block_s, max_blocks),
    ])
    emissions = block_emissions(nearest, ball_seen, speed, priors, params, holder, cue, team,
                                frame, deadm, tv, keeper, whistle)
    bonus = None
    if events is not None and params["bonus_w"] > 0:
        bonus = switch_bonus(events[0], events[1], len(nearest), params, fps)
    return emissions, log_duration, transition_scores(params), bonus


def decode_half(nearest, ball_seen, speed, priors, params, fps, holder=None, cue=None,
                team=None, events=None, frame=None, deadm=None, tv=None, keeper=None,
                whistle=None):
    """Per-frame states 0/1/2 for one half's frames. `events` (frames relative to the half's
    first frame, teams), with bonus_w > 0, adds switch_bonus to the H <-> A switches."""
    blocks = viterbi(*half_model(nearest, ball_seen, speed, priors, params, fps, holder, cue,
                                 team, events, frame, deadm, tv, keeper, whistle))
    return np.repeat(blocks, BLOCK)[:len(nearest)]


def sample_half(model, n_frames, n_samples, rng, temperature=1.0):
    """(n_samples, n_frames) states drawn from the posterior of half_model's `model`.

    `temperature` divides every score (emissions, durations, transitions, bonus): it leaves the
    Viterbi path unchanged and only widens (> 1) or narrows (< 1) the posterior around it.
    """
    emissions, log_duration, trans, bonus = model
    scale = 1.0 / float(temperature)
    bonus = None if bonus is None else bonus * scale
    log_duration, trans = log_duration * scale, trans * scale
    sums = forward_sums(emissions * scale, log_duration, trans, bonus)
    out = np.empty((n_samples, n_frames), dtype=np.int8)
    for i in range(n_samples):
        out[i] = np.repeat(backward_sample(sums, log_duration, trans, bonus, rng),
                           BLOCK)[:n_frames]
    return out


# -- fitting on the IDSSE dev split, leave one match out ---------------------------------------

@lru_cache(maxsize=None)
def idsse_priors(exclude="", speed_lag=2, speed_scale="ms", speed_lead_s=0.0, labels=None):
    """Priors fitted on the clean IDSSE dev matches, minus `exclude` (a Match name).

    Leave-one-match-out on the proxy league; on BAR-ATH nothing is excluded. The held-out
    matches are never part of the fit (D3). `labels` is the label variant to fit on
    (idsse.LABELS when None, D20), part of the cache key.
    """
    from solution.methods import nearest_player_codes
    from solution.sources import idsse
    if labels is None:
        return idsse_priors(exclude, speed_lag, speed_scale, speed_lead_s, idsse.LABELS)
    ids = [m for m in idsse.DEV_IDS if "idsse_" + m != exclude]
    assert not set(ids) & set(idsse.HOLDOUT_IDS)
    return fit_priors([idsse.load(m, labels=labels) for m in ids], nearest_player_codes,
                      lambda m: speed_feature(m, speed_lag, speed_scale, speed_lead_s),
                      RANK_EDGES if speed_scale == "rank" else SPEED_EDGES)
