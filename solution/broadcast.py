"""Broadcast -> tactical frame alignment (backlog E13, Step 0). Matches in, arrays out; no files.

The broadcast recording has its own ATD on its own timeline, with cuts, replays and zooms, and no
mapping to the tactical video is provided (DATA.md). Both ATDs report positions in the same pitch
coordinates, so the mapping can be found from the tracking alone, by algorithm:

  1. Split the broadcast into stretches of valid homography (camera on the pitch), at least
     MIN_STRETCH_S long. Only these carry positions.
  2. Vote for each stretch's offset d (tactical frame = broadcast frame + d) with the ball: every
     sampled broadcast frame that sees the ball votes for every tactical frame whose ball lies
     within VOTE_R of it. The N_CANDIDATES best vote bins are candidates.
  3. Refine each candidate to the frame with a player cost: over sampled broadcast frames, the
     mean distance from each broadcast player (and the ball) to the nearest tactical player (or
     ball) at frame b + d, each capped at CAP_M. The camera shows part of the pitch, so the
     broadcast players are a subset of the tactical ones and the distance runs one way only. No
     identities needed.
  4. A stretch is confident when its best cost is under COST_MAX and under MARGIN times its best
     rival: the best offset more than RIVAL_S away, or a few random offsets.
  5. A live broadcast shows the match continuously except while it airs a replay, so live
     stretches share one offset (or one that steps up when the feed skips time), and a replay
     shows an earlier moment: a smaller d. The live chain is the heaviest (by length) sequence of
     confident stretches whose offsets never fall by more than CHAIN_TOL. Confident stretches
     below it are replay candidates.
  6. Between two chain stretches whose offsets agree within CHAIN_TOL, every broadcast frame,
     on the pitch or not, maps to live time b + d. That gives, per tactical frame, what the TV
     showed at that moment: the live pitch, a cut-away (no homography), a replay, or something
     not aligned.

The two feeds could disagree on the pitch orientation; choose_flip() tests the four mirrorings on
the longest stretches and keeps the one that aligns best.

Every parameter below was set once, before the first run on BAR-ATH, and is not tuned on it.
"""

import warnings

import numpy as np
from scipy.spatial import cKDTree

# Params
MIN_STRETCH_S = 2.0      # valid-homography stretches shorter than this are not aligned
SAMPLE_STEP = 5          # frames between broadcast frames that vote
VOTE_R = 1.5             # metres: a tactical ball this close to the broadcast ball gets a vote
BIN = 25                 # frames per vote bin
N_CANDIDATES = 5         # vote bins refined per stretch
REFINE = 25              # frames either side of a candidate bin's centre
N_COST_FRAMES = 24       # broadcast frames the player cost is averaged over
MIN_POINTS = 3           # broadcast players a frame needs to enter the cost
CAP_M = 5.0              # metres: per-point distance cap in the cost
COST_MAX = 2.0           # metres: a confident alignment costs less than this
MARGIN = 0.6             # ... and less than MARGIN x its best rival
RIVAL_S = 2.0            # a rival offset is at least this far from the best one
N_PROBES = 4             # random offsets added as rivals
CHAIN_TOL = 3            # frames: offsets this close count as the same live offset
DRIFT_MIN_S = 20.0       # stretches at least this long get the drift check
N_DRIFT_FRAMES = 12      # frames per third in the drift check
TEAM_R = 1.5             # metres: nearest-player pairing radius for the team check
FLIPS = ("none", "x", "y", "xy")
N_FLIP_STRETCHES = 10
N_NEIGHBOURS = 3         # pass 2 tries the offsets of this many confident stretches each side

# Ball fill (fuse_ball), fixed before its first run on BAR-ATH
FILL_MAX_SPEED = 40.0    # m/s: a filled ball must be reachable from the tactical ball each side
FILL_SLACK_M = 3.0       # metres added to that reach (position error of both feeds)
FILL_MIN_RUN = 5         # frames: filled runs shorter than this are dropped (no ball flicker)

NOT_COVERED, LIVE, CUT, REPLAY, OTHER = 0, 1, 2, 3, 4
STATE_NAMES = {NOT_COVERED: "not covered", LIVE: "live pitch", CUT: "cut away (no homography)",
               REPLAY: "replay on air", OTHER: "on pitch, not aligned"}


def runs(mask, min_len=1):
    """Half-open (start, stop) spans of the True runs of `mask`, at least `min_len` long."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = stops - starts >= min_len
    return list(zip(starts[keep].tolist(), stops[keep].tolist()))


def flipped(match, flip):
    """A copy of `match` mirrored: 'x' along the length, 'y' across, 'xy' both, 'none' as is."""
    out = match.copy()
    length, width = match.pitch
    if "x" in flip:
        out.player_x, out.ball_x = length - out.player_x, length - out.ball_x
    if "y" in flip:
        out.player_y, out.ball_y = width - out.player_y, width - out.ball_y
    return out


class BallIndex:
    """The tactical ball positions, for the vote."""

    def __init__(self, tac):
        self.frames = np.flatnonzero(tac.ball_seen)
        self.tree = cKDTree(np.column_stack([tac.ball_x[self.frames],
                                             tac.ball_y[self.frames]]).astype(np.float64))


def vote(index, bro, start, stop):
    """([(centre offset, votes)] best first, total votes) for one broadcast stretch."""
    frames = np.arange(start, stop, SAMPLE_STEP)
    frames = frames[bro.ball_seen[frames]]
    if not len(frames) or not len(index.frames):
        return [], 0
    points = np.column_stack([bro.ball_x[frames], bro.ball_y[frames]]).astype(np.float64)
    hits = index.tree.query_ball_point(points, VOTE_R)
    offsets = np.concatenate([index.frames[np.asarray(h, dtype=np.int64)] - b
                              for h, b in zip(hits, frames)])
    if not len(offsets):
        return [], 0
    bins, counts = np.unique(np.floor_divide(offsets, BIN), return_counts=True)
    order = np.lexsort((bins, -counts))
    chosen = []
    for k in order:
        if len(chosen) == N_CANDIDATES:
            break
        if all(abs(bins[k] - c) > 1 for c, _ in chosen):
            chosen.append((int(bins[k]), int(counts[k])))
    return [(c * BIN + BIN // 2, n) for c, n in chosen], int(len(offsets))


def cost_frames(bro, start, stop, n):
    """Up to `n` broadcast frames in [start, stop), evenly spread, each with MIN_POINTS players."""
    seen = np.isfinite(bro.player_x[start:stop]) & np.isfinite(bro.player_y[start:stop])
    eligible = start + np.flatnonzero(seen.sum(axis=1) >= MIN_POINTS)
    if len(eligible) <= n:
        return eligible
    return np.unique(eligible[np.linspace(0, len(eligible) - 1, n).round().astype(int)])


def cost_curve(tac, bro, frames, offsets):
    """For each offset: median over `frames` of the capped one-way distance (metres). NaN if no
    frame lands inside the tactical recording."""
    offsets = np.asarray(offsets, dtype=np.int64)
    table = np.full((len(frames), len(offsets)), np.nan)
    for i, b in enumerate(frames):
        ok = np.isfinite(bro.player_x[b]) & np.isfinite(bro.player_y[b])
        px = bro.player_x[b][ok].astype(np.float64)
        py = bro.player_y[b][ok].astype(np.float64)
        t = b + offsets
        inside = (t >= 0) & (t < tac.n_frames)
        if len(px) < MIN_POINTS or not inside.any():
            continue
        ti = t[inside]
        qx = tac.player_x[ti].astype(np.float64)                      # (D, S)
        qy = tac.player_y[ti].astype(np.float64)
        dist = np.hypot(px[None, :, None] - qx[:, None, :], py[None, :, None] - qy[:, None, :])
        dist = np.where(np.isfinite(dist), dist, np.inf).min(axis=2)  # (D, k)
        total = np.minimum(dist, CAP_M).sum(axis=1)
        count = np.full(len(ti), float(len(px)))
        if bro.ball_seen[b]:
            both = tac.ball_seen[ti]
            ball = np.hypot(float(bro.ball_x[b]) - tac.ball_x[ti].astype(np.float64),
                            float(bro.ball_y[b]) - tac.ball_y[ti].astype(np.float64))
            total += np.where(both, np.minimum(np.nan_to_num(ball, nan=CAP_M), CAP_M), 0.0)
            count += both
        table[i, inside] = total / count
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(table, axis=0)


def _best(curve, offsets):
    if np.all(np.isnan(curve)):
        return None
    k = int(np.nanargmin(curve))
    return float(curve[k]), int(offsets[k])


def align_stretch(tac, index, bro, start, stop, extra=(), seed=0):
    """Best offset for one broadcast stretch, with its cost, its rival's cost and confidence."""
    fps = tac.fps
    candidates, n_votes = vote(index, bro, start, stop)
    centres = [c for c, _ in candidates] + [int(d) for d in extra]
    out = {"start": start, "stop": stop, "votes": n_votes, "d": None, "cost": np.nan,
           "rival": np.nan, "probe": np.nan, "confident": False, "drift": None}
    frames = cost_frames(bro, start, stop, N_COST_FRAMES)
    if len(frames) < 3 or not centres:
        return out
    found = []
    for c in centres:
        offsets = np.arange(c - REFINE, c + REFINE + 1)
        best = _best(cost_curve(tac, bro, frames, offsets), offsets)
        if best is not None:
            found.append(best)
    if not found:
        return out
    cost, d = min(found)
    rng = np.random.RandomState(seed)
    lo, hi = -start, tac.n_frames - stop
    probes = rng.randint(lo, hi, N_PROBES) if hi > lo else np.array([], dtype=np.int64)
    probes = probes[np.abs(probes - d) > RIVAL_S * fps]
    probe_costs = cost_curve(tac, bro, frames, probes) if len(probes) else np.array([])
    rivals = [c for c, o in found if abs(o - d) > RIVAL_S * fps]
    rivals += [float(c) for c in probe_costs if np.isfinite(c)]
    rival = min(rivals) if rivals else np.nan
    out.update(d=d, cost=cost, rival=rival,
               probe=float(np.nanmedian(probe_costs)) if np.isfinite(probe_costs).any()
               else np.nan,
               confident=bool(cost < COST_MAX and np.isfinite(rival) and cost < MARGIN * rival))
    if out["confident"] and stop - start >= DRIFT_MIN_S * fps:
        third = (stop - start) // 3
        offsets = np.arange(d - REFINE, d + REFINE + 1)
        ends = []
        for a, b in ((start, start + third), (stop - third, stop)):
            best = _best(cost_curve(tac, bro, cost_frames(bro, a, b, N_DRIFT_FRAMES), offsets),
                         offsets)
            ends.append(None if best is None else best[1])
        if None not in ends:
            out["drift"] = ends[1] - ends[0]
    return out


def align_stretches(tac, bro, valid, spans=None):
    """align_stretch over every valid stretch; unconfident ones retry with the offsets of up to
    N_NEIGHBOURS confident stretches on each side as candidates (pass 2), for stretches with too
    little ball to vote. More than one each side, because the nearest may be a replay."""
    if spans is None:
        spans = runs(valid, int(round(MIN_STRETCH_S * tac.fps)))
    index = BallIndex(tac)
    results = [align_stretch(tac, index, bro, a, b, seed=k) for k, (a, b) in enumerate(spans)]
    for r in results:
        r["pass"] = 1
    confident = [k for k, r in enumerate(results) if r["confident"]]
    for k, r in enumerate(results):
        if r["confident"]:
            continue
        near = ([j for j in confident if j < k][-N_NEIGHBOURS:]
                + [j for j in confident if j > k][:N_NEIGHBOURS])
        extra = sorted({results[j]["d"] for j in near})
        if extra:
            again = align_stretch(tac, index, bro, r["start"], r["stop"], extra=extra, seed=k)
            if again["confident"]:
                again["pass"] = 2
                results[k] = again
    return results


def live_chain(results, tol=CHAIN_TOL):
    """Indices of the live chain: the heaviest sequence of confident stretches (in broadcast
    order) whose offsets never fall by more than `tol`."""
    idx = [k for k, r in enumerate(results) if r["confident"]]
    if not idx:
        return []
    d = [results[k]["d"] for k in idx]
    w = [results[k]["stop"] - results[k]["start"] for k in idx]
    best, prev = list(w), [-1] * len(idx)
    for j in range(len(idx)):
        for i in range(j):
            if d[j] >= d[i] - tol and best[i] + w[j] > best[j]:
                best[j], prev[j] = best[i] + w[j], i
    j = int(np.argmax(best))
    chain = []
    while j >= 0:
        chain.append(idx[j])
        j = prev[j]
    return chain[::-1]


def classify(results, chain, tol=CHAIN_TOL):
    """Per stretch: 'live' (in the chain), 'replay' (confident, offset below the live one),
    'ahead' (confident, above it), 'off-chain' (confident, at it but left out) or 'unaligned'."""
    in_chain = set(chain)
    kinds = []
    for k, r in enumerate(results):
        if k in in_chain:
            kinds.append("live")
        elif not r["confident"] or not chain:
            kinds.append("unaligned")
        else:
            before = [j for j in chain if results[j]["start"] < r["start"]]
            ref = results[before[-1] if before else chain[0]]["d"]
            kinds.append("replay" if r["d"] < ref - tol else "ahead" if r["d"] > ref + tol
                         else "off-chain")
    return kinds


def live_map(results, chain, n_bro, tol=CHAIN_TOL):
    """Per broadcast frame, the tactical frame of the live moment it airs during; -1 if unknown.

    Chain stretches map by their own offset. A gap between two chain stretches whose offsets
    agree within `tol` is filled (each half from its nearer side); a gap where they disagree is
    where the feed skipped time, at an unknown frame, so it stays unmapped."""
    live_t = np.full(n_bro, -1, dtype=np.int64)
    ordered = sorted(chain, key=lambda k: results[k]["start"])
    for n, k in enumerate(ordered):
        r = results[k]
        live_t[r["start"]:r["stop"]] = np.arange(r["start"], r["stop"]) + r["d"]
        if n + 1 < len(ordered):
            nxt = results[ordered[n + 1]]
            if abs(nxt["d"] - r["d"]) <= tol and nxt["start"] > r["stop"]:
                mid = (r["stop"] + nxt["start"]) // 2
                live_t[r["stop"]:mid] = np.arange(r["stop"], mid) + r["d"]
                live_t[mid:nxt["start"]] = np.arange(mid, nxt["start"]) + nxt["d"]
    return live_t


def align(tac, bro, valid):
    """The whole alignment. `bro` is already in the chosen orientation; `valid` is the broadcast
    homography's valid flag. Returns a dict of per-stretch results and per-frame maps:

      live_t[b]        tactical frame of the live moment broadcast frame b airs during, or -1
      bstate[b]        what the broadcast shows then (STATE_NAMES)
      state[t]         the same per tactical frame
      bro_frame[t]     the broadcast frame airing at tactical frame t's moment, or -1
      replay_frame[t]  a broadcast replay frame showing tactical frame t, or -1
    """
    results = align_stretches(tac, bro, valid)
    chain = live_chain(results)
    kinds = classify(results, chain)
    n_bro, n_tac = bro.n_frames, tac.n_frames
    live_t = live_map(results, chain, n_bro)
    mapped = live_t >= 0
    bstate = np.full(n_bro, NOT_COVERED, dtype=np.int8)
    bstate[mapped & ~valid] = CUT
    bstate[mapped & valid] = OTHER
    replay_frame = np.full(n_tac, -1, dtype=np.int64)
    for r, kind in zip(results, kinds):
        span = slice(r["start"], r["stop"])
        if kind == "live":
            bstate[span] = LIVE
        elif kind == "replay":
            bstate[span] = np.where(mapped[span], REPLAY, NOT_COVERED)
            b = np.arange(r["start"], r["stop"])
            t = b + r["d"]
            ok = (t >= 0) & (t < n_tac)
            replay_frame[t[ok]] = b[ok]
    state = np.full(n_tac, NOT_COVERED, dtype=np.int8)
    bro_frame = np.full(n_tac, -1, dtype=np.int64)
    ok = mapped & (live_t < n_tac)
    state[live_t[ok]] = bstate[ok]
    bro_frame[live_t[ok]] = np.flatnonzero(ok)
    return {"results": results, "chain": chain, "kinds": kinds, "live_t": live_t,
            "bstate": bstate, "state": state, "bro_frame": bro_frame,
            "replay_frame": replay_frame}


def choose_flip(tac, bro, valid, n=N_FLIP_STRETCHES):
    """(flip, rows): the mirroring under which the `n` longest stretches align best (most
    confident, then lowest median cost; FLIPS order breaks ties)."""
    spans = runs(valid, int(round(MIN_STRETCH_S * tac.fps)))
    longest = sorted(spans, key=lambda s: (s[0] - s[1], s[0]))[:n]
    index = BallIndex(tac)
    rows = []
    for flip in FLIPS:
        mirrored = flipped(bro, flip)
        found = [align_stretch(tac, index, mirrored, a, b, seed=k)
                 for k, (a, b) in enumerate(longest)]
        costs = [r["cost"] for r in found if r["confident"]]
        rows.append({"flip": flip, "stretches": len(found),
                     "confident": sum(r["confident"] for r in found),
                     "cost": float(np.median(costs)) if costs else np.nan})
    best = rows[0]
    for row in rows[1:]:
        key = (row["confident"], -(row["cost"] if np.isfinite(row["cost"]) else np.inf))
        ref = (best["confident"], -(best["cost"] if np.isfinite(best["cost"]) else np.inf))
        if key > ref:
            best = row
    return best["flip"], rows


def team_agreement(tac, bro, results, chain):
    """(share, pairs): of broadcast players paired with a tactical player within TEAM_R on the
    chain's sampled frames, the share whose team codes agree. Near 0 means the two metadata
    files list the teams in opposite order."""
    same = total = 0
    for k in chain:
        r = results[k]
        for b in cost_frames(bro, r["start"], r["stop"], N_COST_FRAMES):
            t = b + r["d"]
            if not 0 <= t < tac.n_frames:
                continue
            ok = np.flatnonzero(np.isfinite(bro.player_x[b]) & np.isfinite(bro.player_y[b]))
            qx, qy = tac.player_x[t].astype(np.float64), tac.player_y[t].astype(np.float64)
            for s in ok:
                dist = np.hypot(qx - bro.player_x[b, s], qy - bro.player_y[b, s])
                dist = np.where(np.isfinite(dist), dist, np.inf)
                j = int(np.argmin(dist))
                if dist[j] <= TEAM_R:
                    total += 1
                    same += int(tac.team[j] == bro.team[s])
    return (same / total if total else np.nan), total


def fuse_ball(tac, bro, state, bro_frame):
    """(copy of `tac` with the broadcast ball filled into its ball gaps, stats).

    The tactical ball is never replaced: where both feeds see it they disagree by more than 5 m
    on 12% of frames (EXPERIMENTS section 17), and nothing says which one is right. A frame is
    filled only when the TV shows the live pitch at that moment (`state` LIVE), the tactical ball
    is missing and the broadcast sees it; and only if the filled ball is reachable at
    FILL_MAX_SPEED (+ FILL_SLACK_M) from the nearest tactical ball before and after it. Filled
    runs shorter than FILL_MIN_RUN frames are dropped. `state` and `bro_frame` come from align().
    """
    n = tac.n_frames
    t = np.flatnonzero((state == LIVE) & ~tac.ball_seen & (bro_frame >= 0))
    t = t[bro.ball_seen[bro_frame[t]]]
    fx = np.full(n, np.nan)
    fy = np.full(n, np.nan)
    fx[t] = bro.ball_x[bro_frame[t]]
    fy[t] = bro.ball_y[bro_frame[t]]
    ok = np.ones(len(t), dtype=bool)
    seen = np.flatnonzero(tac.ball_seen)
    if len(seen) and len(t):
        pos = np.searchsorted(seen, t)
        for anchor, has in ((seen[np.maximum(pos - 1, 0)], pos > 0),
                            (seen[np.minimum(pos, len(seen) - 1)], pos < len(seen))):
            reach = FILL_MAX_SPEED * np.abs(t - anchor) / tac.fps + FILL_SLACK_M
            dist = np.hypot(fx[t] - tac.ball_x[anchor], fy[t] - tac.ball_y[anchor])
            ok &= ~has | (dist <= reach)
    gated = np.zeros(n, dtype=bool)
    gated[t[ok]] = True
    keep = np.zeros(n, dtype=bool)
    for a, b in runs(gated, FILL_MIN_RUN):
        keep[a:b] = True
    fused = tac.copy()
    fused.ball_x[keep] = fx[keep].astype(fused.ball_x.dtype)
    fused.ball_y[keep] = fy[keep].astype(fused.ball_y.dtype)
    fused.source = (tac.source + "+broadcast-ball").lstrip("+")
    stats = {"candidates": int(len(t)), "speed_rejected": int((~ok).sum()),
             "short_dropped": int((gated & ~keep).sum()), "filled": int(keep.sum())}
    fused.notes["broadcast_fill"] = stats
    return fused, stats


# (shortest s, longest s, share of gap frames filled) -- BAR-ATH, EXPERIMENTS section 17.
# BAR-ATH has no 1-3 s gaps; that bin takes the 3-10 s rate.
FILL_PROFILE = ((0.0, 1.0, 0.573), (1.0, 10.0, 0.322), (10.0, 30.0, 0.078),
                (30.0, np.inf, 0.059))
FILL_STREAM = 11          # simulate._rng stream for the fill; no other stream uses it


def imitate_fill(degraded, clean, seed):
    """The proxy imitation of fuse_ball: a copy of `degraded` with part of each ball gap given
    its `clean` ball back, FILL_PROFILE's share by gap length, as one chunk at the gap's start or
    end (a coin flip). Chunks under FILL_MIN_RUN frames are skipped (EXPERIMENTS section 17)."""
    out = degraded.copy()
    from solution.simulate import _rng
    rng = _rng(clean, seed, FILL_STREAM)
    fps = clean.fps
    for start, stop in clean.halves:
        for a, b in runs(~out.ball_seen[start:stop]):
            a, b = a + start, b + start
            length = b - a
            share = next(r for lo, hi, r in FILL_PROFILE if lo * fps <= length < hi * fps)
            k = int(round(share * length))
            at_start = rng.rand() < 0.5
            if k < FILL_MIN_RUN:
                continue
            span = slice(a, a + k) if at_start else slice(b - k, b)
            out.ball_x[span] = clean.ball_x[span]
            out.ball_y[span] = clean.ball_y[span]
    return out
