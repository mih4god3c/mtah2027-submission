"""Simulator v2: v1's ball, and players lost where the BAR-ATH ATD loses them.

v1 drops players at random. tools/track_loss_check.py (EXPERIMENTS section 31) measured where
BAR-ATH tracks end, and it is not random: a player within 1 m of another is lost about 19 times as
often as one 5 m from everyone, more in a crowd of three or more, more off the pitch or near a
line, and more on its own while the ball is unseen. The neighbour's team does not matter. An
isolated, moving player is lost once in about 15 player-minutes, against v1's once in 2.

v2 keeps v1's ball (the same gaps, the same random stream) and replaces v1's player model:
  - losses: every tracked player has a per-frame loss rate, exp(log rate), whose terms are those
    of the BAR-ATH fit: nearest player, others within 3 m, touchline, goal line, ball unseen, and
    nearest under 2 m while the ball is unseen. They are read on the clean positions and on the
    degraded ball; for distance and crowding only neighbours nearer the camera count
    (occlusion hides the player behind; the camera is put on the y = 0 side). Speed is left out: BAR-ATH's "slow players are lost" is the tracker slowing a
    track down before it drops it (the next point), not slowness causing losses;
  - slow-down before the drop: 80% of drops are preceded by the track braking over a few frames
    along the way the player was going, while the player moves on, then crawling (1 mm to 2 cm a
    frame) or, for a quarter of them, walking (2 to 8 cm). BAR-ATH never freezes a track exactly
    before dropping it, but 64% of its ends follow at least one nearly-still frame (18% of all
    player-frames), and it almost never drops a track moving at 3 m/s or more;
  - a lost track comes back after a lognormal pause (v1's), into the same slot, and cannot be
    lost again for MIN_ON frames: BAR-ATH has no track end within 5 frames of a track start;
  - episodes that drop several players at once stay (they make the low object counts, p5 14),
    but as a group: one player picked by loss rate and its nearest neighbours, not players
    scattered at random (those would mostly be isolated, BAR-ATH's safest players);
  - ghosts stay as in v1 but jitter: v1's stand exactly still, and BAR-ATH has almost no frames
    of exact stillness (1.3% of player-frames, against 5.9% in v1).

The loss terms and the crawl share in V2 are calibrated (tools/track_loss_calibrate.py) so that
the same fit run on v2's output reproduces the BAR-ATH fit (targets: simulate.BAR_ATH
["track_loss"]). They are not the BAR-ATH ratios themselves: the fit reads crowding on tracked
players, the slow-down moves a track away from its neighbours, and episodes add losses, so the
inputs are larger than what comes out. The episode settings were chosen on the object-count
targets of `make sim-check`, and the braking share on the fit's speed terms, by hand.

How close it is (dev matches, seed 0): every tuned term within 0.14 of BAR-ATH, the overall rate
0.561 ends per player-minute (BAR-ATH 0.552). The untuned speed terms come out with BAR-ATH's
signs but mostly weaker (recent slow 1.43 against 1.99, recent fast -1.61 against -1.98, earlier
slow -0.56 against -1.03), and an opponent within 2 m at +0.11 against 0.00 (the front-only
occlusion rule). Tracklets under 3 s are too common (p10 1.2 s against 2.6), as in v1.

Known gaps, beyond v1's list in simulate.py: re-acquired tracks keep their identity (no ID
switches); off-durations do not depend on where the player is; ghosts only copy teammates;
positions are not noisy; the loss terms are main effects fitted on one match (1,198 track ends).
"""

import numpy as np

# Calibrated v2 parameters. Times are seconds; lognormal sizes are (median, sigma of the log).
V2 = {
    # per-frame log loss rate, by term; keys as tools/track_loss_check.py prints them
    "loss": {
        "intercept": -12.0217,
        "nearest, any team = 0-1 m": 6.6143,
        "nearest, any team = 1-2 m": 5.1174,
        "nearest, any team = 2-3 m": -0.2366,
        "nearest, any team = 3-5 m": -1.9699,
        "others within 3 m = 2": 0.5902,
        "others within 3 m = 3+": 0.8187,
        "touchline = off pitch": 2.9876,
        "touchline = 0-2 m": 0.8901,
        "touchline = 2-5 m": 0.9202,
        "goal line = off pitch": 2.3564,
        "goal line = 0-2 m": 0.7470,
        "ball = unseen": 2.9445,
        "nearest < 2 m x ball unseen = yes": -2.9431,
    },
    "min_on_frames": 5,
    "off": (0.8, 1.0),               # pause before a lost track comes back
    # slow-down before a drop: the share that brakes; of those, the share that ends at a crawl
    # (slow_step_m a frame) rather than a walk (walk_step_m); lengths in frames
    "brake_share": 0.8,
    "slow_share": 0.75,
    "walk_step_m": (0.02, 0.08),
    "slow_frames": (5.0, 1.0),
    "brake_frames": (6.0, 0.5),      # frames over which the track slows to the creep
    "slow_step_m": (0.001, 0.02),
    # episodes, as v1 but players picked by loss rate
    # rarer and bigger than v1's, within a radius of the centre player: BAR-ATH's object counts
    # (p5 14, p25 21) need large drops, and its low isolated loss rate needs them grouped
    "episode_every": 130.0,
    "episode_length": (29.0, 0.8),
    "episode_drop": 8.0,
    "episode_radius_m": 12.0,        # only players this close to the episode's centre player
    # ghosts: v1's share of frames with a ghost (about 9% per team), in fewer and longer ghosts,
    # because each ghost's end is a track end away from everyone and BAR-ATH has few of those;
    # plus a per-frame jitter (sd per axis, metres)
    "ghost_every": 240.0,
    "ghost_length": (15.0, 0.8),
    "ghost_offset": (0.2, 0.6),
    "ghost_jitter_m": 0.03,
}

CROWD_M = 3.0
CHUNK = 4000


def _lognormal_frames(rng, size, fps):
    median, sigma = size
    return max(1, int(round(median * fps * np.exp(sigma * rng.randn()))))


def loss_log_rate(match, ball_seen, coef):
    """(T, S) per-frame log loss rate of each slot from its clean position; NaN where untracked.

    The terms are those of tools/track_loss_check.py --fit: distances to the other points on the
    same frame, the pitch lines, and whether the ball is seen. Only neighbours in front (nearer
    the camera) count for distance and crowding: a fit on the tracked players then measures about
    what is put in, where counting both players of a pair loses the second one while the tool
    already sees it alone, and no multiplier reaches BAR-ATH's 19 under 1 m.
    """
    length, width = match.pitch
    slots = match.player_x.shape[1]
    not_self = ~np.eye(slots, dtype=bool)
    out = np.full(match.player_x.shape, np.nan, dtype=np.float64)
    c = coef
    near_terms = (c["nearest, any team = 0-1 m"], c["nearest, any team = 1-2 m"],
                  c["nearest, any team = 2-3 m"], c["nearest, any team = 3-5 m"], 0.0)
    near_edges = np.array([1.0, 2.0, 3.0, 5.0])
    for begin in range(0, match.n_frames, CHUNK):
        t = slice(begin, min(begin + CHUNK, match.n_frames))
        px = match.player_x[t].astype(np.float64)
        py = match.player_y[t].astype(np.float64)
        here = np.isfinite(px) & np.isfinite(py)
        dist = np.hypot(px[:, :, None] - px[:, None, :], py[:, :, None] - py[:, None, :])
        dist = np.where(np.isnan(dist), np.inf, dist)
        # Occlusion hides the player behind: only neighbours nearer the camera (on the y = 0
        # side) count, so of two overlapping players the one in front stays tracked.
        front = py[:, None, :] < py[:, :, None]
        dist = np.where(front & not_self[None], dist, np.inf)
        near = dist.min(axis=2)
        crowd = np.sum(dist < CROWD_M, axis=2)
        near_bin = np.searchsorted(near_edges, near, side="right")     # inf -> ">= 5 m"
        rate = c["intercept"] + np.take(near_terms, near_bin)
        rate += np.where(crowd == 2, c["others within 3 m = 2"], 0.0)
        rate += np.where(crowd >= 3, c["others within 3 m = 3+"], 0.0)
        touch = np.minimum(py, width - py)
        rate += np.where(touch < 0, c["touchline = off pitch"],
                         np.where(touch < 2, c["touchline = 0-2 m"],
                                  np.where(touch < 5, c["touchline = 2-5 m"], 0.0)))
        goal = np.minimum(px, length - px)
        rate += np.where(goal < 0, c["goal line = off pitch"],
                         np.where(goal < 2, c["goal line = 0-2 m"], 0.0))
        unseen = ~ball_seen[t][:, None]
        rate += np.where(unseen, c["ball = unseen"], 0.0)
        rate += np.where(unseen & (near < 2.0), c["nearest < 2 m x ball unseen = yes"], 0.0)
        out[t] = np.where(here, rate, np.nan)
    return out


def _losses(log_rate, start, stop, rng, fps, params):
    """On mask (n, S) for one half from per-frame loss draws, pauses and the minimum lifetime."""
    n, slots = stop - start, log_rate.shape[1]
    on = np.ones((n, slots), dtype=bool)
    min_on = params["min_on_frames"]
    for s in range(slots):
        p = -np.expm1(-np.exp(np.nan_to_num(log_rate[start:stop, s], nan=-np.inf)))
        draws = np.flatnonzero(rng.rand(n) < p)
        # A track ends on frame t only if it was tracked min_on frames before (t - min_on), as
        # BAR-ATH has no end within 5 frames of a start.
        ready = min_on
        for t in draws:
            if t < ready or t + 1 >= n:
                continue
            back = t + 1 + _lognormal_frames(rng, params["off"], fps)
            on[t + 1:back, s] = False
            ready = back + min_on
    return on


def _episodes(on, log_rate, x, y, present, start, stop, rng, fps, params):
    """Drop a group at once: one player picked by loss rate, and its neighbours within a radius.

    A group, not players scattered over the pitch: spread-out drops are mostly of isolated
    players, whose BAR-ATH loss rate is the lowest of all (EXPERIMENTS section 31).
    """
    n = stop - start
    position = int(rng.exponential(params["episode_every"] * fps))
    while position < n:
        # Only tracks at least min_on frames old: BAR-ATH has no end that soon after a start.
        lo = max(0, position - params["min_on_frames"])
        here = np.flatnonzero(present[start + position] & on[lo:position + 1].all(axis=0))
        if len(here):
            weight = np.exp(log_rate[start + position, here] - np.nanmax(
                log_rate[start + position, here]))
            count = min(len(here), int(rng.geometric(1.0 / params["episode_drop"])))
            centre = rng.choice(here, p=weight / weight.sum())
            t = start + position
            gap = np.hypot(x[t, here] - x[t, centre], y[t, here] - y[t, centre])
            order = np.argsort(gap, kind="stable")[:count]
            drop = here[order[gap[order] <= params["episode_radius_m"]]]
            length = _lognormal_frames(rng, params["episode_length"], fps)
            on[position + 1:position + 1 + length, drop] = False
        position += 1 + int(rng.exponential(params["episode_every"] * fps))


def _slow_down(x, y, on, rng, fps, params):
    """Before a drop, the track brakes, then creeps or walks along the player's last heading.

    Braking: over brake_frames the per-frame step shrinks geometrically from the player's own to
    the creep size, so the track is slow on its last frames, not only on the very last (BAR-ATH
    almost never drops a track moving at 3 m/s or more over its last 0.2 s). Creeping: then
    slow_frames of steps of slow_step_m. Edits x and y (the half's slice, in place) on frames
    the track is still shown.
    """
    ends = np.argwhere(on[:-1] & ~on[1:])
    for t, s in ends:
        if rng.rand() >= params["brake_share"]:
            continue
        # The track ends at a crawl for slow_share of drops, else at a walk.
        low, high = params["slow_step_m"] if rng.rand() < params["slow_share"] else \
            params["walk_step_m"]
        run_start = t
        while run_start > 0 and on[run_start - 1, s]:
            run_start -= 1
        creep = _lognormal_frames(rng, (params["slow_frames"][0] / fps,
                                        params["slow_frames"][1]), fps)
        brake = _lognormal_frames(rng, (params["brake_frames"][0] / fps,
                                        params["brake_frames"][1]), fps)
        length = min(creep + brake, t - run_start)
        if length < 1:
            continue
        a = t - length                             # last frame shown as tracked
        ax, ay = x[a, s], y[a, s]
        if not (np.isfinite(ax) and np.isfinite(ay)):
            continue
        hx, hy = (ax - x[a - 1, s], ay - y[a - 1, s]) if a >= 1 else (0.0, 0.0)
        norm = np.hypot(hx, hy)
        if not np.isfinite(norm) or norm == 0:
            angle = rng.uniform(0, 2 * np.pi)
            hx, hy = np.cos(angle), np.sin(angle)
        else:
            hx, hy = hx / norm, hy / norm
        steps = rng.uniform(low, high, length)
        braking = min(brake, length)
        if np.isfinite(norm) and norm > steps[0]:
            ratio = (steps[0] / norm) ** (1.0 / braking)
            steps[:braking] = np.maximum(norm * ratio ** np.arange(1, braking + 1),
                                         steps[:braking])
        travel = np.cumsum(steps)
        x[a + 1:t + 1, s] = ax + hx * travel
        y[a + 1:t + 1, s] = ay + hy * travel


def _ghosts(match, x, y, rng, params):
    """v1's ghosts (stale same-team tracks in spare slots), with a per-frame jitter."""
    fps = match.fps
    ghosts_x, ghosts_y, ghosts_team, free_from = [], [], [], []
    for start, stop in match.halves:
        for team in (0, 1):
            members = np.flatnonzero(match.team == team)
            position = int(rng.exponential(params["ghost_every"] * fps))
            while position < stop - start:
                t = start + position
                visible = members[np.isfinite(x[t, members])]
                if len(visible):
                    source = int(rng.choice(visible))
                    end = min(stop, t + max(params["min_on_frames"],
                                            _lognormal_frames(rng, params["ghost_length"], fps)))
                    radius = rng.uniform(*params["ghost_offset"])
                    angle = rng.uniform(0, 2 * np.pi)
                    slot = next((i for i, f in enumerate(free_from)
                                 if ghosts_team[i] == team and f <= t), None)
                    if slot is None:
                        slot = len(free_from)
                        ghosts_x.append(np.full(match.n_frames, np.nan, dtype=np.float32))
                        ghosts_y.append(np.full(match.n_frames, np.nan, dtype=np.float32))
                        ghosts_team.append(team)
                        free_from.append(0)
                    jitter = rng.normal(0, params["ghost_jitter_m"], (2, end - t))
                    ghosts_x[slot][t:end] = x[t, source] + radius * np.cos(angle) + jitter[0]
                    ghosts_y[slot][t:end] = y[t, source] + radius * np.sin(angle) + jitter[1]
                    free_from[slot] = end + fps
                position += 1 + int(rng.exponential(params["ghost_every"] * fps))
    if ghosts_x:
        x = np.hstack((x, np.stack(ghosts_x, axis=1)))
        y = np.hstack((y, np.stack(ghosts_y, axis=1)))
    team = np.concatenate((match.team, np.array(ghosts_team, dtype=match.team.dtype)))
    return x, y, team


def degrade_players_v2(match, ball_seen, rng, params=V2):
    """(player_x, player_y, team): losses where BAR-ATH has them, slow-downs, episodes, ghosts."""
    fps = match.fps
    x, y = match.player_x.copy(), match.player_y.copy()
    present = np.isfinite(x) & np.isfinite(y)
    log_rate = loss_log_rate(match, ball_seen, params["loss"])
    for start, stop in match.halves:
        on = _losses(log_rate, start, stop, rng, fps, params)
        _episodes(on, log_rate, x, y, present, start, stop, rng, fps, params)
        on &= present[start:stop]
        _slow_down(x[start:stop], y[start:stop], on, rng, fps, params)
        x[start:stop][~on] = np.nan
        y[start:stop][~on] = np.nan
    return _ghosts(match, x, y, rng, params)
