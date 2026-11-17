"""Make clean external tracking look like the BAR-ATH ATD, so practice scores mean something.

External matches (IDSSE) have the ball on every in-play frame and every player cleanly tracked.
The BAR-ATH automatic tracking loses the ball on ~41% of graded frames, in long blocks (median
~11 s), and its player tracks are broken and noisy. A method validated on clean data would lean on
things it will not have, so every proxy-league score is computed on a degraded copy.

Two versions, both kept so numbers can be compared (`degrade(match, seed, version)`):

v0 degrades the ball only, and only by timing: it alternates visible and missing stretches whose
lengths are drawn from the empirical BAR-ATH distributions. Those distributions are measured
from evaluation/results/baseline_raw.csv, a committed public file in which D marks exactly the
frames where the ATD had no ball -- so nothing here reads the gated match data.

v1 (the default) adds context and messy players, calibrated to the BAR-ATH aggregates in
notebooks/eda/FINDINGS.md ("What it means", points 1 and 4; the numbers are in BAR_ATH below):
  - ball: v0's visible and missing stretches, the same lengths, laid out in a different order so
    that gaps fall in slow play. In BAR-ATH players average 1.17 m/s inside gaps and 2.16 m/s
    outside; v1 gives about 1.24 and 2.26 on IDSSE. Because only the order changes, both length
    distributions stay exactly v0's.
  - players: tracks flicker off briefly, episodes drop a few players at once (sometimes many:
    the p5 of 14 visible), and ghost tracks stay behind where a player was (the frames with 12
    or more on a team). Same-team pairs closer than 0.7 m are not added on purpose: clean IDSSE
    already has them on about 6.6% of frames (walls, set pieces), BAR-ATH on 6.8%.
  `make sim-check` prints every target next to its value on the degraded IDSSE matches.

v2 (solution/simulate_v2.py, opt-in: SIM=v2) keeps v1's ball and loses players where BAR-ATH
loses them: in crowds, at the lines, more on their own when the ball is unseen, after slowing the
track down (EXPERIMENTS section 31). v1 stays the default until a decision says otherwise.

Known gaps between the proxy league and the real task -- read before trusting a number:
  - v1 ties ball gaps to player speed only; the real causes (balls out of play, crowded boxes,
    aerial balls, coverage drifting over each half) are not modelled one by one;
  - during dead ball the IDSSE ball usually sits just beyond a line, so the degraded ball is seen
    beyond a line on ~17% of visible frames against BAR-ATH's 4.2%, and gaps follow a ball last
    seen wide or beyond on ~65% against 54%: an out-of-play cue the real ATD rarely gives;
  - player losses are independent of where players are; ghosts copy a teammate, never the other
    team; re-acquired tracks keep their identity (no ID switches); no position noise; the ball is
    never noisy (BAR-ATH has 216 jumps above 40 m/s). `add_ball_noise` is the opt-in stress
    for that (`make proxy BALL_NOISE=0.05 BALL_JUMPS=3.4`);
  - the DFL ball is frozen during dead ball (median speed 0.00 m/s) and relocated at restarts,
    while BAR-ATH's is almost never still (stationary stretches cover 2% of graded frames);
    ball jitter in the stress removes the freeze;
  - label semantics differ: DFL possessions are choppier (mean ~7.7 s, ~29% under 2 s) than the
    BAR-ATH public window suggests, so DFL rewards splitting more than the real truth may;
  - IDSSE halves start at kickoff, so the BAR-ATH second-half pre-kickoff trap (2,934 graded
    frames of dead ball with the ball tracked) is not exercised. `add_prekickoff` is the opt-in
    stress for that (`make proxy PREKICKOFF=117 PREKICKOFF_PROFILE=busy`); its players and ball
    are invented from the laws of the game and a guess at how teams come out, not measured.

Deterministic: the same (match name, seed, version) always gives the same output.
"""

import os
import warnings
import zlib
from functools import lru_cache

import numpy as np

import games
from solution.match import ROOT, pm

BASELINE_RAW = os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")
VERSIONS = ("v0", "v1", "v2")
DEFAULT_VERSION = "v1"
TRAIN_SIM_ENV = "MTAH_TRAIN_SIM"


def train_version():
    """Simulator version for the IDSSE matches the trained models learn from (frame, dead-ball and
    control models). $MTAH_TRAIN_SIM, else DEFAULT_VERSION, so solution/run.py and every past
    number are unchanged unless it is set. The models cache their training sets per process:
    set it before the first model is fitted (tools/proxy_league.py --train-simulator does)."""
    version = os.environ.get(TRAIN_SIM_ENV) or DEFAULT_VERSION
    if version not in VERSIONS:
        raise ValueError("{}={!r}; choose from {}".format(TRAIN_SIM_ENV, version, VERSIONS))
    return version

# BAR-ATH ATD aggregates the simulator is calibrated to (notebooks/eda/FINDINGS.md, A2-A4).
BAR_ATH = {
    "ball_visible": 0.593,           # gap and visible-stretch lengths: ball_profile()
    "speed_in_gaps": 1.17,
    "speed_ball_visible": 2.16,
    "gap_after_wide_or_beyond": 81 / 150.0,
    "visible_ball_beyond_line": 0.042,
    "objects_mean": 20.49,
    "objects_p": {5: 14, 25: 21, 50: 22},
    "team_over_11": (0.047, 0.081),
    "team_under_9": (0.081, 0.096),
    "tracklet_s": {10: 2.60, 25: 7.92, 50: 35.52, 75: 145.32, 90: 284.63},
    "duplicate_frames": 0.068,
    # Where player tracks end (tools/track_loss_check.py --bar-ath --fit, EXPERIMENTS section
    # 31): 1,198 ends; per-frame log rate ratios of the fit; share of ends that follow at least
    # one nearly-still frame (moves under 2 cm). Simulator v2 is calibrated to these.
    "track_loss": {
        "rate_per_player_minute": 0.552,
        "ends_after_still": 1.0 - 0.362,
        "fit": {
            "intercept": -10.0044,
            "nearest, any team = 0-1 m": 2.9554, "nearest, any team = 1-2 m": 2.1722,
            "nearest, any team = 2-3 m": 0.751, "nearest, any team = 3-5 m": 0.0723,
            "opponent within 2 m = yes": -0.0031,
            "touchline = off pitch": 2.0106, "touchline = 0-2 m": 0.8801,
            "touchline = 2-5 m": 0.8119,
            "goal line = off pitch": 2.2001, "goal line = 0-2 m": 0.6229,
            "ball = unseen": 0.6057,
            "player speed = 0-1 m/s": 1.9928, "player speed = >= 3 m/s": -1.9793,
            "player speed = unknown": -4.1527,
            "others within 3 m = 2": 0.3403, "others within 3 m = 3+": 0.724,
            "player speed 1-2 s earlier = 0-1 m/s": -1.0341,
            "player speed 1-2 s earlier = >= 3 m/s": 0.779,
            "player speed 1-2 s earlier = unknown": -0.675,
            "nearest < 2 m x ball unseen = yes": -0.654,
        },
    },
}
DUPLICATE_M = 0.7                    # same-team points closer than this (EDA A3)
WIDE_M = 15.0                        # wide channel: within this of a touchline (EDA A4)
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16

# v1 parameters. Times are seconds; lognormal sizes are (median, sigma of the log).
V1 = {
    # ball gaps
    "speed_weight": 1.0,             # log-probability per metre run in gaps by the mean player
    "sweeps": 100,                   # Metropolis swaps per stretch
    # player flicker, per track: brief losses that break tracks into pieces
    "track_on": (30.0, 2.7),
    "track_off": (0.8, 1.0),
    # episodes that drop several players at once; they make most of the missing players
    "episode_every": 50.0,           # mean seconds between episode starts
    "episode_length": (11.0, 0.8),
    "episode_drop": 4.5,             # mean players dropped, geometric
    # ghosts: a stale track left where a player was, while the player moves on
    "ghost_every": 70.0,             # mean seconds between ghost starts, per team
    "ghost_length": (3.0, 1.2),
    "ghost_offset": (0.2, 0.6),      # metres from the player when it starts, uniform
}


def _runs(mask, start, stop):
    window = np.concatenate(([False], mask[start:stop], [False])).astype(np.int8)
    edges = np.diff(window)
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


@lru_cache(maxsize=1)
def ball_profile():
    """(visible_lengths, missing_lengths, visible_share) of the BAR-ATH ATD ball, in frames."""
    raw = pm.label_codes(pm.read_label_csv(BASELINE_RAW))
    missing = raw == pm.D
    visible_runs, missing_runs, seen, total = [], [], 0, 0
    for _, first, last in games.half_video_blocks("BAR-ATH"):
        visible_runs.append(_runs(~missing, first, last + 1))
        missing_runs.append(_runs(missing, first, last + 1))
        seen += int(np.sum(~missing[first:last + 1]))
        total += last + 1 - first
    return (np.concatenate(visible_runs), np.concatenate(missing_runs), seen / float(total))


def _rng(match, seed, stream=0):
    return np.random.RandomState(
        (zlib.crc32(match.name.encode()) + 7919 * seed + 104729 * stream) % (2 ** 32))


def degrade(match, seed=0, version=DEFAULT_VERSION, ball_visible=None, players=True):
    """A copy of `match` with the ball and players degraded the way the BAR-ATH ATD has them.

    `ball_visible` (a share, e.g. 0.76) is an opt-in what-if for a better ball tracker: see
    reveal_gaps. None, the default, leaves every version exactly as it was. `players` False
    (v1 only) keeps the clean players: the what-if for a perfect player tracker.
    """
    if version == "v0":
        if not players:
            raise ValueError("players=False is a simulator v1/v2 option")
        return degrade_v0(match, seed, ball_visible=ball_visible)
    if version == "v1":
        return degrade_v1(match, seed, ball_visible=ball_visible, players=players)
    if version == "v2":
        return degrade_v2(match, seed, ball_visible=ball_visible, players=players)
    raise ValueError("unknown simulator version {!r}; choose from {}".format(version, VERSIONS))


def add_position_noise(match, sd, seed=0):
    """A copy with white Gaussian noise of `sd` metres on every player coordinate, per frame.

    A stress test, not part of any simulator version: BAR-ATH position noise is not measured,
    and speed from positions is sensitive to it (white noise of sd s adds about 8.8 s m/s per
    player at the 2-frame lag of mean_player_speed). Its own random stream, so the ball gaps
    and player losses of the version underneath do not move.
    """
    out = match.copy()
    if sd <= 0:
        return out
    rng = _rng(match, seed, 3)
    out.player_x = (out.player_x + rng.normal(0, sd, out.player_x.shape)).astype(np.float32)
    out.player_y = (out.player_y + rng.normal(0, sd, out.player_y.shape)).astype(np.float32)
    out.notes["position_noise"] = sd
    return out

def add_ball_noise(match, sd, jumps_per_min=0.0, seed=0, hop_m=(1.6, 4.4), hop_frames=3.0,
                   smooth_s=0.0, causal=False):
    """A copy with a noisy ball: white jitter of `sd` metres, a tracker's smoothing, short hops.

    A stress test, not part of any simulator version (the IDSSE ball is clean). Hops mimic the
    BAR-ATH teleports: 216 frame-to-frame jumps above 40 m/s on 94,066 visible frame pairs, all
    under 110 m/s (FINDINGS A2), so each moves the ball 1.6 to 4.4 m in one frame. A hop
    displaces the ball by a uniform `hop_m` distance in a random direction for a geometric
    number of frames (mean `hop_frames`), then it snaps back: two jumps per hop. Hops start at
    `jumps_per_min` / 2 per minute of visible ball, so `jumps_per_min` counts jumps as the EDA
    does; BAR-ATH has about 3.4 (216 per 62.7 visible minutes). Jitter also unfreezes the IDSSE
    dead ball, which DFL holds perfectly still. Its own random stream, like add_position_noise.

    `smooth_s` > 0 then smooths the jittered ball with a Gaussian of that sd in seconds, inside
    each visible stretch, before the hops are added. The BAR-ATH ball is smoothed this way:
    the lag-1 autocorrelation of its frame accelerations is +0.99, against -0.10 for IDSSE, and
    its velocity changes are spread over a few tenths of a second
    (`tools/touch_check.py --diagnose --bar-ath`, EXPERIMENTS section 5).

    `causal` True (a what-if for tools/ball_lag_check.py, default off) smooths with past frames
    only instead: an exponential kernel of time constant `smooth_s`, as a tracker's filter
    would, so the ball trails the players by about `smooth_s`.
    """
    out = match.copy()
    if sd <= 0 and jumps_per_min <= 0 and smooth_s <= 0:
        return out
    rng = _rng(match, seed, 4)
    seen = match.ball_seen
    x, y = out.ball_x.astype(np.float64), out.ball_y.astype(np.float64)
    if sd > 0:
        x += rng.normal(0, sd, len(x))
        y += rng.normal(0, sd, len(y))
    if smooth_s > 0:
        x, y = _smooth_stretches(x, seen, smooth_s * match.fps, causal), \
            _smooth_stretches(y, seen, smooth_s * match.fps, causal)
    if jumps_per_min > 0:
        rate = jumps_per_min / 2.0 / (60.0 * match.fps)          # hop starts per visible frame
        for t in np.flatnonzero(seen & (rng.rand(len(x)) < rate)):
            length = int(rng.geometric(1.0 / hop_frames))
            distance, angle = rng.uniform(*hop_m), rng.uniform(0, 2 * np.pi)
            x[t:t + length] += distance * np.cos(angle)
            y[t:t + length] += distance * np.sin(angle)
    out.ball_x = np.where(seen, x, np.nan).astype(np.float32)
    out.ball_y = np.where(seen, y, np.nan).astype(np.float32)
    out.notes["ball_noise"] = (sd, jumps_per_min, smooth_s) + (("causal",) if causal else ())
    return out


def thin_beyond(match, keep, seed=0, margin_m=0.0):
    """A copy with each stretch of ball seen more than `margin_m` beyond a line kept with
    probability `keep`, else hidden.

    A stress for cues that read the ball beyond a line (D7, C2b): DFL holds the dead ball just
    outside the line, so degraded IDSSE sees it beyond a line on 16% of visible frames and
    BAR-ATH on 4.2%; beyond by more than 1 m, 3.3% against 2.7% (FINDINGS A2). `keep` = BAR-ATH's
    share over the proxy's, at the cue's own margin, gives the cue BAR-ATH's rate. Hidden, not
    moved: BAR-ATH loses the ball there (FINDINGS A4). Apply after add_ball_noise. Its own
    random stream.
    """
    out = match.copy()
    if keep >= 1:
        return out
    rng = _rng(match, seed, 7)
    length, width = match.pitch
    x, y = match.ball_x, match.ball_y
    with np.errstate(invalid="ignore"):
        beyond = ((x < -margin_m) | (x > length + margin_m) |
                  (y < -margin_m) | (y > width + margin_m))
    edges = np.diff(np.concatenate(([0], beyond.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if rng.rand() >= keep:
            out.ball_x[a:b] = np.nan
            out.ball_y[a:b] = np.nan
    out.notes["thin_beyond"] = (keep, margin_m)
    return out


def _smooth_stretches(values, seen, sigma, causal=False):
    """Gaussian smoothing of sd `sigma` frames inside each run of `seen`, renormalised at ends.

    `causal`: past frames only, an exponential kernel of time constant `sigma` frames (5 time
    constants long), also renormalised at the start of each run."""
    if causal:
        return _smooth_stretches_causal(values, seen, sigma)
    reach = max(1, int(np.ceil(3 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) / float(sigma)) ** 2)
    out = values.copy()
    edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        part = values[a:b]
        weight = np.convolve(np.ones(b - a), kernel, mode="same")
        if b - a < len(kernel):            # np.convolve "same" pads to the longer input
            weight = np.convolve(np.ones(b - a), kernel)[reach:reach + b - a]
            out[a:b] = np.convolve(part, kernel)[reach:reach + b - a] / weight
        else:
            out[a:b] = np.convolve(part, kernel, mode="same") / weight
    return out


def _smooth_stretches_causal(values, seen, tau):
    """Past-only exponential smoothing of time constant `tau` frames inside each run of `seen`."""
    kernel = np.exp(-np.arange(max(1, int(np.ceil(5 * tau))) + 1) / float(tau))
    out = values.copy()
    edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        n = b - a
        out[a:b] = np.convolve(values[a:b], kernel)[:n] / np.convolve(np.ones(n), kernel)[:n]
    return out


# Two pre-kickoff profiles. calm: players settle early and stand, the ball rests on the mark for
# a while. busy: they mill about at walking-to-jogging pace and pass the ball until the last
# seconds. Neither is fitted to BAR-ATH; tools/kickoff_check.py --bar-ath prints what to fit.
PREKICKOFF_PROFILES = {
    "calm": {"walk_ms": (0.8, 1.5), "arrive": (0.3, 0.8), "placed": (0.5, 0.85), "wander": False},
    "busy": {"walk_ms": (1.0, 2.0), "arrive": (0.85, 0.95), "placed": (0.8, 0.9), "wander": True},
}


def add_prekickoff(match, seconds, seed=0, half=-1, profile="calm", pass_ms=10.0, kick_ms=12.0,
                   sway_m=0.03):
    """A copy with `seconds` of synthetic pre-kickoff dead ball in front of a half (the last).

    A stress test for the kickoff grammar (solution/kickoff.py), not part of any simulator
    version: IDSSE halves start at the kick, while the BAR-ATH second half is graded from about
    117 s before it (PLAN.md finding 9). Apply it to the clean match, then degrade as usual;
    degrade_v1 gives the stretch BAR-ATH's ball stretches rather than v1's. The inserted
    stretch, all labelled D, with the `profile`'s settings (PREKICKOFF_PROFILES):
      - players walk at `walk_ms` from up to 35 m away into the kickoff formation: their
        positions at the half's first frame, pulled back into their own half (Law 8), arriving
        at an `arrive` fraction of the way through, then stand with `sway_m` of jitter. With
        `wander` they mill about among random points within 20 m of their place until then;
      - the ball is played among random players (held at a player's feet, passed at
        `pass_ms`) until it is rolled to the centre mark at a `placed` fraction of the way
        through, and rests there;
      - in the last frames the kick moves it at `kick_ms` to where the half's first frame has
        it, the players likewise; those frames take the half's first label.
    Every later frame shifts by the inserted length. notes["prekickoff"] gives the half, the
    length and the true kick frame. Its own random stream; labels do not depend on the seed.
    """
    out = match.copy()
    n = int(round(seconds * match.fps))
    if n <= 0:
        return out
    rng = _rng(match, seed, 5)
    fps = float(match.fps)
    index = range(len(match.halves))[half]
    a, b = match.halves[index]
    length, width = match.pitch
    cx, cy = length / 2.0, width / 2.0
    x0, y0 = match.player_x[a].astype(np.float64), match.player_y[a].astype(np.float64)
    here = np.isfinite(x0) & np.isfinite(y0)
    side = np.zeros(len(x0))
    for team in (0, 1):
        members = here & (match.team == team)
        side[match.team == team] = (np.sign(np.median(x0[members] - cx)) if members.any()
                                    else (-1.0 if team == 0 else 1.0)) or 1.0
    tx = np.where(here, cx + side * np.maximum(side * (x0 - cx), 0.5), np.nan)
    ty = y0.copy()

    ball_end = np.array([match.ball_x[a], match.ball_y[a]], dtype=np.float64)
    if not np.all(np.isfinite(ball_end)):
        ball_end = np.array([cx, cy])
    kick_frames = int(np.clip(np.ceil(np.hypot(*(ball_end - (cx, cy))) / kick_ms * fps),
                              1, 2 * fps))
    kick = n - kick_frames                     # first frame of the kick, within the stretch
    t = np.arange(n, dtype=np.float64)

    px = np.full((n, len(x0)), np.nan)
    py = np.full((n, len(x0)), np.nan)
    setting = PREKICKOFF_PROFILES[profile]
    for slot in np.flatnonzero(here):
        speed = rng.uniform(*setting["walk_ms"])
        arrive = rng.uniform(*setting["arrive"]) * kick
        distance = min(rng.uniform(5.0, 35.0), speed * arrive / fps)
        angle = rng.uniform(0, 2 * np.pi)
        sx = np.clip(tx[slot] + distance * np.cos(angle), 1.0, length - 1.0)
        sy = np.clip(ty[slot] + distance * np.sin(angle), 1.0, width - 1.0)
        leave = arrive - np.hypot(sx - tx[slot], sy - ty[slot]) / speed * fps
        way_t, way_x, way_y = [0.0], [sx], [sy]
        while setting["wander"] and way_t[-1] < leave:
            wx = np.clip(tx[slot] + rng.uniform(-20.0, 20.0), 1.0, length - 1.0)
            wy = np.clip(ty[slot] + rng.uniform(-20.0, 20.0), 1.0, width - 1.0)
            way_t.append(way_t[-1] + max(1.0, np.hypot(wx - way_x[-1], wy - way_y[-1])
                                         / speed * fps))
            way_x.append(wx)
            way_y.append(wy)
        mill_x, mill_y = np.interp(t, way_t, way_x), np.interp(t, way_t, way_y)
        lx, ly = np.interp(leave, way_t, way_x), np.interp(leave, way_t, way_y)
        f = np.clip((t - leave) / max(arrive - leave, 1.0), 0.0, 1.0)
        px[:, slot] = np.where(t < leave, mill_x, lx + (tx[slot] - lx) * f)
        py[:, slot] = np.where(t < leave, mill_y, ly + (ty[slot] - ly) * f)
    px += rng.normal(0, sway_m, px.shape)
    py += rng.normal(0, sway_m, py.shape)

    bx, by = np.full(n, cx), np.full(n, cy)
    placed = int(rng.uniform(*setting["placed"]) * kick)
    players = np.flatnonzero(here)
    frame = 0
    if len(players) and placed > 0:
        carrier = int(rng.choice(players))
        while True:
            # At a player's feet until the next pass, then a pass; the last one goes to the mark.
            hold = max(frame, min(placed, frame + int(rng.uniform(3.0, 12.0) * fps)))
            offset = rng.uniform(0, 2 * np.pi)
            bx[frame:hold] = px[frame:hold, carrier] + 0.6 * np.cos(offset)
            by[frame:hold] = py[frame:hold, carrier] + 0.6 * np.sin(offset)
            last, frame = (bx[max(hold, 1) - 1], by[max(hold, 1) - 1]), hold
            done = frame >= placed
            carrier = int(rng.choice(players))
            target = (cx, cy) if done else (px[frame, carrier], py[frame, carrier])
            flight = max(1, int(np.hypot(target[0] - last[0], target[1] - last[1])
                                / pass_ms * fps))
            stop = min(n, frame + flight)
            f = (np.arange(frame, stop) - frame + 1) / float(flight)
            bx[frame:stop] = last[0] + (target[0] - last[0]) * f
            by[frame:stop] = last[1] + (target[1] - last[1]) * f
            frame = stop
            if done:
                break
    rest = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.3)
    after = np.arange(n) >= frame
    bx[after] = cx + rest[1] * np.cos(rest[0])
    by[after] = cy + rest[1] * np.sin(rest[0])
    f = (np.arange(kick, n) - kick + 1) / float(kick_frames)
    bx[kick:] = bx[kick - 1] + (ball_end[0] - bx[kick - 1]) * f
    by[kick:] = by[kick - 1] + (ball_end[1] - by[kick - 1]) * f
    px[kick:] = tx + (x0 - tx) * f[:, None]
    py[kick:] = ty + (y0 - ty) * f[:, None]

    labels = None
    if match.labels is not None:
        labels = np.insert(match.labels, a, np.full(n, pm.D, dtype=match.labels.dtype))
        labels[a + kick:a + n] = match.labels[a]
    out.player_x = np.insert(match.player_x, a, px.astype(np.float32), axis=0)
    out.player_y = np.insert(match.player_y, a, py.astype(np.float32), axis=0)
    out.ball_x = np.insert(match.ball_x, a, bx.astype(np.float32))
    out.ball_y = np.insert(match.ball_y, a, by.astype(np.float32))
    out.labels = labels
    out.halves = [(s, e) if i < index else (s + (n if i > index else 0), e + n)
                  for i, (s, e) in enumerate(match.halves)]
    out.notes["prekickoff"] = {"half": index, "frames": n, "kick": a + kick, "profile": profile}
    return out


def add_walkoff(match, seconds, seed=0, walk_ms=(1.1, 1.6), linger=(0.0, 0.6), carry_p=0.5,
                stay_p=0.1, sway_m=0.03):
    """A copy with `seconds` of synthetic walk-off after the end of every half, all labelled D.

    A stress test for the end-of-half rule (whistlecue.end_of_half, D48), not part of any
    simulator version: IDSSE tracking stops at the half's whistle, while BAR-ATH's tracking
    blocks, the graded halves, run on while the players leave (EXPERIMENTS section 20). Apply it
    to the clean match, then degrade as usual. In the inserted stretch, from each player's place
    at the half's last frame:
      - each player lingers (handshakes) for a `linger` fraction of it, swaying by `sway_m`, then
        walks at `walk_ms` (people walk at about 1.4 m/s) to a tunnel beyond one touchline, and is
        no longer tracked once 2 m past it; a `stay_p` share stay on the pitch, ambling;
      - with probability `carry_p` a player picks up the ball after 1-5 s and carries it at his
        feet until he leaves (the ball is then not seen); otherwise it rests where it was.
    The ball near a moving player is what a decoder can read as possession. Every later frame
    shifts by the inserted length. notes["walkoff"] gives the frames per half, where each was
    inserted (in the input's timeline) and the first inserted frame of each half (in the new
    one). Its own random stream; labels do not depend on the seed.
    """
    out = match.copy()
    n = int(round(seconds * match.fps))
    if n <= 0:
        return out
    rng = _rng(match, seed, 14)
    fps = float(match.fps)
    length, width = match.pitch
    side = width + 3.0 if rng.rand() < 0.5 else -3.0
    tunnel = np.array([length / 2.0 + rng.uniform(-15.0, 15.0), side])
    t = np.arange(n, dtype=np.float64)
    blocks = []
    for a, b in match.halves:
        last = b - 1
        x0 = match.player_x[last].astype(np.float64)
        y0 = match.player_y[last].astype(np.float64)
        here = np.isfinite(x0) & np.isfinite(y0)
        px = np.full((n, len(x0)), np.nan)
        py = np.full((n, len(x0)), np.nan)
        for slot in np.flatnonzero(here):
            start = rng.uniform(*linger) * n
            speed = rng.uniform(*walk_ms)
            if rng.rand() < stay_p:
                goal = np.array([np.clip(x0[slot] + rng.uniform(-15, 15), 1, length - 1),
                                 np.clip(y0[slot] + rng.uniform(-10, 10), 1, width - 1)])
                speed *= 0.4
            else:
                goal = tunnel
            dist = np.hypot(*(goal - (x0[slot], y0[slot])))
            f = np.clip((t - start) * speed / fps / max(dist, 1e-6), 0.0, 1.0)
            px[:, slot] = x0[slot] + (goal[0] - x0[slot]) * f
            py[:, slot] = y0[slot] + (goal[1] - y0[slot]) * f
            gone = (py[:, slot] > width + 2.0) | (py[:, slot] < -2.0)
            px[gone, slot] = np.nan
            py[gone, slot] = np.nan
        px += rng.normal(0, sway_m, px.shape)
        py += rng.normal(0, sway_m, py.shape)
        bx = np.full(n, match.ball_x[last], dtype=np.float64)
        by = np.full(n, match.ball_y[last], dtype=np.float64)
        players = np.flatnonzero(here)
        if len(players) and rng.rand() < carry_p:
            carrier = int(rng.choice(players))
            pick = int(rng.uniform(1.0, 5.0) * fps)
            if np.isfinite(bx[0]) and np.isfinite(by[0]):
                f = np.clip(t[:pick] / max(pick, 1), 0, 1)
                bx[:pick] = bx[0] + (px[pick, carrier] - bx[0]) * f
                by[:pick] = by[0] + (py[pick, carrier] - by[0]) * f
            offset = rng.uniform(0, 2 * np.pi)
            bx[pick:] = px[pick:, carrier] + 0.4 * np.cos(offset)
            by[pick:] = py[pick:, carrier] + 0.4 * np.sin(offset)
        blocks.append((b, px, py, bx, by))

    labels = None if match.labels is None else match.labels.copy()
    out.player_x, out.player_y = match.player_x.copy(), match.player_y.copy()
    out.ball_x, out.ball_y = match.ball_x.copy(), match.ball_y.copy()
    for b, px, py, bx, by in reversed(blocks):        # from the end, so `b` stays valid
        out.player_x = np.insert(out.player_x, b, px.astype(np.float32), axis=0)
        out.player_y = np.insert(out.player_y, b, py.astype(np.float32), axis=0)
        out.ball_x = np.insert(out.ball_x, b, bx.astype(np.float32))
        out.ball_y = np.insert(out.ball_y, b, by.astype(np.float32))
        if labels is not None:
            labels = np.insert(labels, b, np.full(n, pm.D, dtype=labels.dtype))
    out.labels = labels
    out.halves = [(s + i * n, e + (i + 1) * n) for i, (s, e) in enumerate(match.halves)]
    out.notes["walkoff"] = {"frames": n, "at": [int(e) for _, e in match.halves],
                            "ends": [int(e + i * n) for i, (_, e) in enumerate(match.halves)]}
    return out


def scale_speeds(match, factor):
    """A copy with every position shrunk towards the pitch centre by `factor`.

    A stress test for tempo: every speed is multiplied by `factor`, while who is nearest the
    ball does not change (all distances scale alike). BAR-ATH's collective speed is about 0.9
    times IDSSE's (tools/speed_check.py, EXPERIMENTS section 3).
    """
    out = match.copy()
    if factor == 1:
        return out
    cx, cy = match.pitch[0] / 2.0, match.pitch[1] / 2.0
    out.player_x = (cx + (out.player_x - cx) * factor).astype(np.float32)
    out.player_y = (cy + (out.player_y - cy) * factor).astype(np.float32)
    out.ball_x = (cx + (out.ball_x - cx) * factor).astype(np.float32)
    out.ball_y = (cy + (out.ball_y - cy) * factor).astype(np.float32)
    out.notes["speed_factor"] = factor
    return out


def retime(match, factor):
    """A copy played `factor` times as fast, with every position unchanged (factor < 1: slower).

    The D9 tempo stress for anything that reads positions (EXPERIMENTS section 16).
    scale_speeds reaches the same speeds by shrinking the pitch, which also moves players away
    from the lines; this resamples time instead. New frame k shows the old match at time
    t = k * factor: positions (players and ball) are interpolated linearly between the two old
    frames around t, NaN if either is missing; labels and halves are those of old frame
    floor(t), so the truth and in_play_mask() stay consistent. Every speed is multiplied by
    `factor`, where anyone stands is unchanged, and the match is 1 / factor times as long.
    Apply it to the clean match, before degrade(): real slower play is tracked as it is.
    """
    out = match.copy()
    if factor == 1:
        return out
    n = match.n_frames
    t = np.arange(int(np.ceil(n / float(factor)))) * float(factor)
    t = t[t <= n - 1]
    lo = np.floor(t).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    w = (t - lo).astype(np.float32)

    def resample(values):
        a, b = values[lo], values[hi]
        if values.ndim == 2:
            mix = a * (1 - w[:, None]) + b * w[:, None]
            exact = (w == 0)[:, None]
        else:
            mix = a * (1 - w) + b * w
            exact = w == 0
        return np.where(exact, a, mix).astype(values.dtype)

    out.player_x, out.player_y = resample(match.player_x), resample(match.player_y)
    out.ball_x, out.ball_y = resample(match.ball_x), resample(match.ball_y)
    if match.labels is not None:
        out.labels = match.labels[lo].copy()
    # A new frame is in a half when its old frame floor(t) is: the same rule as the labels.
    out.halves = [(int(np.searchsorted(lo, a)), int(np.searchsorted(lo, b)))
                  for a, b in match.halves]
    out.notes["tempo"] = factor
    return out


def retime_frame(frame, factor):
    """The first frame of retime(match, factor) that shows old frame `frame` (event times)."""
    return int(np.ceil(frame / float(factor) - 1e-9))

# -- v0: ball timing only ------------------------------------------------------------------------

def ball_dropout_mask(n_frames, rng):
    """Boolean mask of frames whose ball is removed, as an alternating renewal process."""
    visible, missing, visible_share = ball_profile()
    mask = np.zeros(n_frames, dtype=bool)
    hidden = rng.rand() >= visible_share
    # Start part-way through the first stretch, so a half does not always open with a complete
    # one: a stationary start for the alternating process.
    first = int(rng.choice(missing if hidden else visible))
    position = 1 + int(rng.randint(first))
    mask[:position] = hidden
    hidden = not hidden
    while position < n_frames:
        length = int(rng.choice(missing if hidden else visible))
        mask[position:position + length] = hidden
        position += length
        hidden = not hidden
    return mask


def reveal_gaps(hide, target, rng):
    """A copy of a missing-ball mask with whole gaps shown again until about `target` is visible.

    The what-if behind `degrade(..., ball_visible=)`: a better tracker that finds the ball in
    some of the gaps the ATD loses. Whole gaps are revealed, in a random order, so the gaps
    that remain keep v1's placement in slow play and the BAR-ATH gap-length distribution, and
    a revealed gap is not tied to any context (an unbiased "some gaps recovered"). Shrinking
    every gap from its edges instead would reveal exactly the starts and ends of stoppages and
    shift the length distribution. A gap is skipped when revealing it would land further from
    the target than leaving it, so the share ends close to the target. No-op when the mask
    already shows at least `target`; `target` None or >= 1 reveals nothing or everything.
    """
    hide = hide.copy()
    if target is None:
        return hide
    budget = (1.0 - min(1.0, float(target))) * len(hide)        # missing frames allowed
    hidden = int(hide.sum())
    if hidden <= budget:
        return hide
    edges = np.diff(np.concatenate(([0], hide.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for k in rng.permutation(len(starts)):
        if hidden <= budget:
            break
        length = int(stops[k] - starts[k])
        if hidden - length < budget and budget - (hidden - length) > hidden - budget:
            continue
        hide[starts[k]:stops[k]] = False
        hidden -= length
    return hide


def degrade_v0(match, seed=0, ball_visible=None):
    out = match.copy()
    rng = _rng(match, seed)
    reveal_rng = _rng(match, seed, 8)
    for start, stop in match.halves:
        hide = ball_dropout_mask(stop - start, rng)
        if ball_visible is not None:
            hide = reveal_gaps(hide, ball_visible, reveal_rng)
        out.ball_x[start:stop][hide] = np.nan
        out.ball_y[start:stop][hide] = np.nan
    out.notes["degraded"] = {"version": "v0-ball-timing", "seed": seed}
    if ball_visible is not None:
        out.notes["degraded"]["ball_visible"] = ball_visible
    return out


# -- v1: ball gaps in context, messy players -----------------------------------------------------

def mean_player_speed(match, lag=2):
    """Mean speed (m/s) of the tracked players on each frame, from positions over +-lag frames."""
    x, y = match.player_x, match.player_y
    dx = np.full(x.shape, np.nan, dtype=np.float32)
    dy = np.full(y.shape, np.nan, dtype=np.float32)
    dx[lag:-lag] = x[2 * lag:] - x[:-2 * lag]
    dy[lag:-lag] = y[2 * lag:] - y[:-2 * lag]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # frames with nobody
        return np.nanmean(np.hypot(dx, dy), axis=1) * match.fps / (2.0 * lag)


def zones(match):
    """Per-frame ball zone as in the EDA (A4): 0 middle, 1 box, 2 wide, 3 beyond, -1 unknown."""
    length, width = match.pitch
    x, y = match.ball_x, match.ball_y
    out = np.zeros(match.n_frames, dtype=np.int8)
    box = (((x < BOX_DEPTH) | (x > length - BOX_DEPTH)) &
           (np.abs(y - width / 2) < BOX_HALF_WIDTH))
    out[box] = 1
    out[~box & ((y < WIDE_M) | (y > width - WIDE_M))] = 2
    out[(x < 0) | (x > length) | (y < 0) | (y > width)] = 3
    out[~(np.isfinite(x) & np.isfinite(y))] = -1
    return out


def ball_gap_mask_v1(speed, rng, fps, params=V1):
    """Missing-ball mask for one half: v0's stretches, reordered so gaps fall in slow play.

    First the half's visible and missing stretch lengths are drawn exactly as v0 draws them.
    Then their order is resampled: orders are drawn with probability proportional to
    exp(-speed_weight * metres the average player runs while the ball is missing), by Metropolis
    swaps of two visible or two missing stretches, starting from v0's order. Only the order
    changes, so both length distributions are v0's exactly; with zero weight this is v0.
    """
    n = len(speed)
    drawn = ball_dropout_mask(n, rng)
    weight = params["speed_weight"]
    if not weight:
        return drawn
    pieces = np.split(drawn, np.flatnonzero(np.diff(drawn.astype(np.int8))) + 1)
    lengths = np.array([len(p) for p in pieces])
    hidden = np.array([bool(p[0]) for p in pieces])
    # The first stretch is partial, so it stays first; the others alternate as drawn.
    gaps = np.flatnonzero(hidden[1:]) + 1
    shown = np.flatnonzero(~hidden[1:]) + 1
    if len(gaps) < 2:
        return drawn
    total = _cumulative(speed)

    def energy(order):
        stop = np.cumsum(order)
        start = stop - order
        return weight * np.sum(total[stop[gaps]] - total[start[gaps]]) / fps

    order = lengths.copy()
    current = energy(order)
    for _ in range(params["sweeps"] * (len(lengths) - 1)):
        group = gaps if (rng.rand() < 0.5 or len(shown) < 2) else shown
        i, j = rng.choice(group, 2, replace=False)
        if order[i] == order[j]:
            continue
        order[i], order[j] = order[j], order[i]
        proposed = energy(order)
        if proposed <= current or rng.rand() < np.exp(current - proposed):
            current = proposed
        else:
            order[i], order[j] = order[j], order[i]
    mask = np.zeros(n, dtype=bool)
    stop = np.cumsum(order)
    for k in np.flatnonzero(hidden):
        mask[stop[k] - order[k]:stop[k]] = True
    return mask


def _cumulative(values):
    """Running sum with a leading 0, NaN read as the mean; window sums are differences."""
    finite = np.isfinite(values)
    values = np.where(finite, values, np.mean(values[finite]) if finite.any() else 0.0)
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def _lognormal(rng, size, fps):
    """Frames drawn from a (median seconds, sigma) lognormal; at least one."""
    median, sigma = size
    return max(1, int(round(median * fps * np.exp(sigma * rng.randn()))))


def _flicker(n, rng, fps, params):
    """On/off mask for one track over n frames: lognormal on and off stretches."""
    on = np.ones(n, dtype=bool)
    mean_on = params["track_on"][0] * np.exp(params["track_on"][1] ** 2 / 2)
    mean_off = params["track_off"][0] * np.exp(params["track_off"][1] ** 2 / 2)
    tracked = rng.rand() < mean_on / (mean_on + mean_off)
    position = 1 + int(rng.randint(_lognormal(rng, params["track_on" if tracked else "track_off"],
                                              fps)))
    on[:position] = tracked
    while position < n:
        tracked = not tracked
        length = _lognormal(rng, params["track_on" if tracked else "track_off"], fps)
        on[position:position + length] = tracked
        position += length
    return on


def _poisson_starts(n, rng, fps, every):
    starts, position = [], int(rng.exponential(every * fps))
    while position < n:
        starts.append(position)
        position += 1 + int(rng.exponential(every * fps))
    return starts


def degrade_players_v1(match, rng, params=V1):
    """(player_x, player_y, team) with flicker, multi-player episodes and stale ghost tracks."""
    fps = match.fps
    x, y = match.player_x.copy(), match.player_y.copy()
    present = np.isfinite(x) & np.isfinite(y)
    on = np.ones(x.shape, dtype=bool)
    for start, stop in match.halves:
        n = stop - start
        for slot in range(x.shape[1]):
            on[start:stop, slot] = _flicker(n, rng, fps, params)
        for t in _poisson_starts(n, rng, fps, params["episode_every"]):
            t += start
            here = np.flatnonzero(present[t])
            count = min(len(here), int(rng.geometric(1.0 / params["episode_drop"])))
            drop = rng.choice(here, count, replace=False)
            on[t:min(stop, t + _lognormal(rng, params["episode_length"], fps)), drop] = False
    x[~on] = np.nan
    y[~on] = np.nan

    # Ghosts go in spare same-team slots; a slot is reused only after a one-second pause, so
    # each ghost is its own tracklet.
    ghosts_x, ghosts_y, ghosts_team, free_from = [], [], [], []
    for start, stop in match.halves:
        for team in (0, 1):
            members = np.flatnonzero(match.team == team)
            for t in _poisson_starts(stop - start, rng, fps, params["ghost_every"]):
                t += start
                visible = members[np.isfinite(x[t, members])]
                if not len(visible):
                    continue
                source = int(rng.choice(visible))
                end = min(stop, t + _lognormal(rng, params["ghost_length"], fps))
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
                ghosts_x[slot][t:end] = x[t, source] + radius * np.cos(angle)
                ghosts_y[slot][t:end] = y[t, source] + radius * np.sin(angle)
                free_from[slot] = end + fps
    if ghosts_x:
        x = np.hstack((x, np.stack(ghosts_x, axis=1)))
        y = np.hstack((y, np.stack(ghosts_y, axis=1)))
    team = np.concatenate((match.team, np.array(ghosts_team, dtype=match.team.dtype)))
    return x, y, team


def degrade_v1(match, seed=0, params=V1, ball_visible=None, players=True):
    out = match.copy()
    # Separate streams, so changing the player model does not move the ball gaps.
    ball_rng, player_rng = _rng(match, seed, 1), _rng(match, seed, 2)
    # Its own stream too: the gaps drawn, and everything else, match the default run.
    reveal_rng = _rng(match, seed, 8)
    speed = mean_player_speed(match)
    pre = match.notes.get("prekickoff")
    for i, (start, stop) in enumerate(match.halves):
        # A synthetic pre-kickoff (add_prekickoff) is all slow play, so v1 would hide its ball
        # on about three quarters of it; BAR-ATH sees it on 81% (PLAN finding 9). It takes v0's
        # BAR-ATH stretches instead, on its own stream, and the rest of the half is as without.
        skip = pre["frames"] if pre and pre["half"] == i else 0
        hide = np.zeros(stop - start, dtype=bool)
        hide[skip:] = ball_gap_mask_v1(speed[start + skip:stop], ball_rng, match.fps, params)
        if skip:
            hide[:skip] = ball_dropout_mask(skip, _rng(match, seed, 6))
        if ball_visible is not None:
            hide = reveal_gaps(hide, ball_visible, reveal_rng)
        out.ball_x[start:stop][hide] = np.nan
        out.ball_y[start:stop][hide] = np.nan
    if players:
        out.player_x, out.player_y, out.team = degrade_players_v1(match, player_rng, params)
    out.notes["degraded"] = {"version": "v1", "seed": seed}
    if not players:
        out.notes["degraded"]["players"] = "clean"
    if ball_visible is not None:
        out.notes["degraded"]["ball_visible"] = ball_visible
    return out


# -- v2: v1's ball, players lost where BAR-ATH loses them (solution/simulate_v2.py) ------------

def degrade_v2(match, seed=0, ball_visible=None, players=True):
    """v1's ball exactly (same gaps, same streams); players from simulate_v2."""
    from solution import simulate_v2
    out = degrade_v1(match, seed, ball_visible=ball_visible, players=False)
    if players:
        # Players read the default ball gaps, so the ball_visible what-if (a better ball
        # tracker) changes the ball only, as in v1.
        seen = out.ball_seen if ball_visible is None else \
            degrade_v1(match, seed, players=False).ball_seen
        out.player_x, out.player_y, out.team = simulate_v2.degrade_players_v2(
            match, seen, _rng(match, seed, 2))
    out.notes["degraded"] = {"version": "v2", "seed": seed}
    if not players:
        out.notes["degraded"]["players"] = "clean"
    if ball_visible is not None:
        out.notes["degraded"]["ball_visible"] = ball_visible
    return out
