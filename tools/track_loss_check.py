"""Where do player tracks end? Loss rate by crowding, ball, lines and speed; BAR-ATH vs the simulator.

    .venv/bin/python tools/track_loss_check.py               # clean and simulated IDSSE, no data/
    .venv/bin/python tools/track_loss_check.py --bar-ath     # adds BAR-ATH (needs data/)
    make track-loss [ARGS="--bar-ath --fit --seeds 0 1 2"]

Question (Step 0 for a simulator v2): are BAR-ATH player losses tied to where the player is --
next to another player (occlusion), near the ball, near a line -- while simulator v1 drops
players at random? If BAR-ATH's loss rate climbs as players get close and v1's stays flat, v1
treats easy players and hard players alike and a position-dependent loss model is worth building.

Definition, the same for every source. A player-frame is at risk when slot s is tracked on frame t
and t + 1 is in the same half; the track ends there when slot s is not tracked on t + 1. A track
is a run of one slot, as in the EDA's tracklets (notebooks/eda/FINDINGS.md, A3). Every feature
is read on frame t, the last frame the player is seen, from the same match the loss is counted on:
  - nearest: distance to the nearest other tracked point (any team, teammate, opponent), metres;
  - crowd: other tracked points within CROWD_M;
  - ball: distance to the ball, or "unseen" when the ball is not tracked on t;
  - touchline / goal line: distance to the nearest one, negative off the pitch;
  - speed: from the position 5 frames earlier, "unknown" when that frame is untracked;
  - speed 1-2 s earlier: from the positions EARLY_S and 2 x EARLY_S before t, "unknown" when
    either is untracked or before the half. A tracker that holds a track still before dropping
    it makes the recent speed slow but leaves this one alone;
  - nearest x ball: the two together, since crowds gather around the ball;
  - held still: how many frames in a row, up to t, the player moved less than HOLD_M (frozen:
    under 1 mm, a track the tracker holds still; nearly still: under 2 cm, 0.5 m/s). After the
    tables, the share of track ends per hold length, next to the share of all player-frames:
    this is the freeze before the drop that simulator v2 is meant to copy.

The rate is ends per player-minute. The lift is the bin's rate over the source's overall rate,
so sources with different overall loss rates compare by shape: a lift column that is flat near 1
means losses do not depend on that feature. Bins with fewer than MIN_ENDS ends are marked '*'.

--fit adds a Poisson model of the per-frame loss rate: one rate ratio per level of eight factors
(nearest player, an opponent within 2 m, touchline, goal line, ball seen, speed, players within
3 m, speed 1-2 s earlier) and one interaction (nearest under 2 m while the ball is unseen), so
causes that come together (crowds, slow play, ball gaps) are told apart. Both speeds are in the
model together: if the recent one carries the effect and the earlier one does not, slowness is
the tracker holding the track before it drops, not a cause. It is fitted on cells of
player-frames, not on single frames, so only counts per cell are used. The ball within 10 m
stays out of the model; observed over expected ends on it says what the factors miss.
The last lines print the log rate ratios, the input simulator v2 is meant to take.

Read with care:
  - clean IDSSE ends are substitutions, send-offs and DFL's own gaps; they are the floor;
  - simulated ghosts sit 0.2-2 m from a teammate and end after about 3 s, so v1 has some lift at
    short teammate distances that is not occlusion; BAR-ATH may have the same from its ghosts;
  - only aggregates are printed (counts and rates per bin), nothing per frame.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CROWD_M = 3.0
SPEED_LAG = 5                        # frames back for the speed (0.2 s at 25 fps)
EARLY_S = 1.0                        # the earlier speed runs from 2 x EARLY_S to EARLY_S before t
MIN_ENDS = 30
CHUNK = 4000

NEAR_EDGES = (0, 1, 2, 3, 5, 10, np.inf)
BALL_EDGES = (0, 2, 5, 10, 20, 40, np.inf)
LINE_EDGES = (-np.inf, 0, 2, 5, 15, np.inf)
SPEED_EDGES = (0, 1, 3, 5, 7, np.inf)
CROWD_MAX = 3                        # 0, 1, 2, 3+
CROSS_NEAR = (0, 2, 5, np.inf)
CROSS_BALL = 10.0
HOLD_M = {"frozen": 0.001, "nearly still": 0.02}    # per-frame move under this counts as held
HOLD_EDGES = (0, 1, 2, 5, 10, 25, 50, 125, np.inf)  # frames


def _labels(edges, unit="m"):
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        if low == -np.inf:
            out.append("< {:g} {}".format(high, unit))
        elif high == np.inf:
            out.append(">= {:g} {}".format(low, unit))
        else:
            out.append("{:g}-{:g} {}".format(low, high, unit))
    return out


def _bin(values, edges):
    """Bin index per value; len(edges) - 1 for NaN (the extra 'none' bin)."""
    out = np.full(len(values), len(edges) - 1, dtype=np.int64)
    finite = np.isfinite(values)
    out[finite] = np.clip(np.searchsorted(edges, values[finite], side="right") - 1,
                          0, len(edges) - 2)
    return out


FEATURES = {
    # name: (bin labels, the last one for NaN)
    "nearest, any team": _labels(NEAR_EDGES) + ["nobody else"],
    "nearest teammate": _labels(NEAR_EDGES) + ["none tracked"],
    "nearest opponent": _labels(NEAR_EDGES) + ["none tracked"],
    "others within {:g} m".format(CROWD_M): [str(k) for k in range(CROWD_MAX)]
    + ["{}+".format(CROWD_MAX)],
    "distance to ball": _labels(BALL_EDGES) + ["ball unseen"],
    "distance to touchline": _labels(LINE_EDGES) + ["?"],
    "distance to goal line": _labels(LINE_EDGES) + ["?"],
    "player speed": _labels(SPEED_EDGES, "m/s") + ["unknown"],
    "nearest any x ball": ["{} | ball {}".format(n, b)
                           for n in _labels(CROSS_NEAR) + ["nobody"]
                           for b in ("<= {:g} m".format(CROSS_BALL),
                                     "> {:g} m".format(CROSS_BALL), "unseen")],
    "player speed 1-2 s earlier": _labels(SPEED_EDGES, "m/s") + ["unknown"],
}
HOLD_FEATURES = ["frames held {} (moves < {:g} cm)".format(kind, m * 100)
                 for kind, m in HOLD_M.items()]
HOLD_LABELS = ["0 fr", "1 fr", "2-4 fr", "5-9 fr", "10-24 fr", "25-49 fr", "50-124 fr", "125+ fr"]
assert len(HOLD_LABELS) == len(HOLD_EDGES) - 1
for _name in HOLD_FEATURES:
    FEATURES[_name] = HOLD_LABELS + ["?"]



# --fit: a Poisson model of the per-frame loss rate with one multiplier per factor level.
# Each factor: (name, level labels, reference level). Players with nobody else tracked count as
# ">= 5 m"; no opponent tracked counts as "no".
FIT_NEAR = (0, 1, 2, 3, 5, np.inf)
FIT_TOUCH = (-np.inf, 0, 2, 5, np.inf)
FIT_GOAL = (-np.inf, 0, 2, np.inf)
FIT_SPEED = (0, 1, 3, np.inf)
OPP_M = 2.0
FACTORS = (
    ("nearest, any team", _labels(FIT_NEAR), 4),
    ("opponent within {:g} m".format(OPP_M), ["no", "yes"], 0),
    ("touchline", ["off pitch"] + _labels(FIT_TOUCH)[1:], 3),
    ("goal line", ["off pitch"] + _labels(FIT_GOAL)[1:], 2),
    ("ball", ["seen", "unseen"], 0),
    ("player speed", _labels(FIT_SPEED, "m/s") + ["unknown"], 1),
    # "0 within 3 m" is "nearest >= 3 m", so the crowd factor only splits 0-1 from 2 and 3+:
    # the players beyond the nearest one, which nearest distance cannot see.
    ("others within {:g} m".format(CROWD_M), ["0 or 1", "2", "3+"], 0),
    ("player speed 1-2 s earlier", _labels(FIT_SPEED, "m/s") + ["unknown"], 1),
)
INTERACTION = "nearest < 2 m x ball unseen"
# Dimensions kept in the cells but left out of the model: observed over expected ends per level
# checks whether the factors above already explain them.
CHECKS = (
    ("ball within {:g} m (not in the model)".format(CROSS_BALL),
     ["yes", "no or unseen"]),
)
DIMS = tuple(len(levels) for _, levels, _ in FACTORS) + tuple(len(l) for _, l in CHECKS)
RIDGE = 0.01                         # keeps a level with no ends finite; tiny next to the data


class Tally:
    """Per feature and bin: at-risk player-frames and track ends, pooled over matches."""

    def __init__(self, fps):
        self.fps = fps
        self.frames = {name: np.zeros(len(bins), dtype=np.int64) for name, bins in FEATURES.items()}
        self.ends = {name: np.zeros(len(bins), dtype=np.int64) for name, bins in FEATURES.items()}
        self.cell_frames = np.zeros(int(np.prod(DIMS)), dtype=np.int64)
        self.cell_ends = np.zeros(int(np.prod(DIMS)), dtype=np.int64)

    def add(self, name, bins, ended):
        size = len(FEATURES[name])
        self.frames[name] += np.bincount(bins, minlength=size)
        self.ends[name] += np.bincount(bins, weights=ended, minlength=size).astype(np.int64)

    def add_cells(self, cells, ended):
        size = len(self.cell_frames)
        self.cell_frames += np.bincount(cells, minlength=size)
        self.cell_ends += np.bincount(cells, weights=ended, minlength=size).astype(np.int64)

    def total(self):
        name = next(iter(FEATURES))
        return int(self.frames[name].sum()), int(self.ends[name].sum())

    def rate(self, frames, ends):
        return ends / frames * self.fps * 60.0 if frames else np.nan


def _nanmin(dist, mask):
    """Row minimum of dist over the slots mask selects; NaN where none is tracked."""
    picked = np.where(mask[None], dist, np.inf)
    low = picked.min(axis=2)
    return np.where(np.isinf(low), np.nan, low)


def _held(match, start, stop, threshold):
    """(T, S) frames in a row, up to each frame, that each slot moved under `threshold` metres."""
    x = match.player_x[start:stop].astype(np.float64)
    y = match.player_y[start:stop].astype(np.float64)
    step = np.full(x.shape, np.inf)
    step[1:] = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    still = step < threshold                           # NaN and the first frame are not still
    index = np.arange(len(x))[:, None]
    reset = np.maximum.accumulate(np.where(still, -1, index), axis=0)
    return index - reset


def tally_match(match, tally):
    """Add every at-risk player-frame of `match` (inside its halves) to `tally`."""
    length, width = match.pitch
    team = np.asarray(match.team)
    slots = len(team)
    not_self = ~np.eye(slots, dtype=bool)
    same = (team[:, None] == team[None, :]) & not_self
    other = team[:, None] != team[None, :]
    for start, stop in match.halves:
        holds = [_held(match, start, stop, m) for m in HOLD_M.values()]
        for begin in range(start, stop - 1, CHUNK):
            end = min(begin + CHUNK, stop - 1)          # t runs to stop - 2; t + 1 stays inside
            t = np.arange(begin, end)
            px, py = match.player_x[t].astype(np.float64), match.player_y[t].astype(np.float64)
            here = np.isfinite(px) & np.isfinite(py)
            nxt = np.isfinite(match.player_x[t + 1]) & np.isfinite(match.player_y[t + 1])
            at_risk = here
            ended = (here & ~nxt)[at_risk].astype(np.float64)

            # Untracked slots give NaN distances; NaN compares False, so they never count.
            dist = np.hypot(px[:, :, None] - px[:, None, :], py[:, :, None] - py[:, None, :])
            dist = np.where(np.isnan(dist), np.inf, dist)
            near_any = _nanmin(dist, not_self)
            near_team = _nanmin(dist, same)
            near_opp = _nanmin(dist, other)
            crowd = np.sum(np.where(not_self[None], dist, np.inf) < CROWD_M, axis=2)

            bx = match.ball_x[t].astype(np.float64)[:, None]
            by = match.ball_y[t].astype(np.float64)[:, None]
            ball = np.hypot(px - bx, py - by)
            touch = np.minimum(py, width - py)
            goal = np.minimum(px, length - px)

            back = t - SPEED_LAG
            ok = back >= start
            speed = np.full_like(px, np.nan)
            if ok.any():
                bpx = match.player_x[back[ok]].astype(np.float64)
                bpy = match.player_y[back[ok]].astype(np.float64)
                speed[ok] = np.hypot(px[ok] - bpx, py[ok] - bpy) * match.fps / SPEED_LAG

            lag = int(round(EARLY_S * match.fps))
            first, second = t - 2 * lag, t - lag
            ok = first >= start
            early = np.full_like(px, np.nan)
            if ok.any():
                ax = match.player_x[first[ok]].astype(np.float64)
                ay = match.player_y[first[ok]].astype(np.float64)
                cx = match.player_x[second[ok]].astype(np.float64)
                cy = match.player_y[second[ok]].astype(np.float64)
                early[ok] = np.hypot(cx - ax, cy - ay) / EARLY_S

            names = list(FEATURES)
            values = (near_any, near_team, near_opp, None, ball, touch, goal, speed, None)
            edges = (NEAR_EDGES, NEAR_EDGES, NEAR_EDGES, None, BALL_EDGES, LINE_EDGES,
                     LINE_EDGES, SPEED_EDGES, None)
            for name, value, edge in zip(names, values, edges):
                if value is not None:
                    tally.add(name, _bin(value[at_risk], edge), ended)
            tally.add(names[3], np.minimum(crowd[at_risk], CROWD_MAX), ended)

            near_bin = _bin(near_any[at_risk], CROSS_NEAR)
            ball_r = ball[at_risk]
            ball_bin = np.where(np.isnan(ball_r), 2, np.where(ball_r <= CROSS_BALL, 0, 1))
            tally.add(names[8], near_bin * 3 + ball_bin, ended)
            tally.add(names[9], _bin(early[at_risk], SPEED_EDGES), ended)
            for name, held in zip(HOLD_FEATURES, holds):
                tally.add(name, _bin(held[t - start][at_risk].astype(np.float64), HOLD_EDGES),
                          ended)

            levels = (
                np.minimum(_bin(near_any[at_risk], FIT_NEAR), 4),
                (near_opp[at_risk] < OPP_M).astype(np.int64),
                _bin(touch[at_risk], FIT_TOUCH),
                _bin(goal[at_risk], FIT_GOAL),
                np.isnan(ball_r).astype(np.int64),
                _bin(speed[at_risk], FIT_SPEED),
                np.clip(crowd[at_risk] - 1, 0, 2),
                _bin(early[at_risk], FIT_SPEED),
                (~(ball_r <= CROSS_BALL)).astype(np.int64),
            )
            tally.add_cells(np.ravel_multi_index(levels, DIMS), ended)


def report(sources, out=sys.stdout):
    """Side-by-side table per feature: share of player-frames, rate and lift for each source."""
    names = [n for n, _ in sources]
    for name, tally in sources:
        frames, ends = tally.total()
        out.write("{:<14} {:>12,} player-frames, {:>7,} track ends, {:.3f} per player-minute\n"
                  .format(name, frames, ends, tally.rate(frames, ends)))
    out.write("\ncolumns per source: share of at-risk player-frames | ends per player-minute | "
              "lift over that source's overall rate ('*': under {} ends)\n".format(MIN_ENDS))
    for feature, bins in FEATURES.items():
        out.write("\n{}\n".format(feature))
        out.write("  {:<26}".format("") + "".join(" | {:^24}".format(n) for n in names) + "\n")
        for i, label in enumerate(bins):
            cells = []
            empty = True
            for _, tally in sources:
                frames, ends = tally.total()
                f, e = tally.frames[feature][i], tally.ends[feature][i]
                if f:
                    empty = False
                rate = tally.rate(f, e)
                lift = rate / tally.rate(frames, ends) if f and ends else np.nan
                cells.append(" | {:>6.1%} {:>7.3f} {:>6.2f}{}".format(
                    f / frames if frames else 0.0, rate if f else np.nan, lift,
                    "*" if e < MIN_ENDS else " "))
            if not empty:
                out.write("  {:<26}".format(label) + "".join(cells) + "\n")

    for feature in HOLD_FEATURES:
        out.write("\n{}: share of track ends vs share of all player-frames\n".format(feature))
        out.write("  {:<26}".format("") + "".join(" | {:^24}".format(n) for n in names) + "\n")
        for i, label in enumerate(FEATURES[feature][:-1]):
            cells = []
            for _, tally in sources:
                frames, ends = tally.total()
                cells.append(" | ends {:>6.1%}  frames {:>5.1%}".format(
                    tally.ends[feature][i] / ends if ends else np.nan,
                    tally.frames[feature][i] / frames if frames else np.nan))
            out.write("  {:<26}".format(label) + "".join(cells) + "\n")


def design(dims=DIMS):
    """(X, names): one row per cell, intercept + non-reference factor levels + the interaction."""
    grid = np.indices(dims).reshape(len(dims), -1)
    columns, names = [np.ones(grid.shape[1])], ["intercept"]
    for k, (factor, levels, ref) in enumerate(FACTORS):
        for i, level in enumerate(levels):
            if i != ref:
                columns.append((grid[k] == i).astype(np.float64))
                names.append((factor, level))
    near_close = grid[0] <= 1                          # 0-1 m and 1-2 m
    columns.append((near_close & (grid[4] == 1)).astype(np.float64))
    names.append((INTERACTION, "yes"))
    return np.stack(columns, axis=1), names, grid


def fit(tally, iterations=50):
    """Poisson fit of ends on cells with log(player-frames) offset: (coef, se, names, grid, mu)."""
    X, names, grid = design()
    keep = tally.cell_frames > 0
    Xk, y = X[keep], tally.cell_ends[keep].astype(np.float64)
    offset = np.log(tally.cell_frames[keep].astype(np.float64))
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.sum(), 0.5) / tally.cell_frames.sum())
    penalty = np.full(X.shape[1], RIDGE)
    penalty[0] = 0.0
    for _ in range(iterations):
        mu = np.exp(Xk @ beta + offset)
        grad = Xk.T @ (y - mu) - penalty * beta
        hess = (Xk * mu[:, None]).T @ Xk + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = np.exp(Xk @ beta + offset)
    hess = (Xk * mu[:, None]).T @ Xk + np.diag(penalty)
    se = np.sqrt(np.diag(np.linalg.inv(hess)))
    expected = np.zeros(len(keep))
    expected[keep] = mu
    return beta, se, names, grid, expected


def report_fit(sources, out=sys.stdout):
    """Rate ratio and 95% interval per factor level, side by side; then the unmodelled checks."""
    fits = [(name, tally, fit(tally)) for name, tally in sources]
    _, _, columns, grid, _ = fits[0][2]
    out.write("\n=== --fit: Poisson model of the per-frame loss rate (main effects + {})\n"
              .format(INTERACTION))
    out.write("rate ratio against the reference level, 95% interval; '*': under {} ends in the "
              "level. Intervals treat player-frames as independent, so they are too narrow.\n"
              .format(MIN_ENDS))
    for name, tally, (beta, _, _, _, _) in fits:
        out.write("{:<14} base rate {:.3f} ends per player-minute at the reference levels\n"
                  .format(name, np.exp(beta[0]) * tally.fps * 60.0))
    header = "  {:<26}".format("") + "".join(" | {:^24}".format(n) for n, _, _ in fits) + "\n"
    current = None
    for j, column in enumerate(columns):
        if column == "intercept":
            continue
        factor, level = column
        if factor != current:
            current = factor
            ref = next((levels[r] for f, levels, r in FACTORS if f == factor), "no")
            out.write("\n{}  (reference: {})\n".format(factor, ref))
            out.write(header)
        cells = []
        for _, tally, (beta, se, _, g, _) in fits:
            in_level = _level_mask(column, g)
            ends = tally.cell_ends[in_level].sum()
            low, high = np.exp(beta[j] - 1.96 * se[j]), np.exp(beta[j] + 1.96 * se[j])
            cells.append(" | {:>6.2f} ({:>5.2f}-{:>6.2f}){}".format(
                np.exp(beta[j]), low, high, "*" if ends < MIN_ENDS else " "))
        out.write("  {:<26}".format(level) + "".join(cells) + "\n")

    out.write("\nobserved / expected ends on dimensions left out of the model "
              "(near 1: already explained)\n")
    offset = len(FACTORS)
    for k, (check, levels) in enumerate(CHECKS):
        out.write("\n{}\n".format(check))
        out.write(header)
        for i, level in enumerate(levels):
            cells = []
            for _, tally, (_, _, _, g, expected) in fits:
                mask = g[offset + k] == i
                obs, exp = tally.cell_ends[mask].sum(), expected[mask].sum()
                cells.append(" | {:>7} / {:>8.1f} = {:>5.2f}".format(
                    int(obs), exp, obs / exp if exp else np.nan))
            out.write("  {:<26}".format(level) + "".join(cells) + "\n")

    out.write("\nlog rate ratios, for simulator v2 (copy as is):\n")
    for name, _, (beta, _, _, _, _) in fits:
        values = {"intercept": round(float(beta[0]), 4)}
        for column, b in zip(columns[1:], beta[1:]):
            values["{} = {}".format(*column)] = round(float(b), 4)
        out.write("{}: {}\n".format(name, values))


def _level_mask(column, grid):
    factor, level = column
    if factor == INTERACTION:
        return (grid[0] <= 1) & (grid[4] == 1)
    k = next(i for i, (f, _, _) in enumerate(FACTORS) if f == factor)
    return grid[k] == FACTORS[k][1].index(level)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar-ath", action="store_true", help="add BAR-ATH (needs data/)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--simulator", default=None, help="simulator version (default: v1)")
    parser.add_argument("--matches", nargs="+", default=None, help="IDSSE ids (default: dev)")
    parser.add_argument("--no-clean", action="store_true", help="skip clean IDSSE")
    parser.add_argument("--fit", action="store_true",
                        help="also fit a Poisson model of the loss rate (rate ratios per factor)")
    args = parser.parse_args()

    from solution import simulate
    from solution.sources import idsse
    version = args.simulator or simulate.DEFAULT_VERSION
    matches = args.matches or list(idsse.DEV_IDS)
    began = time.time()
    clean, sim, sources = None, None, []
    for match_id in matches:
        raw = idsse.load(match_id)
        if clean is None:
            clean, sim = Tally(raw.fps), Tally(raw.fps)
        if not args.no_clean:
            tally_match(raw, clean)
        for seed in args.seeds:
            tally_match(simulate.degrade(raw, seed, version), sim)
    if not args.no_clean:
        sources.append(("IDSSE clean", clean))
    sources.append(("IDSSE sim " + version, sim))
    if args.bar_ath:
        from solution.sources import atd
        match = atd.load()
        bar = Tally(match.fps)
        tally_match(match, bar)
        sources.append(("BAR-ATH", bar))
    print("track_loss_check: {} IDSSE matches x {} seeds{}, {:.0f} s\n".format(
        len(matches), len(args.seeds), " + BAR-ATH" if args.bar_ath else "",
        time.time() - began))
    report(sources)
    if args.fit:
        # Clean IDSSE has under 50 ends: nothing to fit.
        report_fit([(n, t) for n, t in sources if n != "IDSSE clean"])


if __name__ == "__main__":
    main()
