"""Calibrate simulator v2's player losses so its track-loss fit reproduces BAR-ATH's.

    .venv/bin/python tools/track_loss_calibrate.py              # dev matches, seed 0, 8 rounds
    .venv/bin/python tools/track_loss_calibrate.py --rounds 3 --check   # just compare V2 as is

Indirect inference: run v2 on the IDSSE dev matches, fit tools/track_loss_check.py's model on
its output, and move each loss term of simulate_v2.V2["loss"] by the gap between BAR-ATH's fitted
log rate ratio (simulate.BAR_ATH["track_loss"]["fit"]) and v2's. The loss terms are not the
fitted ratios themselves, because v2's episodes and ghosts add losses that are not tied to
position, and its slow-downs move the speed terms. The intercept follows the overall rate
(0.552 ends per player-minute) and the slow-down share follows the share of ends after a
nearly-still frame. Speed and opponent terms are not parameters of v2: they are the check.

No data/ needed: the targets are committed aggregates. Prints the calibrated V2["loss"] to paste
into solution/simulate_v2.py, and every fitted term next to its target.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import simulate, simulate_v2  # noqa: E402
from solution.sources import idsse  # noqa: E402
from tools import track_loss_check as tlc  # noqa: E402

TARGET = simulate.BAR_ATH["track_loss"]


def measure(inputs, params, seeds):
    """(fitted log ratios by key, rate per player-minute, share of ends after a still frame)."""
    tally = None
    for raw, balls in inputs:
        for seed, ball in zip(seeds, balls):
            # The degraded ball, so the fit sees the gaps the losses were drawn with.
            out = ball.copy()
            out.player_x, out.player_y, out.team = simulate_v2.degrade_players_v2(
                raw, ball.ball_seen, simulate._rng(raw, seed, 2), params)
            if tally is None:
                tally = tlc.Tally(raw.fps)
            tlc.tally_match(out, tally)
    beta, _, names, _, _ = tlc.fit(tally)
    fitted = {"intercept": float(beta[0])}
    for column, b in zip(names[1:], beta[1:]):
        fitted["{} = {}".format(*column)] = float(b)
    frames, ends = tally.total()
    still = tally.ends[tlc.HOLD_FEATURES[1]]
    return fitted, tally.rate(frames, ends), 1.0 - still[0] / ends


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--step", type=float, default=0.7)
    parser.add_argument("--check", action="store_true", help="measure V2 as is, change nothing")
    args = parser.parse_args()

    began = time.time()
    inputs = []
    for match_id in idsse.DEV_IDS:
        raw = idsse.load(match_id)
        balls = [simulate.degrade_v1(raw, seed, players=False) for seed in args.seeds]
        inputs.append((raw, balls))
    params = {k: (dict(v) if isinstance(v, dict) else v) for k, v in simulate_v2.V2.items()}
    rounds = 1 if args.check else args.rounds
    for r in range(rounds):
        fitted, rate, after_still = measure(inputs, params, args.seeds)
        terms = [k for k in params["loss"] if k != "intercept"]
        gap = max(abs(TARGET["fit"][k] - fitted[k]) for k in terms)
        print("round {}: rate {:.3f} (target {:.3f}), ends after still {:.3f} ({:.3f}), "
              "largest term gap {:.3f}, slow share {:.3f}, {:.0f} s".format(
                  r + 1, rate, TARGET["rate_per_player_minute"], after_still,
                  TARGET["ends_after_still"], gap, params["slow_share"], time.time() - began))
        if args.check or r == rounds - 1:
            break
        for k in terms:
            params["loss"][k] += args.step * (TARGET["fit"][k] - fitted[k])
        params["loss"]["intercept"] += np.log(TARGET["rate_per_player_minute"] / rate)
        params["slow_share"] = float(np.clip(
            params["slow_share"] + (TARGET["ends_after_still"] - after_still), 0.0, 1.0))

    print("\nfitted on v2 vs BAR-ATH (log rate ratios; speed and opponent terms are the check)")
    for k, target in TARGET["fit"].items():
        mark = "" if k in params["loss"] else "   (not a v2 parameter)"
        print("  {:<42} v2 {:>8.3f}   BAR-ATH {:>8.3f}{}".format(k, fitted[k], target, mark))
    print("\nV2[\"loss\"] = {")
    for k, v in params["loss"].items():
        print("    \"{}\": {:.4f},".format(k, v))
    print("}}\nV2[\"slow_share\"] = {:.3f}".format(params["slow_share"]))


if __name__ == "__main__":
    main()
