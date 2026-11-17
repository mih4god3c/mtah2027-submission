"""Does player speed look the same to the decoder on BAR-ATH as on the proxy league? Aggregates only.

    .venv/bin/python tools/speed_check.py                        # IDSSE dev, clean and degraded v1
    .venv/bin/python tools/speed_check.py --position-noise 0.1   # ... with the jitter stress test
    .venv/bin/python tools/speed_check.py --bar-ath              # (needs data/) BAR-ATH, human run

Why: the semi_markov decoder (D8) gets nearly all of its gain from collective player speed,
computed from positions (simulate.mean_player_speed), with likelihoods fitted on clean IDSSE.
The EDA's BAR-ATH speeds (1.17 / 2.16 m/s) came from the ATD's delivered speed field, so the
number the decoder actually sees on BAR-ATH has never been measured. If tracking noise inflates
it, the decoder will miss stoppages there.

Each block of output is one decoder variant (VARIANTS: the registered one and the two speed
fixes from the stress test). Everything printed is an aggregate over the graded frames: speed
deciles in m/s at that variant's lag, the mean speed with the ball seen and missing, the share of 5 Hz blocks whose speed alone favours a dead ball, and
what the decoder outputs (dead-ball share, possessions per 90 minutes, median durations). No
frame-level value is printed, so the BAR-ATH output is safe to paste back.
"""

import argparse
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import semi_markov  # noqa: E402
from solution.simulate import add_position_noise, degrade  # noqa: E402

DECILES = (10, 20, 30, 40, 50, 60, 70, 80, 90)
# The registered decoder and the two speed fixes tried in the stress test (EXPERIMENTS
# section 2). One run reports all three, so the BAR-ATH run is needed only once.
VARIANTS = {
    "speed_ms": {"speed_scale": "ms"},
    "speed_lag12": {"speed_scale": "ms", "speed_lag": 12},
    "speed_rank": {"speed_scale": "rank"},       # the registered default (D9)
}


def block_speed(speed, halves):
    """Mean speed per 5 Hz decoder block, inside the halves."""
    out = []
    for start, stop in halves:
        part = speed[start:stop]
        n_blocks = -(-len(part) // decoder.BLOCK)
        index = np.arange(len(part)) // decoder.BLOCK
        finite = np.isfinite(part)
        count = np.bincount(index, weights=finite.astype(float), minlength=n_blocks)
        total = np.bincount(index, weights=np.where(finite, part, 0.0), minlength=n_blocks)
        with np.errstate(invalid="ignore"):
            out.append(total / count)
    return np.concatenate(out)


def spell_seconds(codes, fps):
    lengths = decoder.spell_lengths(codes)
    return {key: np.asarray(values, dtype=float) / fps for key, values in lengths.items()}


def stats(match, settings, predict=True):
    """Aggregates of one match as a decoder variant sees it."""
    settings = dict(decoder.DEFAULTS, **settings)
    lag, scale = settings["speed_lag"], settings["speed_scale"]
    graded = match.in_play_mask()
    speed = decoder.speed_feature(match, lag, "ms")
    seen = match.ball_seen
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row = {
            "deciles": np.nanpercentile(speed[graded], DECILES),
            "seen": np.nanmean(speed[graded & seen]),
            "missing": np.nanmean(speed[graded & ~seen]),
            "no_speed": float(np.mean(~np.isfinite(speed[graded]))),
        }
    priors = decoder.idsse_priors(match.name, lag, scale)
    blocks = block_speed(decoder.speed_feature(match, lag, scale), match.halves)
    known = np.isfinite(blocks)
    log_ratio = priors["speed_log"][1] - priors["speed_log"][0]
    row["speed_says_dead"] = float(np.mean(
        log_ratio[decoder.speed_bins(blocks[known], priors["speed_edges"])] > 0))
    minutes = graded.sum() / match.fps / 60.0
    if predict:
        codes = semi_markov(**settings)(match)
        spells = spell_seconds(codes, match.fps)
        row["pred_dead"] = float(np.mean(codes[graded] == pm.D))
        row["pred_per90"] = len(spells["possession"]) * 90.0 / minutes
        row["pred_poss_s"] = float(np.median(spells["possession"]))
        row["pred_dead_s"] = float(np.median(spells["dead"])) if len(spells["dead"]) else 0.0
    if match.labels is not None:
        spells = spell_seconds(match.labels, match.fps)
        row["true_dead"] = float(np.mean(match.labels[graded] == pm.D))
        row["true_per90"] = len(spells["possession"]) * 90.0 / minutes
        row["true_poss_s"] = float(np.median(spells["possession"]))
        row["true_dead_s"] = float(np.median(spells["dead"]))
    return row


def report(title, rows):
    """Mean over rows (matches x seeds), printed in one block."""
    print("\n" + title + "  ({} match-seed{})".format(len(rows), "" if len(rows) == 1 else "s"))
    deciles = np.mean([r["deciles"] for r in rows], axis=0)
    print("  mean player speed deciles (m/s), p10..p90: " +
          " ".join("{:.2f}".format(v) for v in deciles))
    seen, missing = np.mean([r["seen"] for r in rows]), np.mean([r["missing"] for r in rows])
    print("  ball seen {:.2f} m/s, ball missing {:.2f} m/s, ratio {:.3f};"
          " frames with no speed {:.3f}".format(seen, missing, missing / seen,
                                                np.mean([r["no_speed"] for r in rows])))
    print("  5 Hz blocks whose speed alone favours dead ball: {:.3f}".format(
        np.mean([r["speed_says_dead"] for r in rows])))
    for prefix, label in (("pred", "decoded"), ("true", "truth")):
        if prefix + "_dead" in rows[0]:
            print("  {:<12} dead share {:.3f}, possessions per 90 min {:.0f}, median possession "
                  "{:.1f} s, median dead ball {:.1f} s".format(
                      label, *[np.mean([r[prefix + key] for r in rows])
                               for key in ("_dead", "_per90", "_poss_s", "_dead_s")]))


def report_all(title, matches, predict=True):
    for name, settings in VARIANTS.items():
        report("{} -- {} {}".format(title, name, settings or ""),
               [stats(m, settings, predict) for m in matches])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar-ath", action="store_true",
                        help="the competition match (needs data/; for a human to run)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--position-noise", type=float, nargs="+", default=[],
                        metavar="SD", help="also report IDSSE with these noise levels (m)")
    args = parser.parse_args()

    if args.bar_ath:
        from solution.sources import atd
        report_all("BAR-ATH ATD (priors fitted on all five IDSSE dev matches)", [atd.load()])
        print("\nCompare with: python tools/speed_check.py (IDSSE dev, clean and degraded v1).")
        return

    from solution.sources import idsse
    matches = [idsse.load(m) for m in idsse.DEV_IDS]
    report_all("IDSSE dev, clean", matches, predict=False)
    report_all("IDSSE dev, degraded v1", [degrade(m, s) for m in matches for s in args.seeds])
    for sd in args.position_noise:
        report_all("IDSSE dev, degraded v1 + position noise sd {} m".format(sd),
                   [add_position_noise(degrade(m, s), sd, s)
                    for m in matches for s in args.seeds])


if __name__ == "__main__":
    main()
