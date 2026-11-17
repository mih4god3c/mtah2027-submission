"""Tests for the self-calibrating whistle detector (solution/whistle.py, D47) on synthetic audio.

The referee here is made up -- a two-tone whistle at 3150 + 3620 Hz, or a pea whistle at 2850 Hz
-- so nothing in these tests describes the competition match. Kickoff anchors are given as
frames, as kickoff_anchors() would return them from the tracking.
No data needed: .venv/bin/python -m unittest tests.test_whistle_selfcal -v
"""

import functools
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from solution import whistle  # noqa: E402
from test_whistle import crowd  # noqa: E402

FPS = 25
RATE = whistle.RATE
SECONDS = 600.0
N = int(SECONDS * FPS)
KICKS_S = (150.0, 300.0, 450.0)             # kickoffs; the referee whistles 2 s before each
LEAD_S = 2.0
OTHER_S = [(20.0, 0.4), (65.0, 1.0), (90.0, 0.3), (200.0, 0.6), (250.0, 0.5), (340.0, 0.8),
           (400.0, 0.4), (500.0, 0.7), (560.0, 0.3)]   # the referee's other whistles (start, s)
TWO_TONE = (3150.0, 3620.0)
PEA = 2850.0
# fans: one tone sliding (start s, from Hz, to Hz), some of them right before a kickoff
FANS = [(8.0, 3600.0, 3200.0), (37.0, 3300.0, 3000.0), (147.0, 3200.0, 3500.0),
        (230.0, 2600.0, 2900.0), (296.5, 4400.0, 4100.0), (380.0, 3500.0, 3900.0),
        (446.0, 3000.0, 3400.0), (520.0, 3100.0, 3700.0)]   # some right before a kickoff
HORN = [(110.0, 2.0), (270.0, 2.0), (420.0, 2.0)]   # a steady stadium horn, away from kickoffs
HORN_HZ = 2600.0


def tone(start, length, hz_from, hz_to=None, amplitude=0.03, vibrato_hz=20.0):
    """A tone gliding linearly from hz_from to hz_to, with a small 5 Hz vibrato and 20 ms ramps."""
    n = int(SECONDS * RATE)
    t = np.arange(n) / float(RATE)
    on = (t >= start) & (t < start + length)
    hz_to = hz_from if hz_to is None else hz_to
    hz = hz_from + (hz_to - hz_from) * np.clip((t - start) / length, 0, 1)
    hz = hz + vibrato_hz * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(hz) / RATE
    ramp = np.clip(np.minimum(t - start, start + length - t) / 0.02, 0, 1) * on
    return amplitude * ramp * np.sin(phase)


def blasts(at_kicks=True, others=True):
    out = [(k - LEAD_S, 0.8) for k in KICKS_S] if at_kicks else []
    return sorted(out + (OTHER_S if others else []))


def referee_audio(kind, items):
    if kind == "two_tone":
        return sum(tone(s, l, TWO_TONE[0]) + tone(s, l, TWO_TONE[1]) for s, l in items)
    return sum(tone(s, l, PEA, amplitude=0.05) for s, l in items)


@functools.lru_cache(maxsize=None)
def features(kind="two_tone", at_kicks=True, seed=0):
    rng = np.random.RandomState(seed)
    audio = crowd(rng, SECONDS)
    audio = audio + referee_audio(kind, blasts(at_kicks))
    audio = audio + sum(tone(s, 1.0, a, b, amplitude=0.05) for s, a, b in FANS)
    audio = audio + sum(tone(s, l, HORN_HZ, amplitude=0.05, vibrato_hz=0.0) for s, l in HORN)
    return whistle.peak_features(iter([audio.astype(np.float32)]), N, FPS)


def anchors(kicks=KICKS_S):
    return [int(k * FPS) for k in kicks]


GRADED = np.ones(N, dtype=bool)


class TestCalibration(unittest.TestCase):
    def test_finds_the_two_tone_whistle_from_the_kickoffs(self):
        for seed in range(2):
            cal = whistle.calibrate(features("two_tone", seed=seed), anchors(), GRADED, FPS)
            self.assertIsNotNone(cal, seed)
            self.assertEqual(cal["kind"], "pair", seed)
            for got, want in zip(cal["tones"], TWO_TONE):
                self.assertLess(abs(got - want), 40.0, (seed, cal))
            self.assertEqual(cal["anchors_hit"], len(KICKS_S), seed)

    def test_finds_a_single_tone_pea_whistle(self):
        cal = whistle.calibrate(features("pea"), anchors(), GRADED, FPS)
        self.assertIsNotNone(cal)
        self.assertEqual(cal["kind"], "single", cal)
        self.assertLess(abs(cal["tones"][0] - PEA), 40.0, cal)

    def test_no_whistle_at_the_kickoffs_gives_no_calibration(self):
        # the referee whistles elsewhere; around the kickoffs only fans: nothing is guessed
        self.assertIsNone(whistle.calibrate(features("two_tone", at_kicks=False), anchors(),
                                            GRADED, FPS))

    def test_too_few_anchors_gives_no_calibration(self):
        self.assertIsNone(whistle.calibrate(features(), anchors(KICKS_S[:1]), GRADED, FPS))

    def test_leave_one_anchor_out_is_stable(self):
        full = whistle.calibrate(features(), anchors(), GRADED, FPS)
        for k in range(len(KICKS_S)):
            kicks = KICKS_S[:k] + KICKS_S[k + 1:]
            cal = whistle.calibrate(features(), anchors(kicks), GRADED, FPS)
            self.assertIsNotNone(cal, k)
            for got, want in zip(cal["tones"], full["tones"]):
                self.assertLess(abs(got - want), whistle.CAL_BIN_HZ, k)


class TestDetection(unittest.TestCase):
    def check_found(self, kind, seed):
        feats = features(kind, seed=seed)
        cal = whistle.calibrate(feats, anchors(), GRADED, FPS)
        found = [e for e in whistle.whistle_events(feats, cal, FPS) if e["whistle"]]
        want = blasts()
        self.assertEqual(len(found), len(want), (kind, seed, [e["start"] for e in found]))
        for event, (start, length) in zip(found, want):
            self.assertLessEqual(abs(event["start"] - start * FPS), 2, (kind, seed))
            self.assertLessEqual(abs(event["seconds"] - length), 0.12, (kind, seed))

    def test_finds_every_blast_and_nothing_else(self):
        for seed in range(2):
            self.check_found("two_tone", seed)
        self.check_found("pea", 0)

    def test_no_calibration_no_whistles(self):
        self.assertEqual(whistle.whistle_events(features(), None, FPS), [])

    def test_block_sizes_do_not_change_the_features(self):
        rng = np.random.RandomState(0)
        audio = (crowd(rng, 30.0) + 0.0).astype(np.float32)
        n = int(30.0 * FPS)
        whole = whistle.peak_features(iter([audio]), n, FPS)
        split = whistle.peak_features(iter(np.array_split(audio, 977)), n, FPS)
        np.testing.assert_allclose(whole, split, rtol=0, atol=1e-3)


class TestCrossFeed(unittest.TestCase):
    """tools/whistle_selfcal_check.cross_counts on event lists: the same whistles heard by both
    feeds pair at the lag; shifted, only chance pairs remain."""

    def test_real_pairs_far_above_chance(self):
        from tools.whistle_selfcal_check import cross_counts
        rng = np.random.RandomState(0)
        fps, lag = 25, 5
        referee = np.sort(rng.choice(np.arange(1000, 100000), 60, replace=False))

        def event(start, hz):
            return {"start": int(start), "stop": int(start) + 15, "seconds": 0.6, "hz": hz}

        bro = [event(f, 3600.0 + rng.uniform(-40, 40)) for f in referee[:50]]
        tac = [event(f + lag + rng.randint(-2, 3), 3600.0 + rng.uniform(-40, 40))
               for f in referee[10:]]
        tac += [event(f, 3600.0) for f in rng.randint(1000, 100000, 15)]   # tactical-only
        tac.sort(key=lambda e: e["start"])
        got_lag, n_lag, pairs, chance = cross_counts(tac, bro, fps)
        self.assertLessEqual(abs(got_lag - lag), 1)
        self.assertGreaterEqual(n_lag, whistle.MIN_LAG_PAIRS)
        self.assertGreaterEqual(len(pairs), 38)          # 40 shared whistles
        self.assertLessEqual(max(chance), 3)


if __name__ == "__main__":
    unittest.main()
