"""Tests for tools/whistle_veto_check.py's veto pieces, on hand-made inputs (no data needed)."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.whistle_veto_check import detector, stretch_margin, veto_mask  # noqa: E402


class FakeMatch:
    pitch = (105.0, 68.0)

    def __init__(self, x, y):
        self.ball_x, self.ball_y = np.asarray(x, float), np.asarray(y, float)

    @property
    def ball_seen(self):
        return np.isfinite(self.ball_x) & np.isfinite(self.ball_y)


class TestVeto(unittest.TestCase):
    def test_margin_uses_seen_ball_inside_then_last_seen_before(self):
        x = np.full(100, np.nan)
        y = np.full(100, np.nan)
        x[50:60], y[50:60] = 52.0, 34.0            # centre spot: 34 m from the touchlines
        x[10], y[10] = 3.0, 30.0                   # 3 m from a goal line
        m = FakeMatch(x, y)
        self.assertAlmostEqual(stretch_margin(m, 45, 70, 10), 34.0)
        self.assertAlmostEqual(stretch_margin(m, 15, 20, 10), 3.0)
        self.assertIsNone(stretch_margin(m, 30, 40, 10))   # nothing seen within the lookback

    def test_veto_needs_place_and_no_whistle(self):
        stretches = [(100, 150), (300, 350), (500, 550), (700, 750)]
        margins = [20.0, 20.0, 2.0, None]
        whistles = np.array([290])                 # 10 frames before the second stretch
        out = veto_mask(stretches, margins, whistles, 1000, 10.0, before=25)
        self.assertTrue(out[100:150].all())        # mid-pitch, no whistle: released
        self.assertFalse(out[300:350].any())       # whistled
        self.assertFalse(out[500:550].any())       # at a line
        self.assertFalse(out[700:750].any())       # place unknown

    def test_detector_misses_and_fans(self):
        rng = np.random.RandomState(0)
        whistles = np.arange(0, 10000, 100)
        in_play = np.ones(25 * 60 * 10, dtype=bool)
        self.assertEqual(len(detector(whistles, in_play, 25, 0.0, 0.0, rng)), len(whistles))
        kept = detector(whistles, in_play, 25, 0.5, 0.0, rng)
        self.assertTrue(30 < len(kept) < 70)
        with_fans = detector(whistles, in_play, 25, 0.0, 2.0, rng)
        self.assertTrue(len(with_fans) > len(whistles) + 5)


if __name__ == "__main__":
    unittest.main()
