"""tools/track_loss_check.py: track-end rates by feature (no data needed)."""

import io
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import Match  # noqa: E402
from tools import track_loss_check as tlc  # noqa: E402

FPS = 25


def occlusion_match(n=5000, seed=0):
    """Slot 1 (away) walks past slot 0 (home) again and again and is lost whenever within 1 m."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3), dtype=np.float32)
    y = np.zeros((n, 3), dtype=np.float32)
    x[:, 0], y[:, 0] = 50.0, 34.0
    x[:, 1] = 50.0 + 8.0 * np.sin(np.arange(n) / 40.0)
    y[:, 1] = 34.0 + 0.3
    x[:, 2], y[:, 2] = 20.0 + rng.normal(0, 0.05, n), 10.0      # far away, never lost
    close = np.hypot(x[:, 1] - x[:, 0], y[:, 1] - y[:, 0]) < 1.0
    # Lost on frame t + 1 when within 1 m on t, the last frame seen, where the tool reads.
    lost = np.zeros(n, dtype=bool)
    lost[1:] = close[:-1] & (np.arange(1, n) % 7 == 0)
    x[lost, 1] = np.nan
    y[lost, 1] = np.nan
    ball = np.full(n, np.nan, dtype=np.float32)
    return Match("synthetic_occlusion", FPS, (105.0, 68.0), x, y,
                 np.array([0, 1, 0], dtype=np.int8), ball.copy(), ball.copy(), [(0, n)])


class TrackLoss(unittest.TestCase):
    def test_counts_at_risk_frames_and_ends(self):
        match = occlusion_match()
        tally = tlc.Tally(FPS)
        tlc.tally_match(match, tally)
        frames, ends = tally.total()
        here = np.isfinite(match.player_x[:-1])
        nxt = np.isfinite(match.player_x[1:])
        self.assertEqual(frames, int(here.sum()))                # the last frame is never at risk
        self.assertEqual(ends, int((here & ~nxt).sum()))
        self.assertGreater(ends, 0)

    def test_losses_show_up_at_short_range_only(self):
        tally = tlc.Tally(FPS)
        tlc.tally_match(occlusion_match(), tally)
        name = "nearest, any team"
        ends = tally.ends[name]
        self.assertEqual(ends.sum(), ends[0])                     # every end is under 1 m
        self.assertEqual(tally.frames["distance to ball"][-1], tally.total()[0])   # ball unseen
        self.assertEqual(tally.ends["nearest teammate"][1:-1].sum(), 0)

    def test_halves_do_not_leak(self):
        match = occlusion_match(n=200)
        match.player_x[100:, 2] = np.nan                         # slot 2 ends at the half boundary
        match.halves = [(0, 100), (100, 200)]
        tally = tlc.Tally(FPS)
        tlc.tally_match(match, tally)
        lost_before = np.isfinite(match.player_x[:99]) & ~np.isfinite(match.player_x[1:100])
        lost_after = np.isfinite(match.player_x[100:199]) & ~np.isfinite(match.player_x[101:200])
        self.assertEqual(tally.total()[1], int(lost_before.sum() + lost_after.sum()))

    def test_held_track_is_slow_now_but_fast_earlier(self):
        n = 400
        x = np.zeros((n, 2), dtype=np.float32)
        y = np.full((n, 2), 34.0, dtype=np.float32)
        x[:, 0] = 10.0 + 5.0 * np.arange(n) / FPS                  # runs at 5 m/s
        x[300:, 0] = x[299, 0]                                     # held still for 6 frames
        x[306:, 0] = np.nan                                        # then dropped
        x[:, 1], y[:, 1] = 90.0, 10.0                              # far away, always tracked
        ball = np.full(n, np.nan, dtype=np.float32)
        match = Match("synthetic_hold", FPS, (105.0, 68.0), x, y.copy(),
                      np.array([0, 1], dtype=np.int8), ball.copy(), ball.copy(), [(0, n)])
        tally = tlc.Tally(FPS)
        tlc.tally_match(match, tally)
        self.assertEqual(tally.total()[1], 1)
        recent = tally.ends["player speed"]
        early = tally.ends["player speed 1-2 s earlier"]
        self.assertEqual(recent[0], 1)                             # 0-1 m/s on the last frame
        self.assertEqual(early[3], 1)                              # 5-7 m/s a second before
        frozen = tally.ends[tlc.HOLD_FEATURES[0]]
        self.assertEqual(frozen[3], 1)                             # held 5 frames: "5-9 fr"
        self.assertEqual(tally.ends[tlc.HOLD_FEATURES[1]][3], 1)
        self.assertEqual(tally.frames[tlc.HOLD_FEATURES[0]][0], tally.total()[0] - 5 - 399)

    def test_report_prints_every_source(self):
        tally = tlc.Tally(FPS)
        tlc.tally_match(occlusion_match(), tally)
        out = io.StringIO()
        tlc.report([("a", tally), ("b", tally)], out)
        text = out.getvalue()
        for name in tlc.FEATURES:
            self.assertIn(name, text)
        self.assertIn(" 1.00", text)


class Fit(unittest.TestCase):
    def test_recovers_known_rate_ratios(self):
        X, names, _ = tlc.design()
        rng = np.random.default_rng(0)
        beta = rng.normal(0, 0.5, X.shape[1])
        beta[0] = np.log(1e-3)
        tally = tlc.Tally(FPS)
        tally.cell_frames[:] = 10 ** 7                           # noise-free: ends = expectation
        tally.cell_ends[:] = np.round(np.exp(X @ beta) * 10 ** 7).astype(np.int64)
        fitted, se, _, _, expected = tlc.fit(tally)
        np.testing.assert_allclose(fitted, beta, atol=1e-3)
        self.assertTrue(np.all(se > 0))
        np.testing.assert_allclose(expected, tally.cell_ends, rtol=1e-3)

    def test_occlusion_shows_in_the_fit(self):
        tally = tlc.Tally(FPS)
        tlc.tally_match(occlusion_match(n=20000), tally)
        beta, _, names, _, _ = tlc.fit(tally)
        close = names.index(("nearest, any team", "0-1 m"))
        self.assertGreater(np.exp(beta[close]), 50.0)
        self.assertEqual(tally.cell_frames.sum(), tally.total()[0])
        self.assertEqual(tally.cell_ends.sum(), tally.total()[1])

    def test_report_fit_prints_factors_and_checks(self):
        tally = tlc.Tally(FPS)
        tlc.tally_match(occlusion_match(), tally)
        out = io.StringIO()
        tlc.report_fit([("a", tally)], out)
        text = out.getvalue()
        for factor, _, _ in tlc.FACTORS:
            self.assertIn(factor, text)
        for check, _ in tlc.CHECKS:
            self.assertIn(check, text)
        self.assertIn("log rate ratios", text)


if __name__ == "__main__":
    unittest.main()
