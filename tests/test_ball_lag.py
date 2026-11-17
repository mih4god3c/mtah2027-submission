"""tools/ball_lag_check.py: a known lag is recovered; centred and past-only blur differ in shape.
No data needed."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import Match  # noqa: E402
from solution.simulate import add_ball_noise  # noqa: E402
from tools.ball_lag_check import kick_summary, lag_summary, measure  # noqa: E402

FPS, PITCH = 25, (105.0, 68.0)


def dribble_match(lag_frames, n=3000):
    """One player running a circle at 4 m/s with the ball, the ball `lag_frames` behind."""
    t = np.arange(n + lag_frames)
    angle = 4.0 / 15.0 * t / FPS
    px, py = 52.5 + 15.0 * np.cos(angle), 34.0 + 15.0 * np.sin(angle)
    player_x = np.column_stack((px[lag_frames:], np.full(n, 5.0))).astype(np.float32)
    player_y = np.column_stack((py[lag_frames:], np.full(n, 5.0))).astype(np.float32)
    ball_x = (px[:n] + 0.3).astype(np.float32)        # the ball at t shows the player at t - lag
    ball_y = py[:n].astype(np.float32)
    return Match("dribble", FPS, PITCH, player_x, player_y, np.array([0, 1], np.int8),
                 ball_x, ball_y, [(0, n)])


def kick_match(n=4000):
    """Still for 2.5 s, kicked at 15 m/s for 1.5 s, still again; back and forth along x."""
    x = np.empty(n)
    position, period = 40.0, 100
    for t in range(n):
        phase = t % period
        direction = 1.0 if (t // period) % 2 == 0 else -1.0
        if 62 <= phase < 100:
            position += direction * 15.0 / FPS
        x[t] = position
    player_x = np.full((n, 2), 10.0, np.float32)
    player_y = np.full((n, 2), 10.0, np.float32)
    return Match("kicks", FPS, PITCH, player_x, player_y, np.array([0, 1], np.int8),
                 x.astype(np.float32), np.full(n, 34.0, np.float32), [(0, n)])


class BallLagTest(unittest.TestCase):
    def test_known_lag_is_recovered(self):
        for lag in (0, 6):
            _, _, _, refined = lag_summary([measure(dribble_match(lag))])
            self.assertAlmostEqual(refined, lag / float(FPS), delta=0.03)

    def test_centred_and_past_only_blur_differ(self):
        match = kick_match()
        sharp = kick_summary([measure(match)])
        centred = kick_summary([measure(add_ball_noise(match, 0.0, smooth_s=0.26))])
        causal = kick_summary([measure(add_ball_noise(match, 0.0, smooth_s=0.26, causal=True))])
        self.assertGreater(min(sharp[0], centred[0], causal[0]), 10)
        self.assertLess(sharp[1], 0.1)                  # a clean kick rises within frames
        self.assertGreater(centred[1], 0.4)             # blurred: tenths of a second
        self.assertLess(centred[4], 1.2)                # centred: symmetric
        self.assertGreater(causal[4], 1.8)              # past-only: jump, then creep

    def test_causal_option_is_off_by_default(self):
        match = kick_match(600)
        a = add_ball_noise(match, 0.05, 3.4, 0, smooth_s=0.26)
        b = add_ball_noise(match, 0.05, 3.4, 0, smooth_s=0.26, causal=False)
        self.assertTrue(np.array_equal(a.ball_x, b.ball_x, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
