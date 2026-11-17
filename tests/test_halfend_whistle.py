"""Tests for D48's pieces on hand-made matches (no data needed): the walk-off stress
(simulate.add_walkoff), the whistle shift through it (whistlecue.walkoff_frame, half_ends) and
the end-of-half rule's tracking guard (whistlecue.end_picks, end_of_half).
    .venv/bin/python -m unittest tests.test_halfend_whistle -v"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import whistlecue as wc  # noqa: E402
from solution.match import Match, pm  # noqa: E402
from solution.simulate import add_walkoff, mean_player_speed  # noqa: E402

FPS = 5
HALF_S = 47 * 60                      # each half: 47 minutes
SLOW_S = 60                           # players slow to a walk in each half's last minute


def make_match(slow_end=True, seed=0):
    """Two halves of HALF_S: 22 players running (~3 m/s) and, with `slow_end`, walking (~0.5 m/s)
    in each half's last SLOW_S; the ball at a player's feet; labels H, then D at the slow end."""
    rng = np.random.RandomState(seed)
    n_half = HALF_S * FPS
    n = 2 * n_half
    t = np.arange(n) / float(FPS)
    base_x = rng.uniform(20, 85, 22)
    base_y = rng.uniform(10, 58, 22)
    phase = rng.uniform(0, 2 * np.pi, 22)
    amp = np.full(n, 8.0)
    labels = np.full(n, pm.H, dtype=np.int8)
    if slow_end:
        for k in (1, 2):
            tail = slice(k * n_half - SLOW_S * FPS, k * n_half)
            amp[tail] = 1.3
            labels[tail] = pm.D
    px = (base_x[None, :] + amp[:, None] * np.sin(0.37 * t[:, None] + phase)).astype(np.float32)
    py = (base_y[None, :] + amp[:, None] * np.cos(0.37 * t[:, None] + phase)).astype(np.float32)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    return Match("fake", FPS, (105.0, 68.0), px, py, team, px[:, 0] + 0.5, py[:, 0].copy(),
                 [(0, n_half), (n_half, n)], labels)


def end_onset(match, half, before_s):
    return match.halves[half][1] - int(before_s * FPS)


class TestGuard(unittest.TestCase):
    def test_fires_when_the_players_have_stopped_playing(self):
        match = make_match()
        onsets = [end_onset(match, h, 50) for h in (0, 1)]
        picks = wc.end_picks(match, onsets, guard=True)
        self.assertTrue(all(p["fires"] for p in picks), picks)
        out = wc.end_of_half(match.labels.copy() * 0 + pm.H, match, onsets, pm.D, guard=True)
        for h in (0, 1):
            self.assertTrue(np.all(out[onsets[h]:match.halves[h][1]] == pm.D))
            self.assertEqual(out[onsets[h] - 1], pm.H)

    def test_does_not_fire_while_they_are_playing(self):
        match = make_match(slow_end=False)
        onsets = [end_onset(match, h, 50) for h in (0, 1)]
        self.assertFalse(any(p["fires"] for p in wc.end_picks(match, onsets, guard=True)))
        states = np.full(match.n_frames, pm.H, dtype=np.int8)
        np.testing.assert_array_equal(
            wc.end_of_half(states, match, onsets, pm.D, guard=True), states)
        # without the guard (D46's v7_whistle_end) the same whistles end the halves
        self.assertTrue(all(p["fires"] for p in wc.end_picks(match, onsets, guard=False)))

    def test_too_early_or_no_whistle_does_not_fire(self):
        match = make_match()
        early = [end_onset(match, h, wc.END_MAX_S + 10) for h in (0, 1)]
        self.assertFalse(any(p["fires"] for p in wc.end_picks(match, early, guard=True)))
        picks = wc.end_picks(match, [], guard=True)
        self.assertTrue(all(p["pick"] is None and not p["fires"] for p in picks))

    def test_last_group_is_the_pick(self):
        match = make_match()
        stop = match.halves[0][1]
        onsets = [stop - 90 * FPS, stop - 50 * FPS, stop - 49 * FPS]   # a foul, then two blasts
        pick = wc.end_picks(match, onsets, guard=True)[0]
        self.assertEqual(pick["pick"], stop - 50 * FPS)


class TestWalkoff(unittest.TestCase):
    def test_inserts_dead_ball_after_each_half(self):
        match = make_match(slow_end=False)
        out = add_walkoff(match, 30, seed=1)
        n = 30 * FPS
        self.assertEqual(out.n_frames, match.n_frames + 2 * n)
        (a1, b1), (a2, b2) = match.halves
        self.assertEqual(out.halves, [(a1, b1 + n), (a2 + n, b2 + 2 * n)])
        self.assertTrue(np.all(out.labels[b1:b1 + n] == pm.D))
        self.assertTrue(np.all(out.labels[b2 + n:] == pm.D))
        np.testing.assert_array_equal(out.labels[:b1], match.labels[:b1])
        np.testing.assert_array_equal(out.labels[b1 + n:b2 + n], match.labels[a2:b2])
        self.assertEqual(out.notes["walkoff"], {"frames": n, "at": [b1, b2],
                                                "ends": [b1, b2 + n]})

    def test_players_walk_off_and_labels_do_not_depend_on_the_seed(self):
        match = make_match(slow_end=False)
        a, b = add_walkoff(match, 90, seed=0), add_walkoff(match, 90, seed=1)
        np.testing.assert_array_equal(a.labels, b.labels)
        self.assertFalse(np.array_equal(a.player_x, b.player_x, equal_nan=True))
        stop = match.halves[0][1]
        stretch = slice(stop, stop + 90 * FPS)
        tracked = np.isfinite(a.player_x[stretch]).sum(axis=1)
        self.assertEqual(tracked[0], 22)
        self.assertLess(tracked[-1], 22)               # some have left the pitch
        speed = mean_player_speed(a)[stretch]
        self.assertLess(np.nanmedian(speed), wc.END_GUARD_MS)

    def test_whistle_frames_follow_the_insertion(self):
        match = make_match(slow_end=False)
        out = add_walkoff(match, 30, seed=0)
        n = 30 * FPS
        (_, b1), (a2, b2) = match.halves
        self.assertEqual(wc.walkoff_frame(b1 - 1, out), b1 - 1)
        self.assertEqual(wc.walkoff_frame(a2 + 10, out), a2 + 10 + n)
        self.assertEqual(wc.half_ends(out), [(0, b1), (a2 + n, b2 + n)])
        self.assertEqual(wc.half_ends(match), match.halves)


if __name__ == "__main__":
    unittest.main()
