"""Tests for solution/whistlecue.py on hand-made inputs (no data needed)."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import whistlecue as wc  # noqa: E402
from solution.match import pm  # noqa: E402


class FakeMatch:
    fps = 25
    pitch = (105.0, 68.0)

    def __init__(self, n, x, y, halves):
        self.ball_x, self.ball_y = np.asarray(x, float), np.asarray(y, float)
        self.halves, self.n_frames, self.notes, self.name = halves, n, {}, "fake"

    @property
    def ball_seen(self):
        return np.isfinite(self.ball_x) & np.isfinite(self.ball_y)

    def in_play_mask(self):
        out = np.zeros(self.n_frames, dtype=bool)
        for a, b in self.halves:
            out[a:b] = True
        return out


def cat(since_bin, place):
    return since_bin * wc.N_PLACE + place


class TestCategories(unittest.TestCase):
    def test_since_and_place(self):
        n = 3000
        x, y = np.full(n, 52.0), np.full(n, 34.0)          # centre spot
        x[2000:], y[2000:] = 2.0, 34.0                     # near a goal line
        x[2600:], y[2600:] = np.nan, np.nan                # lost; known for PLACE_S more
        m = FakeMatch(n, x, y, [(100, n)])
        cats = wc.categories(m, [500])
        self.assertEqual(cats[50], -1)                     # outside the half
        self.assertEqual(cats[200], cat(3, 0))             # no whistle yet, mid-pitch
        self.assertEqual(cats[510], cat(0, 0))             # 0.4 s after the whistle
        self.assertEqual(cats[500 + 10 * 25], cat(1, 0))   # 10 s after
        self.assertEqual(cats[500 + 30 * 25], cat(2, 0))   # 30 s after
        self.assertEqual(cats[2100], cat(3, 1))            # 64 s after, near a line
        self.assertEqual(cats[2600 + 5 * 25], cat(3, 1))   # last seen 5 s ago
        self.assertEqual(cats[2600 + 15 * 25], cat(3, 2))  # last seen 15 s ago: unknown

    def test_fit_table_rows_are_distributions(self):
        labels = np.array([pm.H] * 50 + [pm.D] * 50, dtype=np.int8)
        cats = np.array([0] * 50 + [1] * 50, dtype=np.int8)
        table = wc.fit_table([(labels, cats)])
        np.testing.assert_allclose(np.exp(table).sum(axis=1), 1.0)
        self.assertGreater(table[1, 1] - table[0, 1], 5)


class TestOnsetsAndEnd(unittest.TestCase):
    def test_group_onsets(self):
        events = [{"start": 100, "stop": 110}, {"start": 150, "stop": 160},
                  {"start": 400, "stop": 420}]
        np.testing.assert_array_equal(wc.group_onsets(events, 25), [100, 400])

    def test_end_of_half_uses_last_group_near_the_end(self):
        fps = 25
        stop = int(50 * 60 * fps)
        m = FakeMatch(stop, np.zeros(stop), np.zeros(stop), [(0, stop)])
        states = np.zeros(stop, dtype=np.int8)
        near = stop - 60 * fps
        out = wc.end_of_half(states, m, [46 * 60 * fps, near, near + 30], 2)
        self.assertTrue((out[near:] == 2).all())
        self.assertTrue((out[:near] == 0).all())
        far = wc.end_of_half(states, m, [stop - 300 * fps], 2)   # 5 min before the end: unused
        self.assertTrue((far == 0).all())


if __name__ == "__main__":
    unittest.main()
