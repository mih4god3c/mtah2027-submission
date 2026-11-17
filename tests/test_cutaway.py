"""Tests for the TV cut-away cue: solution/cutaway.py and the decoder's tv term. No data.

    .venv/bin/python -m unittest tests.test_cutaway -v
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import broadcast as bc  # noqa: E402
from solution import cutaway, decoder  # noqa: E402
from solution.match import Match, pm  # noqa: E402

FPS = 25


def match_with_stoppages(n=60000, seed=0):
    """Alternating possession (H/A, 20 s) and stoppages of 3, 10, 20 and 40 s; one half."""
    labels = np.full(n, pm.X, dtype=np.int8)
    lengths = (3, 10, 20, 40)
    t, k = 100, 0
    while t < n - 100:
        play = min(20 * FPS, n - 100 - t)
        labels[t:t + play] = pm.H if k % 2 else pm.A
        t += play
        stop = min(lengths[k % 4] * FPS, n - 100 - t)
        labels[t:t + stop] = pm.D
        t += stop
        k += 1
    labels[t:n - 100] = pm.H
    rng = np.random.RandomState(seed)
    z = np.zeros((n, 2), dtype=np.float32)
    match = Match("cut", FPS, (105.0, 68.0), z, z, np.array([0, 1], dtype=np.int8),
                  (rng.rand(n) * 105).astype(np.float32), (rng.rand(n) * 68).astype(np.float32),
                  [(100, n - 100)], labels)
    return match


class Synthetic(unittest.TestCase):

    def test_outside_halves_unknown_and_deterministic(self):
        m = match_with_stoppages()
        tv = cutaway.synthetic(m, m.labels, seed=0)
        self.assertTrue(np.all(tv[:100] == -1))
        self.assertTrue(np.all(tv[-100:] == -1))
        self.assertTrue(set(np.unique(tv[100:-100])) <= {0, 1})
        self.assertTrue(np.array_equal(tv, cutaway.synthetic(m, m.labels, seed=0)))
        self.assertFalse(np.array_equal(tv, cutaway.synthetic(m, m.labels, seed=1)))

    def test_long_stoppages_always_cut_short_ones_rarely(self):
        m = match_with_stoppages()
        tv = cutaway.synthetic(m, m.labels, seed=0, variant="optimistic")
        hit = {3: [], 40: []}
        for start, stop, code in pm.segments(m.labels, pm.valid_runs(m.labels)):
            if code == pm.D and (stop - start) // FPS in hit:
                hit[(stop - start) // FPS].append(tv[start:stop].any())
        self.assertEqual(np.mean(hit[40]), 1.0)
        self.assertLess(np.mean(hit[3]), 0.4)

    def test_pessimistic_spills_into_live_more(self):
        m = match_with_stoppages()
        live = np.isin(m.labels, (pm.H, pm.A))
        opt = cutaway.synthetic(m, m.labels, seed=0, variant="optimistic")
        pes = cutaway.synthetic(m, m.labels, seed=0, variant="pessimistic")
        self.assertGreater((pes[live] == 1).mean(), (opt[live] == 1).mean() + 0.01)
        self.assertLess((opt[live] == 1).mean(), 0.03)


class CueFor(unittest.TestCase):

    def test_attached_then_degraded_seed_then_error(self):
        m = match_with_stoppages()
        attached = m.copy()
        attached.notes["tv"] = np.ones(m.n_frames, dtype=np.int8)
        self.assertTrue(np.all(cutaway.cue_for(attached, "calibrated") == 1))
        degraded = m.copy()
        degraded.notes["degraded"] = {"version": "v1", "seed": 2}
        self.assertTrue(np.array_equal(cutaway.cue_for(degraded, "calibrated"),
                                       cutaway.synthetic(m, m.labels, 2, "calibrated")))
        with self.assertRaises(ValueError):
            cutaway.cue_for(m, "calibrated")                  # clean, no seed: no cue
        unlabelled = degraded.copy()
        unlabelled.labels = None
        with self.assertRaises(ValueError):
            cutaway.cue_for(unlabelled, "calibrated")         # never invents one without truth


class Table(unittest.TestCase):

    def test_fit_table(self):
        labels = np.array([pm.H] * 100 + [pm.D] * 100 + [pm.X] * 10, dtype=np.int8)
        tv = np.zeros(210, dtype=np.int8)
        tv[:10] = 1                    # 10% off while alive
        tv[100:180] = 1                # 80% off while dead
        tv[205:] = 1                   # X: ignored
        table = np.exp(cutaway.fit_table([(labels, tv)]))
        self.assertAlmostEqual(table[0, 1], 0.1)
        self.assertAlmostEqual(table[1, 1], 0.8)
        self.assertTrue(np.allclose(table.sum(axis=1), 1.0))

    def test_from_alignment(self):
        state = np.array([bc.NOT_COVERED, bc.LIVE, bc.CUT, bc.REPLAY, bc.OTHER], dtype=np.int8)
        self.assertEqual(cutaway.from_alignment(state).tolist(), [-1, 0, 1, 1, 0])


class DecoderTerm(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        n = 500
        self.args = (rng.randint(-1, 2, n).astype(np.int8), rng.rand(n) < 0.6, rng.rand(n))
        self.priors = {"agree": 0.7, "speed_edges": np.array([0.33, 0.66]),
                       "speed_log": np.log(np.full((2, 3), 1 / 3.0)),
                       "tv_log": np.log(np.array([[0.98, 0.02], [0.3, 0.7]]))}
        self.tv = np.where(np.arange(n) < 250, 1, 0).astype(np.int8)
        self.tv[400:] = -1

    def emissions(self, **params):
        return decoder.block_emissions(*self.args, self.priors, dict(decoder.DEFAULTS, **params),
                                       tv=self.tv)

    def test_off_by_default(self):
        plain = decoder.block_emissions(*self.args, self.priors, decoder.DEFAULTS)
        self.assertTrue(np.array_equal(plain, self.emissions()))

    def test_off_pitch_favours_dead_and_unknown_is_neutral(self):
        plain = self.emissions()
        on = self.emissions(tv_w=1.0)
        delta = on - plain
        first, middle, last = delta[0], delta[60], delta[-1]      # off, on, unknown blocks
        self.assertGreater(first[decoder.D] - first[decoder.H], 3.0)
        self.assertLess(middle[decoder.D] - middle[decoder.H], 0.0)
        self.assertTrue(np.allclose(last, 0.0))
        self.assertAlmostEqual(first[decoder.H], first[decoder.A])


if __name__ == "__main__":
    unittest.main()
