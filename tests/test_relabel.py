"""Tests for solution/relabel.py, the controlled-possession relabelling (D20). Synthetic only."""

import os
import sys
import unittest
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import relabel  # noqa: E402
from solution.match import pm  # noqa: E402

FPS = 10      # 2 s = 20 frames


def codes(*parts):
    """codes(("H", 50), ("A", 5), ...) -> scorer codes, with one X frame either side."""
    letters = ["X"] + [c for c, n in parts for _ in range(n)] + ["X"]
    return pm.label_codes(np.array(letters))


def letters(labels):
    """Run-length string of a code array, e.g. 'H55 D10'."""
    out = []
    for a, b, k in pm.segments(labels, pm.valid_runs(labels)):
        out.append("{}{}".format("HAD"[k], b - a))
    return " ".join(out)


class ControlledTest(unittest.TestCase):

    def run_rule(self, labels, confirm=None):
        return letters(relabel.controlled(labels, FPS, 2.0, confirm))

    def test_short_touch_goes_back_to_the_last_team(self):
        self.assertEqual(self.run_rule(codes(("H", 50), ("A", 5), ("H", 50))), "H105")

    def test_long_run_is_a_change(self):
        self.assertEqual(self.run_rule(codes(("H", 50), ("A", 20), ("H", 50))), "H50 A20 H50")

    def test_short_run_before_dead_ball_is_absorbed(self):
        self.assertEqual(self.run_rule(codes(("H", 50), ("A", 8), ("D", 30), ("A", 40))),
                         "H58 D30 A40")

    def test_restart_run_is_never_absorbed(self):
        # The first team after a stoppage, and the first of a half, is the clear possessor.
        self.assertEqual(self.run_rule(codes(("D", 30), ("H", 3), ("A", 50))), "D30 H3 A50")
        self.assertEqual(self.run_rule(codes(("H", 3), ("A", 50))), "H3 A50")

    def test_shortest_first_keeps_the_interception_start(self):
        # A intercepts, H touches it once, A keeps it: A from its first frame.
        self.assertEqual(self.run_rule(codes(("H", 50), ("A", 15), ("H", 3), ("A", 40))),
                         "H50 A58")

    def test_contested_ball_stays_with_the_last_team(self):
        self.assertEqual(self.run_rule(codes(("H", 50), ("A", 5), ("H", 2), ("A", 4),
                                             ("H", 50))), "H111")

    def test_dead_ball_untouched(self):
        labels = codes(("H", 50), ("A", 5), ("D", 30), ("H", 7), ("A", 3), ("D", 12))
        out = relabel.controlled(labels, FPS, 2.0)
        np.testing.assert_array_equal(out == pm.D, labels == pm.D)
        np.testing.assert_array_equal(out == pm.X, labels == pm.X)

    def test_pass_confirms_a_short_run(self):
        labels = codes(("H", 50), ("A", 10), ("H", 50))
        start = 1 + 50                                   # first A frame
        confirm = relabel.pass_confirm(([start + 4], [pm.A]), FPS, tol_s=1.0)
        self.assertEqual(self.run_rule(labels, confirm), "H50 A10 H50")
        wrong_team = relabel.pass_confirm(([start + 4], [pm.H]), FPS, tol_s=1.0)
        self.assertEqual(self.run_rule(labels, wrong_team), "H110")
        too_late = relabel.pass_confirm(([start + 10 + 10], [pm.A]), FPS, tol_s=1.0)
        self.assertEqual(self.run_rule(labels, too_late), "H110")

    def test_halves_are_separate(self):
        a = codes(("H", 50), ("A", 5))
        b = codes(("A", 3), ("H", 40))
        labels = np.concatenate([a, b])
        out = relabel.controlled(labels, FPS, 2.0)
        self.assertEqual(len(pm.valid_runs(out)), 2)
        self.assertEqual(letters(out), "H55 A3 H40")

    def test_random_labels_properties_and_determinism(self):
        rng = np.random.RandomState(0)
        parts = []
        for _ in range(400):
            parts.append((rng.choice(["H", "A", "D"], p=[0.45, 0.45, 0.1]),
                          int(rng.randint(1, 80))))
        labels = codes(*parts)
        out = relabel.controlled(labels, FPS, 2.0)
        again = relabel.controlled(labels.copy(), FPS, 2.0)
        self.assertEqual(zlib.crc32(out.tobytes()), zlib.crc32(again.tobytes()))
        np.testing.assert_array_equal(out == pm.D, labels == pm.D)
        runs = pm.valid_runs(out)
        segs = pm.segments(out, runs)
        for i, (a, b, k) in enumerate(segs):
            if k == pm.D or i == 0 or segs[i - 1][2] == pm.D:
                continue
            self.assertGreaterEqual(b - a, 20, "a non-restart team run under keep_s survived")
        self.assertLessEqual(len(pm.possession_spells(out, runs)),
                             len(pm.possession_spells(labels, runs)))
        # Idempotent: controlled labels are already controlled.
        np.testing.assert_array_equal(relabel.controlled(out, FPS, 2.0), out)

    def test_stats_units(self):
        labels = codes(("H", 50), ("A", 10), ("D", 40))
        s = relabel.summary([relabel.stats(labels, FPS)])
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["per90"], 2 / 10.0 * 5400.0)
        self.assertAlmostEqual(s["direct_per_alive_min"], 1 / 6.0 * 60.0)
        self.assertAlmostEqual(s["under2"], 0.5)


if __name__ == "__main__":
    unittest.main()


class LoaderSwitchTest(unittest.TestCase):

    def test_relabelled_copies_and_dfl_is_identity(self):
        from solution.match import Match
        from solution.sources import idsse
        labels = codes(("H", 50), ("A", 5), ("H", 50))
        n = len(labels)
        match = Match("idsse_TEST", 25, (105.0, 68.0), np.zeros((n, 2), np.float32),
                      np.zeros((n, 2), np.float32), np.array([0, 1], np.int8),
                      np.zeros(n, np.float32), np.zeros(n, np.float32), [(1, n - 1)], labels)
        self.assertIs(idsse.relabelled(match, "TEST", "dfl"), match)
        out = idsse.relabelled(match, "TEST", "keep2")
        self.assertIsNot(out, match)
        self.assertEqual(letters(out.labels), "H105")
        self.assertEqual(letters(match.labels), "H50 A5 H50")
        with self.assertRaises(ValueError):
            idsse.relabelled(match, "TEST", "keep9")
        self.assertEqual(idsse.LABELS, os.environ.get("MTAH_PROXY_LABELS", "dfl"))
