"""Tests for tools/diff_submissions.py on synthetic label arrays. No data needed.

    .venv/bin/python -m unittest tests.test_diff_submissions -v       # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from tools import diff_submissions as ds  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X


class DiffRuns(unittest.TestCase):

    def test_runs_are_maximal_constant_and_inside_the_halves(self):
        halves = [(2, 12), (14, 20)]
        a = np.array([X, X, H, H, H, H, A, A, D, D, H, H, X, X, A, A, A, D, D, D], dtype=np.int8)
        b = np.array([X, X, H, A, A, H, A, D, D, D, H, D, X, X, H, H, A, D, A, A], dtype=np.int8)
        runs = ds.diff_runs(a, b, halves)
        self.assertEqual(runs, [(3, 5, H, A), (7, 8, A, D), (11, 12, H, D), (14, 16, A, H),
                                (18, 20, D, A)])

    def test_identical_files_give_no_runs(self):
        a = np.array([X, H, H, D, A, X], dtype=np.int8)
        self.assertEqual(ds.diff_runs(a, a.copy(), [(1, 5)]), [])

    def test_a_run_never_crosses_a_half_boundary(self):
        a = np.array([H, H, H, H], dtype=np.int8)
        b = np.array([A, A, A, A], dtype=np.int8)
        self.assertEqual(ds.diff_runs(a, b, [(0, 2), (2, 4)]), [(0, 2, H, A), (2, 4, H, A)])

    def test_summary_counts_frames_by_label_pair(self):
        halves = [(0, 6)]
        a = np.array([H, H, H, D, D, A], dtype=np.int8)
        b = np.array([H, A, A, D, H, A], dtype=np.int8)
        lines = ds.summary(a, b, halves, 25)
        self.assertTrue(any("frames that differ: 3 of 6" in line for line in lines))
        self.assertTrue(any("A H -> B A:      2 frames" in line for line in lines))
        self.assertTrue(any("A D -> B H:      1 frames" in line for line in lines))

    def test_clock_reads_as_minutes_and_seconds(self):
        self.assertEqual(ds.clock(0, 25), "00:00.0")
        self.assertEqual(ds.clock(25 * 61 + 12, 25), "01:01.5")
        self.assertEqual(ds.clock(113742, 25), "75:49.7")


if __name__ == "__main__":
    unittest.main()
