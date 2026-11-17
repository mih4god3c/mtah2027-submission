"""Tests for tools/tag_anchor_check.py on synthetic arrays. No data needed.

    .venv/bin/python -m unittest tests.test_tag_anchor_check -v       # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from tools import tag_anchor_check as tac  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X
FPS = 25


class Anchors(unittest.TestCase):

    def codes(self):
        # one half of 100 s: A possession, D from 30 s, H restart at 42 s, D again from 80 s
        c = np.full(100 * FPS, A, dtype=np.int8)
        c[30 * FPS:42 * FPS] = D
        c[42 * FPS:80 * FPS] = H
        c[80 * FPS:] = D
        return c

    def test_offsets_shares_and_team_agreement(self):
        codes = self.codes()
        events = np.array([10 * FPS, 41 * FPS, 43 * FPS, 60 * FPS])
        # a tag opening 1 s before the decoded restart, with the right team
        windows = [(41 * FPS, 76 * FPS, "throw_in", H)]
        r = tac.anchor_records(codes, events, windows, [(0, 100 * FPS)], FPS)[0]
        self.assertAlmostEqual(r["offset"], 1.0)
        self.assertAlmostEqual(r["touch_offset"], 0.0)          # the event at 41 s
        self.assertAlmostEqual(r["d_before"], 1.0)               # 31..41 s is all D
        self.assertAlmostEqual(r["d_after"], 1 / 5.0)            # 41..46 s: one second of D
        self.assertAlmostEqual(r["d_rest"], 0.0)
        self.assertTrue(r["team_agree"])
        # the same tag with the wrong team
        r = tac.anchor_records(codes, events, [(41 * FPS, 76 * FPS, "throw_in", A)],
                               [(0, 100 * FPS)], FPS)[0]
        self.assertFalse(r["team_agree"])

    def test_no_restart_in_reach_gives_none(self):
        codes = self.codes()
        windows = [(10 * FPS, 45 * FPS, "corner", None)]    # restart at 42 s is 32 s away
        r = tac.anchor_records(codes, np.array([], dtype=np.int64), windows,
                               [(0, 100 * FPS)], FPS)[0]
        self.assertIsNone(r["offset"])
        self.assertIsNone(r["touch_offset"])
        self.assertIsNone(r["team_agree"])

    def test_summary_rows_pool_into_all(self):
        records = [{"kind": "throw_in", "offset": 0.5, "touch_offset": 0.2, "d_before": 0.9,
                    "d_after": 0.1, "d_rest": 0.0, "team_agree": None},
                   {"kind": "corner", "offset": None, "touch_offset": None, "d_before": 0.2,
                    "d_after": 0.0, "d_rest": 0.0, "team_agree": True}]
        rows = tac.summarise(records)
        self.assertEqual(rows["all"]["n"], 2)
        self.assertAlmostEqual(rows["all"]["near"], 0.5)
        self.assertAlmostEqual(rows["all"]["reach"], 0.5)
        self.assertEqual(rows["corner"]["offset"], "--")
        self.assertEqual(rows["all"]["team"], "1.00")


if __name__ == "__main__":
    unittest.main()
