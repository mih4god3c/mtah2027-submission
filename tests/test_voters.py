"""tools/voters.py's scoring on synthetic calls: no data needed."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from tools import voters  # noqa: E402

H, A, D, N = pm.H, pm.A, pm.D, voters.NONE


class VotersScoreTest(unittest.TestCase):
    def test_team_and_dead_counts(self):
        truth = np.array([H, H, A, A, D, D], np.int8)
        final = np.array([H, A, A, D, D, H], np.int8)
        seen = np.array([1, 1, 0, 0, 1, 1], bool)
        graded = np.ones(6, bool)
        team = np.array([H, H, A, A, N, N], np.int8)       # always right where alive
        calls = {k: np.full(6, N, np.int8) for k in voters.TEAM_VOTERS + voters.DEAD_VOTERS}
        calls["holder"] = team
        calls["deadmodel"] = np.array([H, H, H, H, D, D], np.int8)
        t = voters.Tally()
        voters.score(t, calls, final, seen, truth, graded)
        self.assertEqual(t.share("holder:right", "holder:speaks"), 1.0)
        # final is wrong on frames 1 (other team) and 3 (D): both rescued, one a wrong team
        self.assertEqual(t.c["holder:rescue"], 2)
        self.assertEqual(t.c["holder:rescue_team"], 1)
        self.assertEqual(t.share("final_team_right", "alive"), 0.5)
        self.assertEqual(t.share("deadmodel:right", "deadmodel:speaks"), 1.0)
        self.assertEqual(t.c["deadmodel:rescue"], 2)   # frames 3 and 5
        self.assertEqual(t.c["nearest:speaks"], 0)


class VotersBlocksTest(unittest.TestCase):
    def test_blocks_follow_spans(self):
        # half 1 frames 0..11, kickoff at 2; half 2 starts at 12, kickoff at 14
        starts = voters.block_starts(20, [(2, 12), (14, 20)])
        self.assertEqual(list(starts), [0, 2, 7, 12, 14, 19])

    def test_majority(self):
        self.assertEqual(voters.majority(np.array([H, H, A, N, N], np.int8)), "H")
        self.assertEqual(voters.majority(np.array([H, A, N, N, N], np.int8)), "?")
        self.assertEqual(voters.majority(np.array([N, N], np.int8)), "-")

    def test_block_rows(self):
        n = 10
        calls = {k: np.full(n, N, np.int8) for k in voters.BLOCK_VOTERS}
        calls["holder"][:] = A
        calls["deadmodel"][:5] = D
        calls["deadmodel"][5:] = H                     # the "alive" placeholder
        final = np.array([D] * 5 + [A] * 5, np.int8)
        truth = np.array([pm.X] * 5 + [A] * 5, np.int8)
        raw = {"nearest": np.array([0, 0, 1, -1, -1] * 2, np.int8),
               "holder": np.full(n, 1, np.int8),
               "spread": np.array([1.0, 3.0, np.nan, np.nan, np.nan] + [-2.0] * 5),
               "frame": np.array([0.0] * 5 + [-1.0] * 5),
               "p_dead": np.array([0.9] * 5 + [0.1] * 5)}
        seen = np.array([1] * 5 + [1, 1, 0, 0, 0], bool)
        raw["frame"][7:] = -3.0                        # frames the ball is not seen on
        rows = voters.block_rows(calls, raw, final, seen, truth, [(0, n)])
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0]["truth"], rows[0]["final"], rows[0]["deadmodel"]), ("X", "D", "D"))
        self.assertEqual((rows[1]["truth"], rows[1]["final"], rows[1]["deadmodel"]), ("A", "A", "alive"))
        self.assertEqual(rows[1]["holder"], "A")
        self.assertEqual(rows[1]["nearest"], "-")
        self.assertEqual(rows[1]["frame_to"], 9)
        self.assertEqual((rows[0]["nearest_home"], rows[0]["nearest_away"]), (2, 1))
        self.assertEqual((rows[1]["holder_home"], rows[1]["holder_away"]), (0, 5))
        self.assertEqual(rows[0]["spread_m"], 2.0)    # NaN frames left out
        self.assertEqual(rows[0]["frame_p_home"], 0.5)
        self.assertEqual(rows[1]["frame_log_odds"], -2.2)
        self.assertEqual(rows[1]["frame_seen_log_odds"], -1.0)
        self.assertEqual((rows[0]["p_dead"], rows[1]["p_dead"]), (0.9, 0.1))
        self.assertEqual(rows[1]["ball_seen"], 2)


if __name__ == "__main__":
    unittest.main()
