"""tools/restart_check.py's matching of stoppages to restarts: no data needed."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from tools import restart_check as rc  # noqa: E402


class RestartCheckTest(unittest.TestCase):
    def test_runs_and_classify(self):
        truth = np.array([pm.H] * 10 + [pm.D] * 10 + [pm.A] * 10 + [pm.D] * 10 + [pm.H] * 10,
                         np.int8)
        runs = rc.dead_runs(truth, [(0, 50)])
        self.assertEqual(runs, [(10, 20), (30, 40)])
        # fps 1: a goal kick 2 frames after the first run ends, a free kick after an offside
        # at the second, and an event far from both that matches nothing
        kinds = rc.classify(runs, [(21, "goal_kick"), (39, "free_kick"), (48, "throw_in")],
                            np.array([25]), fps=1)
        self.assertEqual(kinds, ["goal_kick", "offside_fk"])
        kinds = rc.classify(runs, [(39, "free_kick")], np.array([], np.int64), fps=1)
        self.assertEqual(kinds, ["unmatched", "free_kick"])


if __name__ == "__main__":
    unittest.main()
