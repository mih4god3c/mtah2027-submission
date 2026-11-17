"""tools/skillcorner_check.py on synthetic phases. No network and no match data needed."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from tools import skillcorner_check as sc  # noqa: E402

META = {"home_team": {"id": 7}, "match_periods": [{"start_frame": 2, "end_frame": 21},
                                                   {"start_frame": 25, "end_frame": 34}]}


def phase(start, end, team, period=1):
    return {"start": start, "end": end, "team": team, "period": period}


class SkillCornerCheckTest(unittest.TestCase):
    def test_labels_read_frame_end_as_exclusive_and_gaps_as_dead(self):
        phases = [phase(2, 6, 7), phase(6, 10, 9), phase(14, 22, 7),  # 22 > period end 21
                  phase(25, 30, 9, 2)]
        codes, outside = sc.labels(META, phases)
        H, A, D, X = pm.H, pm.A, pm.D, pm.X
        expected = [X, X] + [H] * 4 + [A] * 4 + [D] * 4 + [H] * 8 + [X] * 3 + [A] * 5 + [D] * 5
        self.assertEqual(codes.tolist(), expected)
        self.assertEqual(outside, 0)   # frame 21 is the last period frame; 22 is never covered

    def test_frames_outside_the_periods_are_counted_not_labelled(self):
        codes, outside = sc.labels(META, [phase(20, 27, 7)])
        self.assertEqual(outside, 3)   # frames 22, 23, 24
        self.assertEqual(codes[22:25].tolist(), [pm.X] * 3)

    def test_joins_report_tiling_gaps_and_same_team_neighbours(self):
        phases = [phase(0, 5, 7), phase(5, 9, 7), phase(12, 20, 9), phase(30, 40, 9, 2)]
        gaps, same = sc.joins(phases)
        self.assertEqual(gaps.tolist(), [0, 3])   # no join across periods
        self.assertEqual(same.tolist(), [0])

    def test_stoppages_are_dead_runs_inside_periods(self):
        codes, _ = sc.labels(META, [phase(2, 6, 7), phase(10, 22, 9)])
        np.testing.assert_allclose(sc.stoppages(codes, 10), [0.4, 1.0])


if __name__ == "__main__":
    unittest.main()
