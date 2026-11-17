"""Tests for tools/frame_report.py on synthetic label arrays. No data needed.

    .venv/bin/python -m unittest tests.test_frame_report -v       # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from tools import frame_report as fr  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def kinds_of(gt, pred, runs):
    kinds = fr.classify_frames(np.array(gt, dtype=np.int8), np.array(pred, dtype=np.int8), runs)
    return ["" if k < 0 else fr.KINDS[k] for k in kinds]


class ClassifyFrames(unittest.TestCase):

    def test_late_and_early_boundaries(self):
        gt = [H, H, H, H, A, A, A, A]
        pred = [H, H, H, H, H, H, A, A]      # keeps H two frames past the boundary
        self.assertEqual(kinds_of(gt, pred, [(0, 8)]),
                         ["", "", "", "", "late", "late", "", ""])
        pred = [H, H, A, A, A, A, A, A]      # takes A two frames before the boundary
        self.assertEqual(kinds_of(gt, pred, [(0, 8)]),
                         ["", "", "early", "early", "", "", "", ""])

    def test_missed_true_segment_between_late_and_early_is_missed(self):
        gt = [A, A, H, H, H, D, D, D]
        pred = [A, A, A, D, D, D, D, D]      # the short H never appears
        self.assertEqual(kinds_of(gt, pred, [(0, 8)]),
                         ["", "", "missed", "missed", "missed", "", "", ""])

    def test_spurious_segment_bridging_a_boundary_is_one_mistake(self):
        gt = [A, A, A, A, D, D, D, D, H, H]
        pred = [A, A, H, H, H, H, D, D, H, H]  # invented H across the A>D boundary, D late
        self.assertEqual(kinds_of(gt, pred, [(0, 10)]),
                         ["", "", "spurious", "spurious", "spurious", "spurious", "", "",
                          "", ""])
        gt_a = np.array(gt, dtype=np.int8)
        pred_a = np.array(pred, dtype=np.int8)
        table = fr.error_table(gt_a, pred_a, [(0, 10)])
        self.assertEqual([(s, t, k, b) for s, t, _, _, k, b in table],
                         [(2, 4, "spurious", True), (4, 6, "spurious", True)])
        events = fr.kind_events(gt_a, pred_a, [(0, 10)], table)
        self.assertEqual(events["spurious"], 1)
        self.assertEqual(events["missed"], 0)

    def test_early_across_a_missed_segment(self):
        gt = [A, A, A, D, D, H, H, H]
        pred = [A, A, H, H, H, H, H, H]      # H starts early and bridges the missed D
        self.assertEqual(kinds_of(gt, pred, [(0, 8)]),
                         ["", "", "early", "missed", "missed", "", "", ""])

    def test_nothing_is_other_and_counts_add_up(self):
        rng = np.random.RandomState(0)
        for _ in range(50):
            gt = np.repeat(rng.randint(0, 3, size=12), rng.randint(1, 6, size=12)).astype(np.int8)
            pred = gt.copy()
            flips = rng.randint(0, len(gt), size=6)
            pred[flips] = rng.randint(0, 3, size=6)
            runs = [(0, len(gt))]
            kinds = fr.classify_frames(gt, pred, runs)
            self.assertFalse((kinds == fr.KINDS.index("other")).any())
            self.assertEqual(int((kinds >= 0).sum()), int((gt != pred).sum()))
            table = fr.error_table(gt, pred, runs)
            self.assertEqual(sum(t - s for s, t, _, _, _, _ in table), int((gt != pred).sum()))

    def test_runs_never_cross_a_half_boundary(self):
        gt = [H, H, X, X, A, A]
        pred = [A, A, X, X, A, H]
        self.assertEqual(kinds_of(gt, pred, [(0, 2), (4, 6)]),
                         ["missed", "missed", "", "", "", "spurious"])


class Views(unittest.TestCase):

    def test_segment_rows_split_head_tail_and_interior(self):
        gt = np.array([H, H, H, H, H, H, A, A], dtype=np.int8)
        pred = np.array([A, H, D, H, H, A, A, A], dtype=np.int8)
        rows = fr.segment_rows(gt, pred, [(0, 8)])
        self.assertEqual((rows[0]["head"], rows[0]["tail"], rows[0]["interior"]), (1, 1, 1))
        self.assertEqual(rows[0]["n_pred_segments"], 5)
        self.assertEqual((rows[1]["head"], rows[1]["tail"], rows[1]["interior"]), (0, 0, 0))

    def test_transition_rows_report_latency_or_none(self):
        gt = np.array([H, H, H, A, A, A, D, D, D], dtype=np.int8)
        pred = np.array([H, H, H, H, A, A, A, A, A], dtype=np.int8)
        rows, spurious = fr.transition_rows(gt, pred, [(0, 9)], tol=2)
        self.assertEqual([(f, lat) for f, _, _, lat in rows], [(3, 1), (6, None)])
        self.assertEqual(spurious, [])

    def test_boundary_mask_drops_a_margin_around_true_transitions(self):
        gt = np.array([X, H, H, H, A, A, A, X], dtype=np.int8)
        mask = fr.boundary_mask(gt, [(1, 7)], 1)
        self.assertEqual(mask.tolist(), [False, True, True, False, False, True, True, False])

    def test_repair_costs_sum_to_the_perfect_score(self):
        gt = np.array([H, H, H, H, A, A, D, D, D, H, H], dtype=np.int8)
        pred = np.array([H, H, A, A, A, A, A, D, D, D, H], dtype=np.int8)
        runs = [(0, 11)]
        graded = np.ones(11, dtype=bool)
        table = fr.error_table(gt, pred, runs)
        fixed = pred.copy()
        for s, t, _, _, _, _ in table:
            fixed[s:t] = gt[s:t]
        self.assertEqual(fr.macro_f1_of(gt, fixed, graded), 1.0)
        costs = fr.repair_costs(gt, pred, table, graded)
        self.assertGreater(costs["early"], 0.0)
        self.assertGreater(costs["late"], 0.0)

    def test_timeline_marks_cells(self):
        gt = np.array([H] * 4 + [A] * 4, dtype=np.int8)
        pred = np.array([H] * 4 + [H, A, A, A], dtype=np.int8)
        lines = fr.timeline(gt, pred, 0, 8, res_s=2 / 25.0, per_line_s=8 / 25.0)
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[1].endswith("HHAA  truth"))
        self.assertTrue(lines[2].endswith("HHHA  file"))
        self.assertTrue(lines[3].endswith("  h   differs"))


if __name__ == "__main__":
    unittest.main()
