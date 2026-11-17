"""tools/segments.py on synthetic labels: no data needed."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from tools.segments import clock, to_segments  # noqa: E402


def _codes(text):
    return pm.label_codes(np.array(list(text), dtype=object))


class SegmentsTest(unittest.TestCase):
    def test_segments_per_half_with_abstention_filled(self):
        # half 1 = frames 2..7, half 2 = frames 10..13; X outside, one X inside half 1
        codes = _codes("XXHHXAADXXAAHHX")
        blocks = [(1, 2, 7), (2, 10, 13)]
        rows = to_segments(codes, blocks, fps=1.0, keep=("H", "A", "D"))
        got = [(r["half"], r["label"], r["start_frame"], r["end_frame"]) for r in rows]
        assert got == [(1, "H", 2, 4), (1, "A", 5, 6), (1, "D", 7, 7),
                       (2, "A", 10, 11), (2, "H", 12, 13)]
        assert rows[0]["start_s"] == 2.0 and rows[0]["end_s"] == 5.0 and rows[0]["duration_s"] == 3.0

    def test_possessions_only_by_default(self):
        rows = to_segments(_codes("HHDDAA"), [(1, 0, 5)], fps=1.0)
        assert [r["label"] for r in rows] == ["H", "A"]
        assert [r["segment"] for r in rows] == [1, 2]

    def test_clock(self):
        assert clock(0) == "0:00:00.00"
        assert clock(3725.5) == "1:02:05.50"
