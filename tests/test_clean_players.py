"""simulate.degrade(players=False): the perfect player tracker what-if (no data needed)."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.simulate import degrade  # noqa: E402
from tests.test_ballmodel import walking_match  # noqa: E402


class CleanPlayers(unittest.TestCase):
    def test_same_ball_gaps_clean_players(self):
        _, clean = walking_match(n=6000)
        normal = degrade(clean, seed=3)
        kept = degrade(clean, seed=3, players=False)
        self.assertTrue(np.array_equal(normal.ball_seen, kept.ball_seen))
        self.assertTrue(np.array_equal(kept.player_x, clean.player_x, equal_nan=True))
        self.assertTrue(np.array_equal(kept.team, clean.team))
        self.assertEqual(kept.notes["degraded"]["players"], "clean")
        self.assertNotIn("players", normal.notes["degraded"])

    def test_v0_refuses(self):
        _, clean = walking_match(n=3000)
        with self.assertRaises(ValueError):
            degrade(clean, seed=0, version="v0", players=False)


if __name__ == "__main__":
    unittest.main()
