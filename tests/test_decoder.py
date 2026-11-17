"""Tests for the semi-Markov decoder (solution/decoder.py). Synthetic inputs only, no data needed.

    .venv/bin/python -m unittest tests.test_decoder -v       # or: make test
"""

import itertools
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder, methods  # noqa: E402
from solution.match import Match, pm  # noqa: E402

H, A, D = decoder.H, decoder.A, decoder.D
FPS = 25


def synthetic_priors():
    """Priors shaped like the IDSSE fit: fast play when alive, slow when dead."""
    centres = decoder.SPEED_EDGES + 0.125
    alive = -0.5 * ((centres - 2.4) / 0.6) ** 2
    dead = -0.5 * ((centres - 1.0) / 0.5) ** 2
    speed_log = np.vstack([alive - np.logaddexp.reduce(alive), dead - np.logaddexp.reduce(dead)])
    return {"durations": {"possession": {"mu": math.log(4.0), "sigma": 1.2, "min_s": 0.2},
                          "dead": {"mu": math.log(18.0), "sigma": 0.9, "min_s": 2.0}},
            "agree": 0.77, "speed_log": speed_log, "speed_edges": decoder.SPEED_EDGES}


def synthetic_evidence(truth, rng, flip=0.2, missing=0.3, gaps=6):
    """Per-frame nearest team, ball seen and speed for a per-frame truth of states 0/1/2."""
    nearest = np.where(truth == D, rng.randint(0, 2, len(truth)), truth).astype(np.int8)
    flips = rng.rand(len(truth)) < flip
    nearest[flips] = 1 - nearest[flips]
    seen = np.ones(len(truth), dtype=bool)
    for start in rng.choice(len(truth), gaps, replace=False):  # long gaps, as in the ATD
        seen[start:start + int(rng.randint(50, 200))] = False
    seen &= rng.rand(len(truth)) > missing * 0.1
    nearest[~seen] = -1
    speed = np.where(truth == D, 1.0, 2.4) + rng.normal(0, 0.3, len(truth))
    return nearest, seen, np.abs(speed)


def runs(states):
    """[(start, stop, state)] maximal runs."""
    edges = np.flatnonzero(np.diff(states)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(states)]))
    return [(int(a), int(b), int(states[a])) for a, b in zip(starts, stops)]


def truth_from(spec):
    """[(state, seconds), ...] -> per-frame states."""
    return np.concatenate([np.full(int(seconds * FPS), state, dtype=np.int8)
                           for state, seconds in spec])


SPEC = [(H, 8), (A, 5), (D, 20), (H, 12), (A, 3), (H, 6), (D, 15), (A, 25), (H, 4), (A, 10)]


class Recovery(unittest.TestCase):

    def test_recovers_clean_segments(self):
        truth = truth_from(SPEC)
        nearest, seen, speed = synthetic_evidence(truth, np.random.RandomState(0), gaps=0)
        out = decoder.decode_half(nearest, seen, speed, synthetic_priors(), decoder.DEFAULTS, FPS)
        self.assertEqual(out.shape, truth.shape)
        self.assertEqual([state for _, _, state in runs(out)], [state for state, _ in SPEC])
        for (a, b, _), (c, d, _) in zip(runs(out), runs(truth)):
            self.assertLessEqual(abs(a - c), 2 * decoder.BLOCK)
            self.assertLessEqual(abs(b - d), 2 * decoder.BLOCK)

    def test_beats_the_majority_vote_on_the_same_evidence(self):
        truth = truth_from(SPEC * 3)
        nearest, seen, speed = synthetic_evidence(truth, np.random.RandomState(1), flip=0.3)
        out = decoder.decode_half(nearest, seen, speed, synthetic_priors(), decoder.DEFAULTS, FPS)
        vote = pm.label_codes(methods.baseline.smooth(nearest, 50))
        self.assertGreater(np.mean(out == truth), np.mean(vote == truth))
        self.assertLess(abs(len(runs(out)) - len(runs(truth))), len(runs(vote)) - len(runs(truth)))


class MinimumDurations(unittest.TestCase):

    def test_viterbi_never_decodes_a_segment_below_its_minimum(self):
        rng = np.random.RandomState(2)
        minimum = {H: 3, A: 2, D: 6}
        log_duration = np.zeros((3, 40))
        for state, m in minimum.items():
            log_duration[state, :m - 1] = -np.inf
        for _ in range(20):
            # Evidence that flips every block, so short segments are what it asks for.
            emissions = rng.normal(0, 3, (200, 3))
            out = decoder.viterbi(emissions, log_duration, np.zeros((3, 3)))
            for start, stop, state in runs(out):
                self.assertGreaterEqual(stop - start, minimum[state])

    def test_decoded_dead_ball_respects_the_fitted_minimum(self):
        priors = synthetic_priors()
        rng = np.random.RandomState(3)
        truth = truth_from([(H, 10), (D, 0.4), (H, 3), (D, 0.6), (A, 10)])   # dead balls < 2 s
        nearest, seen, speed = synthetic_evidence(truth, rng, flip=0.0)
        out = decoder.decode_half(nearest, seen, speed, priors, decoder.DEFAULTS, FPS)
        minimum = priors["durations"]["dead"]["min_s"] * FPS
        for start, stop, state in runs(out):
            if state == D:
                self.assertGreaterEqual(stop - start, minimum)

    def test_duration_prior_is_a_distribution_with_zero_mass_below_the_minimum(self):
        log_p = decoder.duration_log_prior(math.log(4.0), 1.2, 1.0, 0.2, 600)
        self.assertAlmostEqual(float(np.exp(log_p).sum()), 1.0, places=9)
        self.assertTrue(np.all(np.isneginf(log_p[:4])))
        self.assertTrue(np.all(np.isfinite(log_p[4:])))


class Exactness(unittest.TestCase):

    def test_viterbi_matches_brute_force(self):
        rng = np.random.RandomState(4)
        n, d_max = 7, 4
        for _ in range(10):
            emissions = rng.normal(0, 1, (n, 3))
            log_duration = rng.normal(-1, 1, (3, d_max))
            log_duration[D, 0] = -np.inf                   # D needs at least 2 blocks
            trans = rng.normal(-1, 1, (3, 3))
            out = decoder.viterbi(emissions, log_duration, trans)
            best = -np.inf
            for cuts in itertools.product([False, True], repeat=n - 1):
                bounds = [0] + [i + 1 for i, c in enumerate(cuts) if c] + [n]
                lengths = np.diff(bounds)
                if lengths.max() > d_max:
                    continue
                for states in itertools.product(range(3), repeat=len(lengths)):
                    score = sum(emissions[a:b, k].sum() + log_duration[k, b - a - 1]
                                for a, b, k in zip(bounds, bounds[1:], states))
                    score += sum(trans[j, k] for j, k in zip(states, states[1:]))
                    best = max(best, score)

            # Score the decoded path. Same-state neighbours may be one segment or two, so take
            # the better reading: a single segment if it fits, else the best split.
            def path_score(path):
                total, previous = 0.0, None
                for a, b, k in runs(path):
                    segment = max(score_split(emissions, log_duration, trans, a, b, k, d_max))
                    total += segment + (0.0 if previous is None else trans[previous, k])
                    previous = k
                return total
            self.assertAlmostEqual(path_score(out), best, places=9)

    def test_deterministic(self):
        truth = truth_from(SPEC)
        nearest, seen, speed = synthetic_evidence(truth, np.random.RandomState(5))
        first = decoder.decode_half(nearest, seen, speed, synthetic_priors(), decoder.DEFAULTS, FPS)
        second = decoder.decode_half(nearest.copy(), seen.copy(), speed.copy(),
                                     synthetic_priors(), dict(decoder.DEFAULTS), FPS)
        np.testing.assert_array_equal(first, second)
        # Exact ties have one answer too.
        flat = decoder.viterbi(np.zeros((50, 3)), np.zeros((3, 10)), np.zeros((3, 3)))
        np.testing.assert_array_equal(flat, decoder.viterbi(np.zeros((50, 3)), np.zeros((3, 10)),
                                                            np.zeros((3, 3))))


def score_split(emissions, log_duration, trans, a, b, k, d_max):
    """Every way to cover blocks [a, b) with same-state segments of k; their scores."""
    if b - a <= d_max:
        yield emissions[a:b, k].sum() + log_duration[k, b - a - 1]
    for cut in range(a + 1, b):
        if cut - a <= d_max:
            head = emissions[a:cut, k].sum() + log_duration[k, cut - a - 1] + trans[k, k]
            for tail in score_split(emissions, log_duration, trans, cut, b, k, d_max):
                yield head + tail


class OnAMatch(unittest.TestCase):

    def test_copes_with_missing_players_and_ghost_slots(self):
        rng = np.random.RandomState(6)
        n, halves = 6000, [(0, 2800), (3000, 6000)]
        slots = 25                                          # 22 players plus 3 ghost slots
        x = (rng.rand(n, slots) * 105).astype(np.float32)
        y = (rng.rand(n, slots) * 68).astype(np.float32)
        x[rng.rand(n, slots) < 0.2] = np.nan
        x[1000:1100] = np.nan                               # nobody tracked at all
        team = np.array([0] * 11 + [1] * 11 + [0, 1, 1], dtype=np.int8)
        ball_x = (rng.rand(n) * 105).astype(np.float32)
        ball_x[2000:2600] = np.nan
        match = Match("synthetic", FPS, (105.0, 68.0), x, y, team, ball_x,
                      (rng.rand(n) * 68).astype(np.float32), halves, source="test")
        with mock.patch.object(decoder, "idsse_priors", lambda *args: synthetic_priors()):
            codes = methods.semi_markov(speed_scale="ms")(match)
            again = methods.semi_markov(speed_scale="ms")(match)
            ranked = methods.semi_markov()(match)              # the default reads ranks
        self.assertEqual(ranked.shape, (n,))
        np.testing.assert_array_equal(codes, again)
        self.assertEqual(codes.shape, (n,))
        self.assertTrue(np.all(codes[2800:3000] == pm.X))
        self.assertTrue(np.all(np.isin(codes[match.in_play_mask()], [pm.H, pm.A, pm.D])))

    def test_dead_min_s_lifts_only_the_stoppage_minimum(self):
        rng = np.random.RandomState(3)
        truth = truth_from([(H, 10), (D, 6), (A, 8)])
        nearest, seen, speed = synthetic_evidence(truth, rng)
        priors = synthetic_priors()                       # fitted dead minimum 2.0 s
        params = dict(decoder.DEFAULTS, speed_scale="ms")

        def banned(p):
            _, log_duration, _, _ = decoder.half_model(nearest, seen, speed, priors, p, FPS)
            return [int(np.sum(np.isinf(log_duration[k]))) for k in (H, A, D)]
        self.assertEqual(banned(params), banned(dict(params, dead_min_s=None)))
        self.assertEqual(banned(params)[D], 9)            # 2.0 s = 10 blocks: 9 entries banned
        self.assertEqual(banned(dict(params, dead_min_s=1.0)), banned(params))   # never lowered
        lifted = banned(dict(params, dead_min_s=4.0))
        self.assertEqual(lifted[D], 19)                   # 4.0 s = 20 blocks
        self.assertEqual(lifted[:2], banned(params)[:2])  # possession rows untouched
        states = decoder.decode_half(nearest, seen, speed, priors,
                                     dict(params, dead_min_s=4.0), FPS)
        for start, stop, state in runs(states):
            if state == D and stop < len(states):
                self.assertGreaterEqual(stop - start, 4.0 * FPS - FPS / 2)

    def test_unknown_parameter_is_refused(self):
        with self.assertRaises(ValueError):
            methods.semi_markov(swtich=2)

    def test_fit_priors_reads_durations_from_labels(self):
        n, halves = 3000, [(0, 3000)]
        labels = np.full(n, pm.H, dtype=np.int8)
        labels[1000:1500] = pm.D                            # 20 s of dead ball
        labels[1500:3000] = pm.A
        match = Match("synthetic", FPS, (105.0, 68.0), np.zeros((n, 2), np.float32),
                      np.zeros((n, 2), np.float32), np.array([0, 1], np.int8),
                      np.zeros(n, np.float32), np.zeros(n, np.float32), halves, labels)
        priors = decoder.fit_priors([match], lambda m: m.labels.copy(),
                                    lambda m: np.where(m.labels == pm.D, 1.0, 2.5))
        dead = priors["durations"]["dead"]
        self.assertAlmostEqual(dead["mu"], math.log(20.0))
        self.assertEqual(priors["durations"]["possession"]["n"], 2)
        self.assertEqual(priors["agree"], 1.0)
        fast, slow = decoder.speed_bins(np.array([2.5, 1.0]))
        self.assertGreater(priors["speed_log"][0, fast], priors["speed_log"][1, fast])
        self.assertGreater(priors["speed_log"][1, slow], priors["speed_log"][0, slow])


class SpeedRobustness(unittest.TestCase):

    def moving_match(self, seed=7):
        rng = np.random.RandomState(seed)
        n = 2000
        steps = rng.normal(0, 0.08, (n, 22)).astype(np.float32)
        x = 50 + np.cumsum(steps, axis=0)
        y = 30 + np.cumsum(steps[:, ::-1], axis=0)
        return Match("synthetic", FPS, (105.0, 68.0), x, y,
                     np.array([0] * 11 + [1] * 11, dtype=np.int8),
                     np.full(n, 50, np.float32), np.full(n, 30, np.float32), [(0, n)],
                     source="test")

    def test_rank_ignores_a_monotone_change_of_speed(self):
        match = self.moving_match()
        faster = match.copy()
        faster.player_x, faster.player_y = 2 * match.player_x, 2 * match.player_y
        rank = decoder.speed_feature(match, 2, "rank")
        np.testing.assert_allclose(rank, decoder.speed_feature(faster, 2, "rank"), atol=1e-6)
        known = np.isfinite(rank)
        self.assertTrue(np.all((rank[known] > 0) & (rank[known] <= 1)))

    def test_longer_lag_shrinks_the_noise_bias(self):
        from solution.simulate import add_position_noise
        match = self.moving_match()
        noisy = add_position_noise(match, 0.1, seed=0)
        np.testing.assert_array_equal(noisy.player_x, add_position_noise(match, 0.1, 0).player_x)
        np.testing.assert_array_equal(add_position_noise(match, 0.0).player_x, match.player_x)
        bias = {lag: np.nanmean(decoder.speed_feature(noisy, lag) - decoder.speed_feature(match, lag))
                for lag in (2, 12)}
        self.assertGreater(bias[2], 3 * bias[12])

if __name__ == "__main__":
    unittest.main()


class DeadModelTerm(unittest.TestCase):
    """deadm_w (section 16): the learned dead-ball log-ratio replaces the speed likelihood."""

    def setUp(self):
        rng = np.random.RandomState(4)
        self.truth = np.repeat(rng.randint(0, 3, 12), 50).astype(np.int8)
        self.nearest, self.seen, self.speed = synthetic_evidence(self.truth, rng)
        self.priors = synthetic_priors()

    def emissions(self, deadm, **params):
        return decoder.block_emissions(self.nearest, self.seen, self.speed, self.priors,
                                       dict(decoder.DEFAULTS, **params), deadm=deadm)

    def test_off_or_unread_is_unchanged(self):
        base = self.emissions(None)
        ratio = np.where(self.truth == 2, 2.0, -2.0)
        np.testing.assert_array_equal(self.emissions(ratio), base)          # deadm_w 0
        nothing = np.full(len(self.truth), np.nan)
        np.testing.assert_array_equal(self.emissions(nothing, deadm_w=1.0), base)

    def test_replaces_speed_and_adds_to_dead_only(self):
        ratio = np.where(self.truth == 2, 2.0, -2.0)
        no_speed = decoder.block_emissions(self.nearest, self.seen,
                                           np.full(len(self.truth), np.nan), self.priors,
                                           decoder.DEFAULTS)
        on = self.emissions(ratio, deadm_w=1.0)
        block_ratio = ratio.reshape(-1, decoder.BLOCK).mean(axis=1)
        np.testing.assert_allclose(on[:, decoder.H], no_speed[:, decoder.H])
        np.testing.assert_allclose(on[:, decoder.A], no_speed[:, decoder.A])
        np.testing.assert_allclose(on[:, decoder.D], no_speed[:, decoder.D] + block_ratio)

    def test_partial_blocks_mix_by_the_share_read(self):
        ratio = np.full(len(self.truth), np.nan)
        ratio[::5] = 1.0                       # one frame in five read
        base = self.emissions(None)
        no_speed = decoder.block_emissions(self.nearest, self.seen,
                                           np.full(len(self.truth), np.nan), self.priors,
                                           decoder.DEFAULTS)
        on = self.emissions(ratio, deadm_w=1.0)
        speed_part = base - no_speed
        np.testing.assert_allclose(on[:, decoder.H], no_speed[:, decoder.H] +
                                   0.8 * speed_part[:, decoder.H])
        np.testing.assert_allclose(on[:, decoder.D], no_speed[:, decoder.D] +
                                   0.8 * speed_part[:, decoder.D] + 0.2 * 1.0)

    def test_seen_only_keeps_speed_in_gaps(self):
        ratio = np.full(len(self.truth), 1.5)
        base = self.emissions(None)
        on = self.emissions(ratio, deadm_w=1.0, deadm_where="seen")
        seen = self.seen.reshape(-1, decoder.BLOCK)
        hidden = ~seen.any(axis=1)
        np.testing.assert_array_equal(on[hidden], base[hidden])     # gap blocks: unchanged
        full = seen.all(axis=1)
        everywhere = self.emissions(ratio, deadm_w=1.0)
        np.testing.assert_allclose(on[full], everywhere[full])      # fully seen: as "all"
        with self.assertRaises(ValueError):
            self.emissions(ratio, deadm_w=1.0, deadm_where="gaps")

