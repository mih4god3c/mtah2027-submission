"""#6 Step 0: the posterior sampler (decoder.forward_sums / backward_sample) and the fast spellF1
of tools/mbr_check.py. No data needed."""

import itertools
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder  # noqa: E402
from solution.match import pm  # noqa: E402
from tools.mbr_check import merge_spells, spell_arrays, spell_f1  # noqa: E402

H, A, D = decoder.H, decoder.A, decoder.D


def tiny_model(seed=0, n=6, d_max=3, with_bonus=True):
    rng = np.random.RandomState(seed)
    emissions = rng.normal(0.0, 1.0, (n, 3))
    log_duration = np.log(rng.dirichlet(np.ones(d_max), size=3))
    log_duration[2, 0] = -np.inf                          # D needs at least 2 blocks
    trans = rng.normal(0.0, 0.5, (3, 3))
    np.fill_diagonal(trans, 0.0)
    bonus = rng.normal(0.0, 0.7, (n, 2)) if with_bonus else None
    return emissions, log_duration, trans, bonus


def segmentations(n, d_max):
    """Every [(state, start, length)] covering n blocks."""
    if n == 0:
        yield []
        return
    for d in range(1, min(d_max, n) + 1):
        for rest in segmentations(n - d, d_max):
            for k in (H, A, D):
                yield rest + [(k, n - d, d)]


def brute_force(model):
    """{labelling tuple: log total score}, summed over the segmentations giving it."""
    emissions, log_duration, trans, bonus = model
    n, d_max = emissions.shape[0], log_duration.shape[1]
    out = {}
    for segs in segmentations(n, d_max):
        score, labels, prev = 0.0, [], None
        for k, start, d in segs:
            score += emissions[start:start + d, k].sum() + log_duration[k, d - 1]
            if prev is not None:
                score += trans[prev, k]
                if bonus is not None and (prev, k) == (A, H):
                    score += bonus[start, 0]
                if bonus is not None and (prev, k) == (H, A):
                    score += bonus[start, 1]
            labels += [k] * d
            prev = k
        if not np.isfinite(score):
            continue
        key = tuple(labels)
        out[key] = np.logaddexp(out.get(key, -np.inf), score)
    return out


class MBRTest(unittest.TestCase):
    def test_forward_sums_equals_brute_force(self):
        for seed in range(3):
            model = tiny_model(seed)
            _, total, _ = decoder.forward_sums(*model)
            expected = np.logaddexp.reduce(list(brute_force(model).values()))
            assert math.isclose(np.logaddexp.reduce(total[-1]), expected, rel_tol=1e-9)


    def test_samples_follow_the_posterior(self):
        model = tiny_model(1)
        exact = brute_force(model)
        norm = np.logaddexp.reduce(list(exact.values()))
        sums = decoder.forward_sums(*model)
        rng = np.random.RandomState(0)
        n_draws = 20000
        counts = {}
        for _ in range(n_draws):
            key = tuple(decoder.backward_sample(sums, model[1], model[2], model[3], rng))
            counts[key] = counts.get(key, 0) + 1
        assert set(counts) <= set(exact)
        for key, log_score in exact.items():
            p = math.exp(log_score - norm)
            if p > 0.01:
                observed = counts.get(key, 0) / float(n_draws)
                assert abs(observed - p) < 4 * math.sqrt(p * (1 - p) / n_draws) + 1e-3, key


    def test_cold_samples_are_viterbi_and_seeded(self):
        model = tiny_model(2)
        best = decoder.viterbi(*model)
        cold = decoder.sample_half(model, 6 * decoder.BLOCK, 5, np.random.RandomState(0), 1e-3)
        assert (cold == np.repeat(best, decoder.BLOCK)[None, :]).all()
        first = decoder.sample_half(model, 30, 7, np.random.RandomState(3), 2.0)
        again = decoder.sample_half(model, 30, 7, np.random.RandomState(3), 2.0)
        assert np.array_equal(first, again)


    def test_fast_spell_f1_matches_the_scorer(self):
        rng = np.random.RandomState(0)
        n = 3000
        for _ in range(40):
            def labels():
                codes = np.repeat(rng.choice([pm.H, pm.A, pm.D], 120, p=[0.4, 0.4, 0.2]),
                                  rng.randint(1, 50, 120))[:n].astype(np.int8)
                codes = np.concatenate((codes, np.full(max(0, n - len(codes)), pm.H, np.int8)))
                codes[:100] = pm.X
                codes[1500:1700] = pm.X
                return codes
            truth, pred = labels(), labels()
            pred[truth == pm.X] = pm.X
            runs = pm.valid_runs(truth)
            official = pm.evaluate(truth, pred)["spell_f1_mean"]
            assert math.isclose(spell_f1(spell_arrays(truth, runs), spell_arrays(pred, runs)),
                                official, abs_tol=1e-12)


    def test_spell_f1_identity(self):
        codes = np.array([pm.H] * 10 + [pm.A] * 5 + [pm.D] * 3 + [pm.H] * 7, dtype=np.int8)
        spells = spell_arrays(codes, [(0, len(codes))])
        assert spell_f1(spells, spells) == 1.0


    def test_merge_spells_absorbs_into_a_neighbour(self):
        H_, A_, D_ = pm.H, pm.A, pm.D
        codes = np.array([H_] * 4 + [A_] * 2 + [H_] * 3 + [D_] * 2 + [A_] * 2 + [D_] * 2 +
                         [A_] * 3, dtype=np.int8)
        runs = [(0, len(codes))]
        spells = spell_arrays(codes, runs)          # H, A, H, A, A
        drop = np.array([False, True, False, True, False])
        out = merge_spells(codes, spells, drop, runs)
        self.assertEqual(out[4:6].tolist(), [H_, H_])   # H A H -> H H H
        self.assertEqual(out[11:13].tolist(), [D_, D_])  # D A D -> D D D
        self.assertEqual(len(spell_arrays(out, runs)[0]), 2)


if __name__ == "__main__":
    unittest.main()
