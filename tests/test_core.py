"""Deterministic tests that need no torch (pure numpy / stdlib)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jev import prompts  # noqa: E402
from jev.metrics import accuracy, brier, ece, order_consistency  # noqa: E402
from jev.prompts import make_views  # noqa: E402
from jev.schema import DecisionItem  # noqa: E402


class TestSchema(unittest.TestCase):
    def test_roundtrip(self):
        it = DecisionItem("x", "s", "q", ["a", "b"], gold=1)
        self.assertEqual(DecisionItem.from_json(it.to_json()), it)

    def test_gold_bounds(self):
        with self.assertRaises(ValueError):
            DecisionItem("x", "s", "q", ["a"], gold=3)


class TestPrompts(unittest.TestCase):
    def test_views_cover_every_position(self):
        views = make_views(3, n_paraphrases=2)
        rotations = {v.order for v in views if v.paraphrase_id == 0}
        self.assertIn((0, 1, 2), rotations)
        self.assertIn((1, 2, 0), rotations)
        self.assertIn((2, 0, 1), rotations)
        self.assertEqual(len(views), 2 * 3)

    def test_label_offsets_match_letters(self):
        prompt, letters, offsets, ends = prompts.build_prompt(
            "state", "question", ["first", "second"], paraphrase_id=0
        )
        self.assertEqual(letters, ["A", "B"])
        for letter, off in zip(letters, offsets):
            self.assertEqual(prompt[off], letter)
        # text-end offsets point at the last character of each option text
        self.assertEqual(prompt[ends[0]], "t")  # "first"
        self.assertEqual(prompt[ends[1]], "d")  # "second"
        self.assertTrue(all(e > o for e, o in zip(ends, offsets)))


class TestMetrics(unittest.TestCase):
    def test_accuracy_and_ece(self):
        probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]])
        gold = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(accuracy(probs, gold), 0.5)
        self.assertGreaterEqual(ece(probs, gold), 0.0)
        self.assertGreaterEqual(brier(probs, gold), 0.0)

    def test_temperature_scaling_reduces_ece(self):
        from jev.calibrate import apply_temperature, fit_temperature

        # overconfident and only 50% correct
        logits = np.array([[4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [-4.0, 4.0]])
        gold = np.array([0, 1, 0, 1])
        before = ece(apply_temperature(logits, 1.0), gold)
        temp = fit_temperature(logits, gold)
        after = ece(apply_temperature(logits, temp), gold)
        self.assertGreater(temp, 1.0)
        self.assertLess(after, before)

    def test_order_consistency_detects_disagreement(self):
        vp = np.array([[[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]], [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]])
        out = order_consistency(vp)
        self.assertAlmostEqual(out["argmax_agreement"], 0.5)

    def test_gold_none_is_ignored(self):
        probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        gold = np.array([0, -1])
        self.assertAlmostEqual(accuracy(probs, gold), 1.0)


class TestDebias(unittest.TestCase):
    def _biased_views(self):
        """Three cyclic views with a true option score plus an additive slot bias."""
        k = 3
        c = np.array([2.0, 1.0, 0.0])
        b = np.array([3.0, 0.0, -3.0])  # the model loves slot 0, hates slot 2
        vp, vo = [], []
        for s in range(k):
            order = [(j + s) % k for j in range(k)]  # option shown at each slot
            z = c[order] + b
            p = np.exp(z - z.max())
            p = p / p.sum()
            canon = np.zeros(k)
            canon[order] = p
            vp.append(canon)
            vo.append(order)
        return np.array([vp]), np.array([vo])  # [1, V, K]

    def test_position_roundtrip(self):
        from jev.debias import position_to_view, view_to_position

        vp = np.random.default_rng(0).dirichlet(np.ones(4), size=(5, 3))
        order = np.stack([np.random.default_rng(i).permutation(4) for i in range(5 * 3)]).reshape(5, 3, 4)
        back = position_to_view(view_to_position(vp, order), order)
        np.testing.assert_allclose(back, vp, atol=1e-12)

    def test_logmean_cancels_additive_slot_bias(self):
        from jev.debias import debias, flip_rate

        vp, order = self._biased_views()
        true = np.exp(np.array([2.0, 1.0, 0.0]))
        true = true / true.sum()
        logmean = debias(vp, order, combine="logmean")[0]
        mean = debias(vp, order, combine="mean")[0]
        # raw per-view readout is maximally order-sensitive
        self.assertAlmostEqual(flip_rate(vp, order), 1.0)
        # a geometric mean cancels the additive slot bias exactly
        np.testing.assert_allclose(logmean, true, atol=2e-3)
        # the arithmetic mean keeps a residual slot bias
        self.assertGreater(np.abs(mean - true).max(), 0.1)

    def test_batch_prior_is_label_free_and_normalized(self):
        from jev.debias import apply_prior, batch_prior

        vp, _order = self._biased_views()
        prior = batch_prior(np.repeat(vp, 4, axis=0))
        self.assertEqual(prior.shape, (3, 3))
        np.testing.assert_allclose(prior.sum(axis=-1), np.ones(3), atol=1e-9)
        fixed = apply_prior(vp, prior)
        np.testing.assert_allclose(fixed.sum(axis=-1), np.ones((1, 3)), atol=1e-9)


    def test_content_free_prior_per_item(self):
        from jev.debias import content_free_prior

        cf = np.full((2, 3, 2, 4), 0.25)  # [N, V, P, K]
        prior = content_free_prior(cf)
        self.assertEqual(prior.shape, (2, 3, 4))
        np.testing.assert_allclose(prior, 0.25 * np.ones((2, 3, 4)), atol=1e-12)


class TestCoverage(unittest.TestCase):
    def test_coverage_at_risk(self):
        from jev.metrics import coverage_at_risk

        # 4 perfectly confident and correct, 6 confident and wrong
        probs = np.array([[0.9, 0.1]] * 4 + [[0.8, 0.2]] * 6)
        gold = np.array([0] * 4 + [1] * 6)
        self.assertAlmostEqual(coverage_at_risk(probs, gold, target_risk=0.0), 0.4)
        self.assertAlmostEqual(coverage_at_risk(probs, gold, target_risk=0.5), 0.8)
        self.assertAlmostEqual(coverage_at_risk(probs, gold, target_risk=-0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
