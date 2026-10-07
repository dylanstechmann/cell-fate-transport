from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cellfate.pipeline import infer
from cellfate.transport import transport, unbalanced_transport

# Two well-separated states. Between snapshots state A triples while B is unchanged,
# so the marginals a balanced solver is handed are inconsistent with the real lineage.
N_A_SOURCE, N_B_SOURCE = 20, 20
N_A_TARGET, N_B_TARGET = 60, 20


def fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    source = np.vstack([rng.normal(0.0, 0.2, (N_A_SOURCE, 2)),
                        rng.normal(10.0, 0.2, (N_B_SOURCE, 2))])
    target = np.vstack([rng.normal(0.0, 0.2, (N_A_TARGET, 2)),
                        rng.normal(10.0, 0.2, (N_B_TARGET, 2))])
    source_is_a = np.arange(len(source)) < N_A_SOURCE
    target_is_a = np.arange(len(target)) < N_A_TARGET
    return source, target, source_is_a, target_is_a


class BalancedAssumptionTests(unittest.TestCase):
    def test_balanced_transport_invents_a_transfer_that_did_not_happen(self):
        source, target, source_is_a, target_is_a = fixture()
        result = transport(source, target, epsilon=0.5)
        b_to_a = result.transition[~source_is_a][:, target_is_a].sum(axis=1)
        # Fixed marginals force each B cell to send about half its mass to A'.
        self.assertGreater(b_to_a.mean(), 0.4)
        self.assertAlmostEqual(b_to_a.mean(), 0.5, delta=0.05)
        a_to_a = result.transition[source_is_a][:, target_is_a].sum(axis=1)
        self.assertGreater(a_to_a.mean(), 0.99)

    def test_relaxing_the_marginals_recovers_the_true_structure(self):
        source, target, source_is_a, target_is_a = fixture()
        result = unbalanced_transport(source, target, epsilon=0.5, growth_relaxation=1.0)
        b_to_a = result.transition[~source_is_a][:, target_is_a].sum(axis=1)
        self.assertLess(b_to_a.max(), 1e-6)
        a_to_a = result.transition[source_is_a][:, target_is_a].sum(axis=1)
        self.assertGreater(a_to_a.min(), 0.99)
        # Rows remain conditional distributions even though their mass moved.
        np.testing.assert_allclose(result.transition.sum(axis=1), 1.0, atol=1e-10)

    def test_implied_mass_change_points_the_right_way_but_is_not_a_rate(self):
        source, target, source_is_a, _target_is_a = fixture()
        ratios = []
        for relaxation in (10.0, 1.0, 0.2):
            result = unbalanced_transport(source, target, epsilon=0.5,
                                          growth_relaxation=relaxation)
            change = result.implied_relative_mass_change
            grown, flat = change[source_is_a].mean(), change[~source_is_a].mean()
            self.assertGreater(grown, flat)
            ratios.append(grown / flat)
        # The direction is stable; the magnitude is not, which is why it is reported
        # as model-implied rather than as a growth rate. The true ratio is 3.
        self.assertTrue(all(ratio > 1.0 for ratio in ratios))
        self.assertGreater(max(ratios) - min(ratios), 0.2)
        self.assertLess(min(ratios), 3.0)

    def test_large_relaxation_approaches_balanced_transport(self):
        source, target, _source_is_a, _target_is_a = fixture()
        balanced = transport(source, target, epsilon=0.5)
        # Marginal deviation shrinks as the penalty grows, approaching the balanced
        # solution in the limit; the deviation scales roughly with epsilon / tau.
        deviations = []
        for relaxation in (100.0, 1000.0, 50000.0):
            result = unbalanced_transport(source, target, epsilon=0.5,
                                          growth_relaxation=relaxation)
            deviations.append(float(np.abs(result.implied_relative_mass_change - 1).max()))
        self.assertEqual(deviations, sorted(deviations, reverse=True))
        self.assertLess(deviations[-1], 5e-3)
        nearly = unbalanced_transport(source, target, epsilon=0.5, growth_relaxation=50000.0)
        np.testing.assert_allclose(nearly.realized_source_mass, balanced.source_mass, atol=5e-3)

    def test_unbalanced_rejects_the_same_malformed_inputs_as_balanced(self):
        source, target, _a, _b = fixture()
        for kwargs in ({"epsilon": 0.0}, {"epsilon": -1.0}, {"growth_relaxation": 0.0},
                       {"growth_relaxation": -2.0}, {"growth_relaxation": float("nan")},
                       {"tolerance": 0.0}, {"tolerance": 1.0}, {"max_iterations": 0},
                       {"max_pairs": 0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    unbalanced_transport(source, target, **kwargs)
        with self.assertRaises(ValueError):
            unbalanced_transport(np.zeros((2, 2)), np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            unbalanced_transport(source, target, source_mass=np.zeros(len(source)))


class PipelineDefaultTests(unittest.TestCase):
    def table(self, path: Path) -> None:
        source, target, _a, _b = fixture(seed=1)
        rows = []
        for index, point in enumerate(source):
            rows.append({"cell_id": f"s{index}", "time": 0.0, "state": "progenitor",
                         "f_x": point[0], "f_y": point[1]})
        for index, point in enumerate(target):
            state = "fate_a" if index < N_A_TARGET else "fate_b"
            rows.append({"cell_id": f"t{index}", "time": 1.0, "state": state,
                         "f_x": point[0], "f_y": point[1]})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_default_run_is_balanced_and_says_so(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.table(root / "cells.csv")
            report, _rows = infer(root / "cells.csv", root / "balanced")
            self.assertIsNone(report["configuration"]["growth_relaxation"])
            self.assertIn("balanced", report["configuration"]["marginals"])
            self.assertIn("marginal_l1_error", report["maps"][0])
            self.assertNotIn("implied_relative_mass_change_median", report["maps"][0])
            self.assertTrue(any("cannot estimate cell proliferation" in item
                                for item in report["warnings"]))

    def test_opt_in_run_records_the_relaxation_and_its_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.table(root / "cells.csv")
            report, _rows = infer(root / "cells.csv", root / "relaxed", growth_relaxation=1.0)
            self.assertEqual(report["configuration"]["growth_relaxation"], 1.0)
            self.assertIn("KL-relaxed", report["configuration"]["marginals"])
            entry = report["maps"][0]
            self.assertEqual(entry["marginals"], "kl_relaxed")
            self.assertGreater(entry["implied_relative_mass_change_max"],
                               entry["implied_relative_mass_change_min"])
            self.assertIn("not a measured growth", entry["implied_mass_change_interpretation"])
            self.assertTrue(any("model artifact" in item for item in report["warnings"]))
            self.assertFalse(any("cannot estimate cell proliferation" in item
                                 for item in report["warnings"]))
            json.dumps(report, allow_nan=False)

    def test_invalid_relaxation_is_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.table(root / "cells.csv")
            with self.assertRaisesRegex(ValueError, "growth_relaxation"):
                infer(root / "cells.csv", root / "bad", growth_relaxation=0.0)
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
