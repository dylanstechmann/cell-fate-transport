import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

from cellfate.pipeline import infer, read_cells
from cellfate.transport import probability_mass, pull_back_fates, transport


class TransportTests(unittest.TestCase):
    def test_marginals_and_stochastic_rows_for_unequal_populations(self):
        fit = transport([[0], [1]], [[0], [0.5], [1]], source_mass=[2, 1], target_mass=[1, 2, 3])
        np.testing.assert_allclose(fit.coupling.sum(axis=1), [2/3, 1/3], atol=1e-8)
        np.testing.assert_allclose(fit.coupling.sum(axis=0), [1/6, 2/6, 3/6], atol=1e-8)
        np.testing.assert_allclose(fit.transition.sum(axis=1), 1, atol=1e-12)
        self.assertLessEqual(fit.marginal_l1_error, 1e-8)

    def test_zero_cost_returns_independent_coupling(self):
        fit = transport(np.zeros((2, 3)), np.zeros((3, 3)), source_mass=[3, 1], target_mass=[1, 1, 2])
        np.testing.assert_allclose(fit.coupling, np.outer([.75, .25], [.25, .25, .5]), atol=1e-10)

    def test_small_entropy_agrees_with_independent_linear_program(self):
        x, y = np.array([[0.], [1.], [2.]]), np.array([[.1], [1.1], [2.1]])
        costs = cdist(x, y, "sqeuclidean")
        constraints = np.vstack([np.kron(np.eye(3), np.ones((1, 3))), np.kron(np.ones((1, 3)), np.eye(3))])
        reference = linprog(costs.ravel(), A_eq=constraints, b_eq=np.ones(6)/3, bounds=(0, None), method="highs")
        self.assertTrue(reference.success)
        fit = transport(x, y, epsilon=.02)
        self.assertAlmostEqual(fit.transport_cost, reference.fun, places=6)

    def test_permutation_equivariance(self):
        x, y = np.array([[0.], [.4], [1.]]), np.array([[.1], [.8]])
        a = transport(x, y)
        b = transport(x[[2, 0, 1]], y[[1, 0]])
        np.testing.assert_allclose(b.coupling, a.coupling[[2, 0, 1]][:, [1, 0]], atol=1e-9)

    def test_log_domain_handles_large_distances(self):
        fit = transport([[0.], [1000.]], [[0.], [1000.]], epsilon=.1)
        np.testing.assert_allclose(fit.coupling, [[.5, 0], [0, .5]], atol=1e-10)

    def test_markov_fate_composition_has_known_answer(self):
        p = np.array([[.8, .2], [.1, .9]])
        q = np.array([[1., 0], [.25, .75]])
        names, fates = pull_back_fates([p, q], ["A", "B"])
        self.assertEqual(names, ["A", "B"])
        np.testing.assert_allclose(fates[0], p @ q)
        np.testing.assert_array_equal(fates[-1], np.eye(2))
        with self.assertRaises(ValueError):
            pull_back_fates([p * 2], ["A", "B"])

    def test_reject_invalid_inputs_limits_and_nonconvergence(self):
        for args in [dict(epsilon=0), dict(epsilon=float("nan")), dict(max_pairs=1), dict(tolerance=1), dict(max_iterations=True)]:
            with self.assertRaises(ValueError):
                transport([[0], [1]], [[0], [1]], **args)
        for masses in [[0, 1], [-1, 1], [float("nan"), 1], [1]]:
            with self.assertRaises(ValueError):
                probability_mass(masses, 2)
        with self.assertRaises(ValueError):
            transport([[float("inf")]], [[1]])
        with self.assertRaises(RuntimeError):
            transport([[0], [1]], [[0], [2]], epsilon=.5, tolerance=1e-14, max_iterations=1)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.path = self.root / "cells.csv"
        self.path.write_text("cell_id,time,state,batch_id,f_x\na,0,hidden,one,0\nb,0,hidden,one,1\nc,1,A,two,0\nd,1,B,two,1\n")

    def test_pipeline_records_id_alignment_and_ignores_early_labels(self):
        report, rows = infer(self.path, self.root / "first")
        self.assertEqual(report["features"], ["f_x"])
        self.assertTrue(any("confounded" in warning for warning in report["warnings"]))
        with np.load(self.root / "first/transport_00.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["source_ids"], ["a", "b"])
            np.testing.assert_array_equal(archive["target_ids"], ["c", "d"])
        self.path.write_text(self.path.read_text().replace("hidden", "different-label"))
        _, again = infer(self.path, self.root / "second")
        self.assertEqual([r["p_state_0"] for r in rows], [r["p_state_0"] for r in again])
        with self.assertRaises(FileExistsError):
            infer(self.path, self.root / "first")

    def test_input_hash_identifies_the_parsed_snapshot(self):
        original = self.path.read_bytes()
        read_bytes = Path.read_bytes
        reads = []

        def replace_after_read(path):
            data = read_bytes(path)
            reads.append(path)
            path.write_text("changed after reading\n", encoding="utf-8")
            return data

        with patch.object(Path, "read_bytes", replace_after_read):
            rows, features, times, x, mass, digest = read_cells(self.path)
        self.assertEqual(reads, [self.path])
        self.assertEqual(digest, hashlib.sha256(original).hexdigest())
        self.assertEqual(len(rows), 4)
        self.assertEqual(features, ["f_x"])
        self.assertNotEqual(self.path.read_bytes(), original)

    def test_reject_bad_csv_and_missing_terminal_labels(self):
        for text in ["cell_id,time,state,f_x\na,0,A,1\na,1,B,2\n",
                     "cell_id,time,state,f_x\na,0,A,1\nb,1,,2\n",
                     "cell_id,time,state,f_x\na,nan,A,1\nb,1,B,2\n",
                     "cell_id,time,state,f_x\na,0,A,1,extra\nb,1,B,2\n",
                     "cell_id,time,state,f_x\na,0,A,nan\nb,1,B,2\n"]:
            self.path.write_text(text)
            with self.assertRaises(ValueError): read_cells(self.path)

    def test_compose_transitions_and_sankey_generation(self):
        from cellfate.transport import build_sankey_data, compose_transitions
        t0 = np.array([[0.8, 0.2], [0.3, 0.7]])
        t1 = np.array([[0.6, 0.4], [0.1, 0.9]])
        chained = compose_transitions([t0, t1])
        np.testing.assert_allclose(chained, t0 @ t1)
        np.testing.assert_allclose(chained.sum(axis=1), [1.0, 1.0])

        with self.assertRaises(ValueError):
            compose_transitions([])
        with self.assertRaises(ValueError):
            compose_transitions([t0, np.ones((3, 3))])
        for invalid in [np.zeros((2, 2)), np.array([[0.2, 0.2], [0.5, 0.5]])]:
            with self.assertRaisesRegex(ValueError, "row-stochastic"):
                compose_transitions([invalid])
            with self.assertRaisesRegex(ValueError, "row-stochastic"):
                compose_transitions([t0, invalid])

        states = [["early", "early"], ["mid_A", "mid_B"], ["term_A", "term_B"]]
        times = [0.0, 1.0, 2.0]
        sankey = build_sankey_data(None, [t0, t1], states, times)
        self.assertEqual(len(sankey["nodes"]), 5)
        self.assertGreater(len(sankey["links"]), 0)
        # Sum of flows from t0->t1 must equal 1.0
        t0_links = [l["value"] for l in sankey["links"] if l["source_time"] == 0.0]
        self.assertAlmostEqual(sum(t0_links), 1.0, places=5)

    def test_multi_timepoint_chain_and_sankey_pipeline_export(self):
        # 3 snapshots: t=0, 1, 2
        csv_text = (
            "cell_id,time,state,f_x\n"
            "c0_1,0,early,0.0\n"
            "c0_2,0,early,0.5\n"
            "c1_1,1,mid,1.0\n"
            "c1_2,1,mid,1.5\n"
            "c2_1,2,terminal_A,2.0\n"
            "c2_2,2,terminal_B,2.5\n"
        )
        chain_csv = self.root / "chain_cells.csv"
        chain_csv.write_text(csv_text)
        out_dir = self.root / "chain_out"
        report, rows = infer(chain_csv, out_dir)

        # Check composed_chain.npz
        self.assertTrue((out_dir / "composed_chain.npz").is_file())
        with np.load(out_dir / "composed_chain.npz", allow_pickle=False) as data:
            self.assertIn("composed_transition", data)
            self.assertIn("composed_coupling", data)
            np.testing.assert_array_equal(data["source_ids"], ["c0_1", "c0_2"])
            np.testing.assert_array_equal(data["target_ids"], ["c2_1", "c2_2"])

        # Check sankey.json
        self.assertTrue((out_dir / "sankey.json").is_file())
        import json
        sankey = json.loads((out_dir / "sankey.json").read_text())
        self.assertIn("nodes", sankey)
        self.assertIn("links", sankey)
        self.assertEqual(len(sankey["nodes"]), 4)  # early, mid, terminal_A, terminal_B

        # Check report.json
        self.assertIn("chain_composition", report)
        self.assertIn("sankey", report)


if __name__ == "__main__":
    unittest.main()
