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


if __name__ == "__main__":
    unittest.main()
