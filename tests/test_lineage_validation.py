import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from cellfate.lineage_validation import (
    clone_is_test, fit_expression_space, normalized_log_expression,
    read_expression_rows, read_larry_header, run_larry, score_clones,
    select_cohort, sha256, transform_expression,
)


class CloneEvaluationTests(unittest.TestCase):
    def test_frequency_outcomes_equal_clone_weights_and_paired_interval(self):
        truth = np.array([[1., 0.], [0.25, 0.75]])
        result = score_clones(truth, truth, [0.5, 0.5], bootstrap=400)
        self.assertEqual(result["transport_brier"]["estimate"], 0)
        self.assertAlmostEqual(result["training_prior_brier"]["estimate"], 0.3125)
        self.assertGreater(result["paired_brier_improvement"]["ci95"][0], 0)
        self.assertEqual(result, score_clones(truth, truth, [0.5, 0.5], bootstrap=400))
        with self.assertRaises(ValueError):
            score_clones([[np.nan, 0]], [[1, 0]], [0.5, 0.5])
        with self.assertRaises(ValueError):
            score_clones([[1, 1]], [[1, 0]], [0.5, 0.5])

    def test_training_only_transform_and_library_scale_invariance(self):
        values = sparse.csr_matrix([[1., 8., 0.], [3., 1., 4.], [0., 5., 5.], [5., 4., 1.]])
        training = normalized_log_expression(values)
        fitted = fit_expression_space(training, n_genes=3, n_components=2)
        earlier = transform_expression(training, fitted)
        transformed = transform_expression(normalized_log_expression(values * 5), fitted)
        np.testing.assert_allclose(earlier, transformed, atol=1e-12)
        untouched = {k: v.copy() for k, v in fitted.items()}
        transform_expression(normalized_log_expression(sparse.csr_matrix([[9000., 0., 1.]])), fitted)
        for key in fitted:
            np.testing.assert_array_equal(fitted[key], untouched[key])
        with self.assertRaises(ValueError):
            normalized_log_expression(sparse.csr_matrix([[0., 0.]]))
        with self.assertRaises(ValueError):
            normalized_log_expression(sparse.csr_matrix([[5., -1.]]))

    def test_selection_is_whole_clone_and_independent_of_states(self):
        clones = np.repeat(np.arange(30), 4)
        times = np.tile([2., 4., 6., 6.], 30)
        data = {"clones": clones, "times": times, "ids": np.array([f"c{i}" for i in range(120)]),
                "states": np.repeat("ignored", 120)}
        cohort = select_cohort(data, max_clones=3, training_per_day=10)
        self.assertFalse(set(clones[cohort["training"]]) & set(cohort["clones"]))
        self.assertTrue(all(clone_is_test(c) for c in cohort["clones"]))
        data["states"][:] = "changed"
        second = select_cohort(data, max_clones=3, training_per_day=10)
        np.testing.assert_array_equal(cohort["training"], second["training"])
        np.testing.assert_array_equal(cohort["clones"], second["clones"])
        self.assertFalse(set(cohort["truth"]) & set(cohort["reference"]))


@unittest.skipUnless(importlib.util.find_spec("h5py"), "optional validation dependency h5py")
class LarryImportTests(unittest.TestCase):
    def fixture(self, path):
        import h5py

        clones = np.repeat(np.arange(30), 4)
        n = len(clones)
        values = sparse.csr_matrix(np.random.default_rng(40).integers(1, 10, (n, 8)).astype(float))
        membership = sparse.csr_matrix((np.ones(n), (np.arange(n), clones)), shape=(n, 30))
        with h5py.File(path, "w") as handle:
            for name, matrix in (("X", values), ("obsm/X_clone", membership)):
                group = handle.create_group(name)
                group.attrs["encoding-type"] = "csr_matrix"
                group.attrs["shape"] = matrix.shape
                for key in ("data", "indices", "indptr"):
                    group.create_dataset(key, data=getattr(matrix, key))
            obs = handle.create_group("obs")
            obs.create_dataset("_index", data=np.array([f"id{i}".encode() for i in range(n)]))
            obs.create_dataset("time_info", data=np.tile([0, 1, 2, 2], 30))
            obs.create_dataset("state_info", data=clones % 2)
            categories = obs.create_group("__categories")
            categories.create_dataset("time_info", data=np.array([b"2", b"4", b"6"]))
            categories.create_dataset("state_info", data=np.array([b"A", b"B"]))
            handle.create_dataset("var/_index", data=np.array([f"g{i}".encode() for i in range(8)]))
            # An inferred field must never be consumed by this importer.
            obs.create_dataset("MLPClassifier_predicted_bias", data=np.full(n, np.nan))
            handle.create_dataset("obsm/X_pca", data=np.full((n, 2), np.nan))
        return values

    def test_subset_import_ignores_inferred_fields_and_rejects_invalid_membership(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.h5ad"
            values = self.fixture(path)
            data = read_larry_header(path)
            self.assertEqual(data["shape"], values.shape)
            rows = np.array([0, 6, 100])
            np.testing.assert_array_equal(read_expression_rows(path, rows).toarray(), values[rows].toarray())
            with self.assertRaises(ValueError):
                read_expression_rows(path, [6, 0])
            with h5py.File(path, "r+") as handle:
                handle["obsm/X_clone/data"][0] = 0
            with self.assertRaises(ValueError):
                read_larry_header(path)

    def test_end_to_end_withholds_terminal_test_cells_and_excludes_clone_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.h5ad"
            self.fixture(path)
            output = Path(directory) / "audit"
            with patch("cellfate.lineage_validation.LARRY_SHA256", sha256(path)):
                result = run_larry(path, output)
            data = read_larry_header(path)
            test_clones = set(result["selected_test_clones"])
            with (output / "cells.csv").open() as handle:
                reader = csv.DictReader(handle)
                self.assertFalse(any("clone" in c for c in reader.fieldnames))
                rows = list(reader)
            id_to_index = {value: i for i, value in enumerate(data["ids"])}
            for row in rows:
                index = id_to_index[row["cell_id"]]
                if float(row["time"]) == 6:
                    self.assertNotIn(data["clones"][index], test_clones)
                else:
                    self.assertEqual(row["state"], "")
            self.assertEqual(set(result["metrics_by_source_day"]), {"2", "4"})
            self.assertTrue(all(m["marginal_l1_error"] <= 1e-8 for m in result["maps"]))
            with self.assertRaises(FileExistsError):
                run_larry(path, output)
            with self.assertRaises(ValueError):
                run_larry(path, Path(directory) / "invalid-hash")


if __name__ == "__main__":
    unittest.main()
