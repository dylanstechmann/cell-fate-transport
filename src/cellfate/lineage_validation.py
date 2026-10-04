"""Held-out-clone audit of balanced transport on public, measured LARRY data.

The optional h5py dependency reads an author-provided H5AD without importing
CoSpar, its inferred fate fields, its embeddings or serialized Python objects.
The experiment protocol is docs/LARRY_VALIDATION_PLAN.md.
"""

import csv
import hashlib
import json
from pathlib import Path
import platform
import urllib.request

import numpy as np
import scipy
from scipy import linalg, sparse

from cellfate.pipeline import infer


LARRY_URL = "https://kleintools.hms.harvard.edu/tools/downloads/cospar/LARRY_adata_preprocessed.h5ad"
LARRY_SHA256 = "16607046cd2550dbcdb33c24c58e5b8852c5558f7b52a437e0ae20b0137ef881"
PAPER_DOI = "https://doi.org/10.1126/science.aaw3381"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(prefix, value):
    return hashlib.sha256(f"{prefix}:{value}".encode()).hexdigest()


def clone_is_test(clone):
    """A stable whole-clone split, independent of expression and fate."""
    return int(_rank("larry-split-v1", clone)[:8], 16) % 5 == 0


def _strings(values):
    return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def _categorical(group, name):
    if "__categories" in group and name in group["__categories"]:
        categories = _strings(group["__categories"][name][:])
        codes = group[name][:]
    elif hasattr(group[name], "keys"):
        categories = _strings(group[name]["categories"][:])
        codes = group[name]["codes"][:]
    else:
        return _strings(group[name][:])
    if not np.issubdtype(codes.dtype, np.integer) or np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError(f"invalid or missing categorical values for {name}")
    return categories[codes]


def read_larry_header(path):
    """Read only measured fields and single-clone memberships; never embeddings."""
    import h5py

    with h5py.File(path, "r") as handle:
        x = handle["X"]
        shape = tuple(map(int, x.attrs["shape"]))
        if x.attrs.get("encoding-type") != "csr_matrix" or len(shape) != 2:
            raise ValueError("expected CSR cell-by-gene expression")
        ids = _strings(handle["obs/_index"][:])
        genes = _strings(handle["var/_index"][:])
        times = _categorical(handle["obs"], "time_info").astype(float)
        states = _categorical(handle["obs"], "state_info")
        membership = handle["obsm/X_clone"]
        clone_shape = tuple(map(int, membership.attrs["shape"]))
        ptr = membership["indptr"][:]
        columns = membership["indices"][:]
        values = membership["data"][:]
        if membership.attrs.get("encoding-type") != "csr_matrix" or clone_shape[0] != shape[0]:
            raise ValueError("clone membership and expression must align")
        if (ptr.shape != (shape[0] + 1,) or ptr[0] != 0 or ptr[-1] != len(columns)
                or len(columns) != len(values) or np.any(np.diff(ptr) < 0)
                or np.any(columns < 0) or np.any(columns >= clone_shape[1])
                or not np.all(values == 1)):
            raise ValueError("invalid binary CSR clone membership")
        clones = np.full(shape[0], -1, dtype=int)
        single = np.flatnonzero(np.diff(ptr) == 1)
        clones[single] = columns[ptr[single]]
    if (len(ids) != shape[0] or len(genes) != shape[1] or len(times) != shape[0]
            or len(states) != shape[0] or len(set(ids)) != len(ids)
            or len(set(genes)) != len(genes) or not np.isfinite(times).all()
            or set(times) != {2.0, 4.0, 6.0} or np.any(states == "")):
        raise ValueError("LARRY IDs, genes, measured days or annotations are invalid")
    return {"ids": ids, "genes": genes, "times": times, "states": states,
            "clones": clones, "shape": shape, "n_clone_columns": clone_shape[1]}


def select_cohort(data, *, max_clones=150, source_per_clone=4, training_per_day=600):
    """Select by measured coverage and hash ranks, never by predictive success."""
    times, clones, ids = (data[k] for k in ("times", "clones", "ids"))
    usable = clones >= 0
    all_clones = sorted(set(clones[usable]))
    training_mask = usable & np.array([not clone_is_test(c) if c >= 0 else False for c in clones])
    eligible = [c for c in all_clones if clone_is_test(c)
                and np.any(times[clones == c] == 2) and np.any(times[clones == c] == 4)
                and np.sum(times[clones == c] == 6) >= 2]
    chosen = sorted(eligible, key=lambda c: _rank("larry-cohort-v1", c))[:max_clones]
    if not chosen:
        raise ValueError("no eligible held-out clones")
    training = []
    for time in (2.0, 4.0, 6.0):
        rows = np.flatnonzero(training_mask & (times == time))
        training.extend(sorted(rows, key=lambda i: _rank("larry-reference-v1", ids[i]))[:training_per_day])
    sources = {}
    for time in (2.0, 4.0):
        rows = []
        for clone in chosen:
            available = np.flatnonzero((clones == clone) & (times == time))
            rows.extend(sorted(available, key=lambda i: _rank("larry-source-v1", ids[i]))[:source_per_clone])
        sources[time] = np.array(sorted(rows), dtype=int)
    truth = np.flatnonzero(np.isin(clones, chosen) & (times == 6.0))
    training = np.array(sorted(training), dtype=int)
    reference = training[times[training] == 6.0]
    if not len(reference) or any(not np.any(times[training] == t) for t in (2.0, 4.0, 6.0)):
        raise ValueError("each training day needs measured expression")
    if set(clones[training]) & set(chosen):
        raise ValueError("clone leakage in training split")
    return {"training": training, "sources": sources, "reference": reference,
            "truth": truth, "clones": np.array(chosen, dtype=int), "eligible_test_clones": len(eligible)}


def read_expression_rows(path, rows):
    """Bound memory by scanning CSR blocks; leave unused expression on disk."""
    import h5py

    rows = np.asarray(rows, dtype=int)
    if rows.ndim != 1 or len(rows) != len(set(rows)) or np.any(np.diff(rows) <= 0):
        raise ValueError("expression rows must be sorted and unique")
    blocks = []
    with h5py.File(path, "r") as handle:
        group = handle["X"]
        n, d = map(int, group.attrs["shape"])
        if len(rows) == 0 or rows[0] < 0 or rows[-1] >= n:
            raise ValueError("expression row outside input")
        for start in range(0, n, 2048):
            chosen = rows[(rows >= start) & (rows < start + 2048)]
            if not len(chosen):
                continue
            end = min(start + 2048, n)
            ptr = group["indptr"][start:end + 1]
            first, last = int(ptr[0]), int(ptr[-1])
            values = group["data"][first:last].astype(float)
            indices = group["indices"][first:last]
            block = sparse.csr_matrix((values, indices, ptr - first), shape=(end - start, d))
            blocks.append(block[chosen - start])
    matrix = sparse.vstack(blocks, format="csr")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("expression must be finite and nonnegative")
    return matrix


def normalized_log_expression(matrix):
    if not sparse.issparse(matrix) or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("expression must be sparse, finite and nonnegative")
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(totals <= 0) or not np.isfinite(totals).all():
        raise ValueError("each measured cell needs positive total expression")
    normalized = sparse.diags(10000.0 / totals) @ matrix
    normalized.data = np.log1p(normalized.data)
    return normalized.tocsr()


def fit_expression_space(training, *, n_genes=500, n_components=20):
    """Fit all learned operations on training clones, not on their test relatives."""
    if training.shape[0] < 3 or n_genes < 1 or n_components < 1:
        raise ValueError("need at least three training cells and positive dimensions")
    mean = np.asarray(training.mean(axis=0)).ravel()
    variance = np.asarray(training.power(2).mean(axis=0)).ravel() - mean ** 2
    candidates = np.flatnonzero(variance > 1e-12)
    if not len(candidates):
        raise ValueError("training expression has no variable genes")
    columns = candidates[np.argsort(-variance[candidates], kind="stable")[:n_genes]]
    dense = training[:, columns].toarray()
    center = dense.mean(axis=0)
    dense -= center
    _, singular, loadings = linalg.svd(dense, full_matrices=False, lapack_driver="gesdd")
    count = min(n_components, training.shape[0] - 1, len(columns))
    count = min(count, int(np.sum(singular > 1e-10)))
    if count == 0:
        raise ValueError("training PCA has zero rank")
    loadings = loadings[:count].copy()
    for vector in loadings:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    scale = np.std(dense @ loadings.T, axis=0, ddof=1)
    return {"columns": columns, "center": center, "loadings": loadings, "scale": scale}


def transform_expression(matrix, fitted):
    return ((matrix[:, fitted["columns"]].toarray() - fitted["center"])
            @ fitted["loadings"].T) / fitted["scale"]


def score_clones(predicted, truth, prior, *, bootstrap=2000, seed=20261004):
    """Equal-clone scores with paired resampling, for frequency-valued outcomes."""
    predicted, truth, prior = [np.asarray(x, dtype=float) for x in (predicted, truth, prior)]
    if (predicted.ndim != 2 or truth.shape != predicted.shape or len(predicted) == 0
            or prior.shape != (predicted.shape[1],) or bootstrap < 1):
        raise ValueError("aligned nonempty clone-by-state probabilities are required")
    for value in (predicted, truth, prior):
        if not np.isfinite(value).all() or np.any(value < 0) or not np.allclose(value.sum(axis=-1), 1):
            raise ValueError("probabilities must be finite, nonnegative and sum to one")
    scores = {"transport_brier": np.sum((predicted - truth) ** 2, axis=1),
              "training_prior_brier": np.sum((prior - truth) ** 2, axis=1),
              "uniform_brier": np.sum((1.0 / predicted.shape[1] - truth) ** 2, axis=1),
              "transport_total_variation": np.sum(np.abs(predicted - truth), axis=1) / 2}
    scores["paired_brier_improvement"] = scores["training_prior_brier"] - scores["transport_brier"]
    sampled = np.random.default_rng(seed).integers(0, len(predicted), (bootstrap, len(predicted)))
    result = {"n_clones": len(predicted), "bootstrap_resamples": bootstrap, "bootstrap_seed": seed}
    for name, values in scores.items():
        result[name] = {"estimate": float(values.mean()),
                        "ci95": list(map(float, np.quantile(values[sampled].mean(axis=1), [0.025, 0.975])))}
    return result


def run_larry(path, output):
    """Execute the locked protocol and retain a locally auditable manifest."""
    import h5py

    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    digest = sha256(path)
    if digest != LARRY_SHA256:
        raise ValueError("LARRY bytes differ from the pinned source SHA-256; audit before use")
    data = read_larry_header(path)
    cohort = select_cohort(data)
    selected = np.unique(np.concatenate([cohort["training"], *cohort["sources"].values()]))
    expression = normalized_log_expression(read_expression_rows(path, selected))
    row_lookup = {int(row): i for i, row in enumerate(selected)}
    training_rows = np.array([row_lookup[int(i)] for i in cohort["training"]])
    fitted = fit_expression_space(expression[training_rows])
    features = transform_expression(expression, fitted)
    output.mkdir(parents=True, exist_ok=False)
    table = output / "cells.csv"
    inference_rows = np.concatenate([*cohort["sources"].values(), cohort["reference"]])
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cell_id", "time", "state", *[f"f_pc{k + 1}" for k in range(features.shape[1])]])
        for i in inference_rows:
            state = data["states"][i] if data["times"][i] == 6 else ""
            writer.writerow([data["ids"][i], data["times"][i], state, *features[row_lookup[int(i)]]])
    transport_report, fates = infer(table, output / "inference", epsilon=0.5)
    classes = sorted(set(data["states"]))
    class_index = {state: i for i, state in enumerate(classes)}
    prior = np.zeros(len(classes))
    for row in cohort["reference"]:
        prior[class_index[data["states"][row]]] += 1 / len(cohort["reference"])
    fate_by_id = {row["cell_id"]: row for row in fates}
    truth = []
    terminal_counts = []
    for clone in cohort["clones"]:
        rows = cohort["truth"][data["clones"][cohort["truth"]] == clone]
        frequency = np.zeros(len(classes))
        for i in rows:
            frequency[class_index[data["states"][i]]] += 1 / len(rows)
        truth.append(frequency)
        terminal_counts.append(len(rows))
    truth = np.array(truth)
    metrics, local_audit = {}, []
    for time, source in cohort["sources"].items():
        predictions = []
        for k, clone in enumerate(cohort["clones"]):
            rows = source[data["clones"][source] == clone]
            probability = np.zeros((len(rows), len(classes)))
            for r, i in enumerate(rows):
                for column, state in transport_report["class_columns"].items():
                    probability[r, class_index[state]] = fate_by_id[data["ids"][i]][column]
            averaged = probability.mean(axis=0)
            predictions.append(averaged)
            local_audit.append({"clone_column": int(clone), "source_day": time, "source_n": len(rows),
                                "terminal_n": terminal_counts[k], "prediction": averaged.tolist(),
                                "observed_terminal_frequency": truth[k].tolist()})
        metrics[str(int(time))] = score_clones(predictions, truth, prior)
    manifest = {"schema_version": 1, "source_url": LARRY_URL, "source_sha256": digest,
                "source_bytes": path.stat().st_size, "paper": PAPER_DOI, "geo": "GSE140802",
                "source_license": "No explicit data redistribution license identified; data are not redistributed.",
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "scipy": scipy.__version__, "h5py": h5py.__version__},
                "published_processed_n_cells": data["shape"][0], "n_genes": data["shape"][1],
                "n_clone_columns": data["n_clone_columns"],
                "n_observed_clones": len(set(data["clones"]) - {-1}),
                "single_clone_n_cells": int(np.sum(data["clones"] >= 0)),
                "measured_days_n_cells": {str(int(t)): int(np.sum(data["times"] == t)) for t in (2, 4, 6)},
                "eligible_test_clones": cohort["eligible_test_clones"],
                "selected_test_clones": cohort["clones"].tolist(),
                "preprocessing_training_n_cells": len(cohort["training"]),
                "preprocessing_training_n_clones": len(set(data["clones"][cohort["training"]])),
                "test_source_n_cells": {str(int(t)): len(rows) for t, rows in cohort["sources"].items()},
                "held_out_terminal_n_cells": len(cohort["truth"]), "terminal_reference_n_cells": len(cohort["reference"]),
                "terminal_reference_n_clones": len(set(data["clones"][cohort["reference"]])),
                "classes": classes, "training_terminal_prior": prior.tolist(),
                "unsupported_terminal_classes": [state for state, p in zip(classes, prior) if p == 0],
                "fitted_n_genes": len(fitted["columns"]), "fitted_n_components": len(fitted["scale"]),
                "selected_gene_names": data["genes"][fitted["columns"]].tolist(),
                "inference_input_sha256": transport_report["input_sha256"],
                "maps": transport_report["maps"], "metrics_by_source_day": metrics,
                "protocol": "docs/LARRY_VALIDATION_PLAN.md; locked before predictive metrics",
                "interpretation": "Within-experiment held-out-clone frequency prediction; not individual lineage links or human rejuvenation.",
                "uncertainty": "Clone bootstrap conditional on fixed reference, transforms and sampled terminal cells; not calibrated fate confidence."}
    (output / "validation.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output / "clone-audit.json").write_text(json.dumps(local_audit, indent=2, allow_nan=False) + "\n")
    (output / "selection.json").write_text(json.dumps({"training_ids": data["ids"][cohort["training"]].tolist(),
        "test_source_ids": {str(int(t)): data["ids"][rows].tolist() for t, rows in cohort["sources"].items()},
        "held_out_terminal_ids": data["ids"][cohort["truth"]].tolist()}, indent=2) + "\n")
    np.savez_compressed(output / "preprocessing.npz", **fitted)
    return manifest


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Pinned public LARRY held-out-clone balanced transport validation")
    parser.add_argument("--data", required=True, help="local ignored H5AD cache path")
    parser.add_argument("--out", required=True, help="new ignored output directory")
    parser.add_argument("--download", action="store_true", help="download pinned author data if absent")
    args = parser.parse_args(argv)
    path = Path(args.data)
    if args.download and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        try:
            urllib.request.urlretrieve(LARRY_URL, temporary)
            if sha256(temporary) != LARRY_SHA256:
                raise ValueError("download source SHA-256 mismatch")
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
    result = run_larry(path, args.out)
    print(json.dumps({"metrics": result["metrics_by_source_day"], "source_sha256": result["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
