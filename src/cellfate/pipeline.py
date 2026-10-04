"""Strict CSV input and reviewable outputs; only f_* columns enter distances."""

import csv
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import entr

from cellfate import __version__
from cellfate.transport import build_sankey_data, compose_transitions, probability_mass, pull_back_fates, transport


def read_cells(path):
    path = Path(path)
    raw = path.read_bytes()
    # Parse exactly the bytes identified by the input hash, even if the file
    # changes on disk while the report is being prepared.
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not {"cell_id", "time", "state"} <= set(header):
            raise ValueError("provide unique columns including cell_id, time and state")
        features = [name for name in header if name.startswith("f_")]
        if not features:
            raise ValueError("at least one f_* feature is required")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("ragged CSV row")
            rows.append({key: value.strip() for key, value in row.items()})
    if not rows or any(not row["cell_id"] for row in rows):
        raise ValueError("provide nonempty cell IDs")
    if len({row["cell_id"] for row in rows}) != len(rows):
        raise ValueError("cell IDs must be globally unique across snapshots")
    try:
        times = np.array([float(row["time"]) for row in rows])
        x = np.array([[float(row[name]) for name in features] for row in rows])
        mass = np.array([float(row.get("mass", "1")) for row in rows])
    except ValueError as exc:
        raise ValueError("times, masses and features must be numeric") from exc
    if not np.isfinite(times).all() or not np.isfinite(x).all() or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("times/features must be finite and masses finite and strictly positive")
    if len(set(times)) < 2:
        raise ValueError("at least two distinct time points are required")
    if any(not rows[i]["state"] for i in np.flatnonzero(times == times.max())):
        raise ValueError("terminal states cannot be blank")
    return rows, features, times, x, mass, hashlib.sha256(raw).hexdigest()


def infer(path, output, *, epsilon=0.5, tolerance=1e-8, max_iterations=20000, max_pairs=4_000_000):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    rows, features, times, x, mass, digest = read_cells(path)
    levels = sorted(set(times))
    # Stable source/target ordering is reused for every adjacent map and fate table.
    indices = [np.flatnonzero(times == t) for t in levels]
    maps = []
    for left, right in zip(indices[:-1], indices[1:]):
        maps.append(transport(x[left], x[right], source_mass=mass[left], target_mass=mass[right],
                              epsilon=epsilon, tolerance=tolerance, max_iterations=max_iterations, max_pairs=max_pairs))
    classes, fates = pull_back_fates([item.transition for item in maps], [rows[i]["state"] for i in indices[-1]])
    class_columns = {f"p_state_{i}": state for i, state in enumerate(classes)}
    terminal_prior = probability_mass(mass[indices[-1]], len(indices[-1])) @ fates[-1]
    report = {"schema_version": 1, "input_sha256": digest, "n_cells": len(rows),
              "features": features, "times": list(map(float, levels)), "class_columns": class_columns,
              "terminal_class_mass": dict(zip(classes, map(float, terminal_prior))),
              "configuration": {"epsilon": epsilon, "tolerance": tolerance, "max_iterations": max_iterations,
                  "max_pairs": max_pairs, "cost": "mean squared feature distance; no automatic normalization",
                  "solver": "log-domain Sinkhorn; dual warm starts at 8, 4, 2 times epsilon when iteration budget permits",
                  "mass": "positive input masses normalized independently within each time point",
                  "composition": "row-conditional transitions; Markov assumption"},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "cellfate": __version__},
              "maps": [], "warnings": [
                  "Snapshot couplings are geometric hypotheses, not observed lineages or causal effects.",
                  "Balanced transport fixes the supplied marginals; it cannot estimate cell proliferation or death.",
                  "Fate probabilities depend on features, sampling, terminal labels and epsilon; they are not calibrated confidence.",
                  "Each pair is matched regardless of time interval; no physical velocity or continuous-time rate is inferred.",
                  "No batch correction or raw RNA preprocessing is performed; choose a shared feature space upstream."]}
    if "batch_id" in rows[0]:
        batch_sets = [{rows[i]["batch_id"] for i in idx} - {""} for idx in indices]
        report["batches_by_time"] = {str(float(t)): sorted(b) for t, b in zip(levels, batch_sets)}
        if any(not a & b for a, b in zip(batch_sets[:-1], batch_sets[1:])):
            report["warnings"].append("Adjacent times have no shared batch ID; technical and temporal shifts may be confounded.")
    output.mkdir(parents=True, exist_ok=False)
    for k, (item, left, right) in enumerate(zip(maps, indices[:-1], indices[1:])):
        name = f"transport_{k:02d}.npz"
        np.savez_compressed(output / name, coupling=item.coupling, transition=item.transition,
                            source_mass=item.source_mass, target_mass=item.target_mass,
                            source_ids=np.array([rows[i]["cell_id"] for i in left]),
                            target_ids=np.array([rows[i]["cell_id"] for i in right]))
        report["maps"].append({"file": name, "source_time": float(levels[k]), "target_time": float(levels[k + 1]),
            "source_n": len(left), "target_n": len(right), "iterations": item.iterations,
            "marginal_l1_error": item.marginal_l1_error, "transport_cost": item.transport_cost})
    fate_rows = []
    for t, idx, probabilities in zip(levels, indices, fates):
        for i, probability in zip(idx, probabilities):
            fate_rows.append({"cell_id": rows[i]["cell_id"], "time": float(t), "state": rows[i]["state"],
                              **{column: float(p) for column, p in zip(class_columns, probability)},
                              "entropy_nats": float(entr(probability).sum())})
    with (output / "fates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fate_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(fate_rows)

    if len(levels) > 2:
        chain_trans = compose_transitions([item.transition for item in maps])
        chain_coupling = maps[0].source_mass[:, None] * chain_trans
        chain_file = "composed_chain.npz"
        np.savez_compressed(
            output / chain_file,
            composed_transition=chain_trans,
            composed_coupling=chain_coupling,
            source_mass=maps[0].source_mass,
            target_mass=maps[-1].target_mass,
            source_ids=np.array([rows[i]["cell_id"] for i in indices[0]]),
            target_ids=np.array([rows[i]["cell_id"] for i in indices[-1]]),
            source_time=float(levels[0]),
            target_time=float(levels[-1]),
            chain_times=np.array(levels, dtype=float),
        )
        report["chain_composition"] = {
            "file": chain_file,
            "source_time": float(levels[0]),
            "target_time": float(levels[-1]),
            "n_snapshots": len(levels),
            "source_n": len(indices[0]),
            "target_n": len(indices[-1]),
            "note": "Composed transition map across multi-timepoint chain under Markov assumption: T_0->K = T_0->1 @ ... @ T_{K-1}->K",
        }

        snapshot_states = [[rows[i]["state"] for i in idx] for idx in indices]
        sankey_data = build_sankey_data(
            [item.coupling for item in maps],
            [item.transition for item in maps],
            snapshot_states,
            [float(t) for t in levels],
            [item.source_mass for item in maps],
            mass_tolerance=max(tolerance, 1e-7),
        )
        sankey_file = "sankey.json"
        (output / sankey_file).write_text(json.dumps(sankey_data, indent=2, allow_nan=False) + "\n")
        report["sankey"] = {
            "file": sankey_file,
            "n_nodes": len(sankey_data["nodes"]),
            "n_links": len(sankey_data["links"]),
            "interpretation": "Model-implied transport allocations from snapshot state labels; not observed lineage or cell ancestry.",
        }

    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Cell-state transport", "", f"{len(rows)} cells, {len(levels)} snapshots; input SHA-256 `{digest}`.", "",
             "| Interval | Source / target cells | Iterations | Marginal L1 error | Cost |",
             "|---|---:|---:|---:|---:|"]
    for item in report["maps"]:
        lines.append(f"| {item['source_time']:g} → {item['target_time']:g} | {item['source_n']} / {item['target_n']} | "
                     f"{item['iterations']} | {item['marginal_l1_error']:.3g} | {item['transport_cost']:.4f} |")
    lines += ["", "Class columns: " + ", ".join(f"`{k}` = `{v}`" for k, v in class_columns.items()) + "."]
    if len(levels) > 2:
        lines += [
            "", "## Multi-Timepoint Chain & Sankey Allocation", "",
            f"Automatically composed {len(levels)} snapshots from t={levels[0]:g} to t={levels[-1]:g}.",
            f"End-to-end transport coupling saved to `{report['chain_composition']['file']}`.",
            f"Sankey allocation graph saved to `{report['sankey']['file']}` with {report['sankey']['n_nodes']} state nodes and {report['sankey']['n_links']} model-implied links. These are not observed lineage or cell ancestry.",
        ]
    lines += ["", "## Interpretation", "", *[f"- {warning}" for warning in report["warnings"]], ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return report, fate_rows
