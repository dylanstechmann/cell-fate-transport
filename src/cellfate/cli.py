"""Runnable synthetic falsification example and user-table inference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cellfate.pipeline import infer, read_cells


def demo(output, *, seed=0, plot=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    table = output / "cells.csv"
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cell_id", "time", "state", "batch_id", "f_progress", "f_branch"])
        for time in range(3):
            hidden_branch = rng.permutation(np.repeat([-1, 1], 40))
            for i, branch in enumerate(hidden_branch):
                # At time 0 neither feature contains the randomly assigned hidden branch.
                writer.writerow([f"t{time}-c{i:03d}", time, "branch_A" if branch < 0 else "branch_B",
                                 "synthetic", time + rng.normal(0, 0.08), time * branch + rng.normal(0, 0.3)])
    report, fate_rows = infer(table, output / "inference", epsilon=0.5)
    rows, _, times, x, _, _ = read_cells(table)
    by_id = {row["cell_id"]: row for row in fate_rows}
    column = next(k for k, v in report["class_columns"].items() if v == "branch_B")
    scores = []
    for time in [0, 1]:
        selection = [i for i, row in enumerate(rows) if float(row["time"]) == time]
        truth = np.array([rows[i]["state"] == "branch_B" for i in selection])
        predicted = np.array([by_id[rows[i]["cell_id"]][column] for i in selection])
        scores.append({"time": time, "n": len(selection), "brier_branch_B": float(np.mean((predicted - truth) ** 2)),
                       "constant_prior_brier": 0.25, "accuracy": float(np.mean((predicted >= 0.5) == truth)),
                       "note": "no fate information in features" if time == 0 else "branches separated by construction"})
    fixture = {"kind": "synthetic software fixture", "seed": seed, "scores": scores,
               "interpretation": "Hidden branch labels are independent of time-0 features. Apparent early fate certainty can therefore be false. Later success is constructed geometry, not biological validation."}
    (output / "benchmark.json").write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
    lines = ["# Synthetic branching benchmark", "", "240 synthetic cells, three independently sampled snapshots, seed " + str(seed) + ".", "",
             "The first snapshot deliberately contains **no information** about the hidden branch label.",
             "The next two snapshots separate branches geometrically. This tests both a recoverable case",
             "and a case where confident-looking transport assignments cannot establish fate.", "",
             "| Snapshot | Branch-B Brier score ↓ | Constant 0.5 baseline ↓ | Accuracy |",
             "|---|---:|---:|---:|"]
    for score in scores:
        lines.append(f"| {score['time']} | {score['brier_branch_B']:.4f} | 0.2500 | {score['accuracy']:.3f} |")
    lines += ["", "Terminal labels define the target, so their tautological fit is excluded from scoring.",
              "This is a single synthetic seed, with no biological interpretation or confidence interval.",
              "The earlier state labels are used only to score the fixture, never as transport features.", ""]
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        probability = np.array([by_id[row["cell_id"]][column] for row in rows])
        scatter = axes[0].scatter(x[:, 0], x[:, 1], c=probability, cmap="coolwarm", vmin=0, vmax=1, s=18)
        axes[0].set(xlabel="Synthetic progress feature", ylabel="Synthetic branch feature",
                    title="Inferred terminal branch-B probability")
        fig.colorbar(scatter, ax=axes[0], label="Probability under the transport model")
        axes[1].bar([0, 1], [s["brier_branch_B"] for s in scores], color=["#CC6677", "#4477AA"], width=0.55)
        axes[1].axhline(0.25, color="#555555", linestyle="--", label="Uninformative 0.5 baseline")
        axes[1].set(xticks=[0, 1], xticklabels=["Early: no fate signal", "Middle: separated"],
                    ylabel="Branch-B Brier score (lower is better)", title="Synthetic labels reveal the ambiguity")
        axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.2)
        fig.suptitle("Synthetic demonstration • inferred couplings are not observed lineages", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output / "benchmark.png", dpi=160); plt.close(fig)
        lines += ["![Synthetic transport and ambiguity benchmark](benchmark.png)", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auditable balanced transport between cell-state snapshots")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("demo", help="synthetic branching example with an intentionally unidentifiable early fate")
    d.add_argument("--out", required=True); d.add_argument("--seed", type=int, default=0)
    d.add_argument("--plot", action="store_true")
    run = commands.add_parser("infer", help="infer adjacent transport maps from a feature CSV")
    run.add_argument("csv"); run.add_argument("--out", required=True)
    run.add_argument("--epsilon", type=float, default=0.5)
    run.add_argument("--tolerance", type=float, default=1e-8)
    run.add_argument("--max-iterations", type=int, default=20000)
    run.add_argument("--max-pairs", type=int, default=4_000_000)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo(args.out, seed=args.seed, plot=args.plot)
            print(json.dumps(result, indent=2))
        else:
            report, _ = infer(args.csv, args.out, epsilon=args.epsilon, tolerance=args.tolerance,
                              max_iterations=args.max_iterations, max_pairs=args.max_pairs)
            print(json.dumps({"n_cells": report["n_cells"], "maps": len(report["maps"]), "input_sha256": report["input_sha256"]}))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
