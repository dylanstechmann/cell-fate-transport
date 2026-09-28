# Cell Fate Transport

**Compare cell-state populations across measured time points, and make the
assumptions behind inferred fates inspectable.**

This CPU toolkit computes balanced, entropy-regularized transport between
adjacent snapshots, exports every coupling, and propagates terminal state
labels backwards through row-conditional transition matrices. It is a compact
research baseline for shared features from single-cell or imaging data.

The [included synthetic benchmark](examples/demo/REPORT.md) contains a deliberate
failure case: the first snapshot has no information about the eventual branch.
Its branch-B Brier score is **0.2626**, worse than the **0.2500** uninformative
baseline. The later, geometrically separated snapshot scores **0.000077**.
The example demonstrates both a recoverable mapping and an unidentifiable one.
These are software-fixture results, not stem-cell measurements.

![Synthetic transport benchmark](examples/demo/benchmark.png)

## Run

Python 3.10+, NumPy and SciPy; no GPU or downloaded biological data required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plots]'
python -m unittest discover -s tests -v
cellfate demo --out artifacts/demo --plot
```

Use `pip install -e .` and omit `--plot` for a smaller installation. Output
directories must be new; previous analyses are never silently overwritten.

## Bring a cell-state table

```csv
cell_id,time,state,batch_id,f_pc1,f_pc2
early-001,0,,batch-a,0.1,-0.2
early-002,0,,batch-a,0.2,0.3
late-001,1,state_A,batch-a,1.0,-0.5
late-002,1,state_B,batch-a,1.1,0.6
```

This tiny table illustrates the schema; it is not an adequate biological study.

```bash
cellfate infer examples/demo/cells.csv --epsilon 0.5 \
  --out artifacts/inference
```

| Column | Meaning |
|---|---|
| `cell_id` | Globally unique measurement ID; never a feature |
| `time` | Finite numeric collection time; at least two distinct times |
| `state` | Required terminal annotation; earlier values may be blank and do not enter inference |
| `f_*` | Finite coordinates in a shared, scientifically meaningful feature space |
| `mass` (optional) | Strictly positive sample weight, normalized separately at each time |
| `batch_id` (optional) | Reported for potential time/batch confounding; no automatic correction |

Only `f_*` columns enter the cost. Normalize RNA counts, choose features and
establish a shared embedding upstream. Per-time independently fitted PCA axes
are generally not aligned. Two-dimensional UMAP proximity is not automatically
an appropriate transport metric. Record the preprocessing and its limitations.

## Outputs

- `REPORT.md` and `report.json`: settings, source hash, versions, population
  sizes, convergence diagnostics and limitations.

- `transport_*.npz`: complete coupling, row-conditional transition, normalized
  masses, and explicit source/target IDs. Load with `allow_pickle=False`.
- `fates.csv`: terminal-state probabilities and entropy for every snapshot.
  `report.json` maps each `p_state_*` column to its state name.

The loader parses the same CSV bytes it hashes, so an input file changed during
loading cannot make the report hash refer to a different snapshot of the table.

Maps compose under a **Markov assumption**: the current measured state is
assumed sufficient for the next transition. Snapshot data do not establish that
assumption. Higher entropy means a more diffuse model assignment, not calibrated
biological uncertainty. A concentrated assignment can still be wrong.

## What this implementation supports

The objective uses mean squared feature distance plus entropic regularization.
Log-domain Sinkhorn updates avoid directly exponentiating a large negative
cost matrix. Higher-entropy warm starts assist convergence; the final coupling
must satisfy the requested marginal tolerance. Smaller epsilon or strongly
separated populations can still converge slowly; failure raises an error.
The default dense pair limit is four million entries per adjacent map.

This is **balanced transport**: normalized source and target masses are fixed.
It cannot infer proliferation, death, reprogramming yield or absolute cell
counts. Changing sample counts does not identify population growth. Physical
time gaps are recorded but do not determine a velocity or transition rate.
There is no gene-regulatory, intervention-response or rejuvenation model here.

For growth-aware reprogramming analysis, see the established
[Waddington-OT project](https://broadinstitute.github.io/wot/).
This package is an independently implemented, narrower baseline; it does not
reproduce WOT's unbalanced transport, growth refinement or RNA processing.

## Validation and research handoff

Tests check marginal conservation, unequal sample sizes, permutation
equivariance, a linear-programming reference, log-domain stability, known
Markov compositions, sample alignment and strict input validation. CI runs the
tests and complete demo on Python 3.10 and 3.12.

Read the [method derivation](docs/METHODS.md) and
[data/model card](docs/DATA_MODEL_CARD.md) before applying it to real data.
A useful next study would compare couplings with independently measured lineage
information, report batch and sampling sensitivity, and evaluate held-out
time points without using them for feature fitting or parameter selection.

Portfolio links: [regen-benchmark-kit](https://github.com/dylanstechmann/regen-benchmark-kit)
for supervised evaluation; [senescence-module-score](https://github.com/dylanstechmann/senescence-module-score)
for exploratory expression summaries. Their outputs require appropriate
preprocessing before becoming comparable trajectory features.

## References and license

- Cuturi (2013), [Sinkhorn Distances](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html).
- Schiebinger et al. (2019), [Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming](https://doi.org/10.1016/j.cell.2019.01.006).
- [Waddington-OT tutorial](https://broadinstitute.github.io/wot/tutorial/), including growth-aware modeling and temporal composition.

MIT for original code and synthetic fixtures. No source-paper code, figures,
biological data or pretrained weights are redistributed. Cite the methods
papers separately from this software.
