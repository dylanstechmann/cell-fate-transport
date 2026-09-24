# Data and model card

## Included data

`cellfate demo --seed 0` generates 80 independently sampled artificial records
at each of three time points. Each time contains 40 branch-A and 40 branch-B
labels. Progress is time plus Gaussian noise (standard deviation 0.08). The
branch feature is time times the ±1 branch label plus Gaussian noise (standard
deviation 0.3). At time zero, the branch label therefore contributes nothing
to either feature. No repeated-cell lineage is generated or claimed.

All labels are visible in the fixture for auditing. Only terminal labels enter
fate propagation; earlier labels are used afterwards to score the demonstration.
The Brier score is mean squared error for the binary branch-B probability.
Terminal-label fit is tautological and is excluded from the benchmark.

The final fixture uses epsilon 0.5 and seed 0. This is a numerical demonstration,
not a registered experiment or tuned biological model. During implementation,
epsilon 0.2 converged too slowly on its separated branches to meet the requested
1e−8 tolerance within 20,000 updates; no map from that failed run is included.
The software reports such failures explicitly.

## Intended use

Small shared-feature snapshots, teaching, numerical baselines and preparation
for collaborating-lab evaluation. Typical scientific questions concern state
redistribution during differentiation or reprogramming. This package does not
predict an intervention, measure biological age or establish rejuvenation.

## Limitations

- Assumed geometry and balanced marginals drive the assignment; plausible
  couplings are not observed parent–daughter relationships.
- Sample proportions can reflect selection and capture bias. Input masses are
  relative weights, not an estimator of proliferation or mortality.
- No correction for batch, donor, cell cycle, dissociation or modality effects.
- No raw-count normalization, gene selection or embedding estimation.
- No external cohort, lineage-tracing study, timepoint interpolation benchmark,
  calibration test or confidence interval is included.
- Dense O(nm) storage per adjacent pair; all pairwise results are retained until
  export, so total memory also grows with the number of snapshots.

## Before using a real dataset

Record accession, license, processing provenance, time units, experimental
replicates, batch hierarchy and the meaning of terminal labels. Select a
defensible metric and evaluate stability under sampling and epsilon changes.
Keep real donor identifiers and private matrices outside the repository.
For unbalanced growth-aware transport use an established method and validate
its assumptions with relevant measurements.

Original code and generated fixtures are MIT licensed. The primary literature
is cited in the README; none of its datasets or software are copied here.
