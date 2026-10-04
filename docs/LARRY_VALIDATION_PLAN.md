# Locked public lineage validation plan

Written on 2026-10-04 before computing any transport or predictive metric.
This is a scoped validation of the existing balanced baseline, using mouse
hematopoietic progenitor differentiation, not human rejuvenation or treatment.

## Data and selection

Use the primary author-maintained CoSpar download
`https://kleintools.hms.harvard.edu/tools/downloads/cospar/LARRY_adata_preprocessed.h5ad`
from Weinreb et al., Science (2020), DOI `10.1126/science.aaw3381`, GEO
`GSE140802`. Record downloaded bytes and SHA-256. This processed file includes
expression, measured collection days 2/4/6, cell-state annotations and an
independent LARRY barcode membership matrix. Read only `X`, gene names,
`time_info`, `state_info`, cell IDs and `X_clone`. Ignore supplied PCA, embedding,
predicted fate, progenitor masks and growth fields. No inferred field can enter
features or ground truth. The source data do not have an explicit redistribution
license on the inspected author download page; cache data under ignored
`artifacts/`, publish original code and aggregate results only. The repository's
MIT license does not grant rights to external data.

Keep cells assigned exactly one clone. Require test clones to have at least one
cell at day 2, at least one at day 4 and at least two cells with measured
annotations at day 6. Include all published terminal annotations, including
undifferentiated and ambiguous `Neu_Mon`; do not invent a mature outcome for them.
Assign entire clones to training/reference (80%) or test (20%) by deterministic
SHA-256 of `larry-split-v1:<clone column index>`, using the first 8 hexadecimal
digits modulo 5 (zero means test). Select at most 150 eligible test clones by a
second SHA-256 rank, independent of expression and fate. Retain at most 4 source
cells per selected clone per earlier day by a hash rank. Use up to 600 training
cells per measured day, ranked by cell ID, for fitted preprocessing; the day-6
training subset is the only terminal reference population in transport.

## Inference and evaluation

Use each-cell library-size normalization to 10,000 followed by natural `log1p`
on the downloaded nonnegative expression. This protects against differing
source normalization scale, without treating already normalized values as raw
UMI counts. Fit selection of 500 genes by log-expression variance, centering,
20-component PCA and component standardization **only on training clones**.
Apply those frozen transforms to test day-2/day-4 cells. Do not use test day-6
expression, test terminal labels or any clone IDs to fit features or set model
parameters. No UMAP/SPRING cost. Sort PCA signs deterministically.

Run the unmodified balanced transport pipeline on genuine day 2 and day 4 test
snapshots and the day-6 training reference snapshot. Fixed epsilon 0.5,
marginal L1 tolerance 1e-8, max 20,000 iterations, default 4,000,000 pair cap.
No tuning after metrics. Changing sample counts is sampling, not population
growth/death. Test clone identities enter splitting and scoring only; they are
excluded from the inference CSV. Known terminal reference labels define fate
propagation; held-out clone terminal observations are excluded from inference.

For each held-out clone, truth is the day-6 observed terminal state frequency
over **all** its usable terminal cells, with no outcome subsampling. Average
earlier per-cell predicted fate distributions within that clone separately at
days 2 and 4. Score each clone equally, rather than assigning a single-cell fate
to unsampled descendants. The primary metric is multiclass Brier score
`mean_clone(sum_state((prediction - observed_clone_frequency)**2))`.
Report the fixed training terminal frequency baseline, the uniform-state
baseline, total-variation error and paired improvement over the training prior.
Resample clones 2,000 times with seed 20261004 for percentile 95% intervals
for scores and the paired baseline-minus-model difference. These intervals
describe sampled-clone variation conditional on the fixed dataset, selection,
preprocessing and reference; they omit replicate, barcode and terminal-sampling
uncertainty. Keep a per-clone audit locally, publish aggregate metrics and
selection/hash manifests. Terminal state fit itself is never scored.

## Interpretation constraints

This is held-out-clone evaluation within one existing experiment, not validation
on an independent experiment. Clonal relatives are not observed individual
parent/child links. Outcomes are sparse sampled day-6 descendants, not complete
true lifetime fate. The public processed file lacks sequencing-library and well
metadata, preventing a controlled batch/well analysis; include this caveat.
Its upstream QC/normalization may have used the complete dataset, which limits
the claim despite refitting all local learned transforms on training clones.
Compare with the baseline even if transport performs worse. No claim of
proliferation, death, intervention efficacy, lineage reconstruction accuracy,
human reprogramming, or eternal youth follows from this experiment.
