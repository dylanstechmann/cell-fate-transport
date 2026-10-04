# Public LARRY lineage outcome validation

Completed 2026-10-04 in the Linux Docker development container. A held-out-clone
comparison with independently measured barcode outcomes improves on the fixed
terminal-prior baseline, especially from day 4. It validates a limited state-to-
sampled-clone-outcome prediction task within one mouse differentiation experiment.
It does not validate human reprogramming, biological age reversal or longevity.

## Source and provenance

[Weinreb et al., Science (2020)](https://doi.org/10.1126/science.aaw3381),
GEO [GSE140802](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802):
in-vitro mouse bone-marrow hematopoietic progenitor differentiation with LARRY
expressed lineage barcodes. The [primary author data page](https://github.com/AllonKleinLab/paper-data/blob/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation/README.md)
describes genuine collection days 2, 4 and 6, normalized cell-by-gene expression,
clone memberships and measured cell-type annotations.

The [author-maintained CoSpar loader](https://github.com/AllonKleinLab/cospar/blob/master/cospar/datasets.py)
provides the manageable clonally labelled processed cohort used here, from
[Harvard's LARRY H5AD download](https://kleintools.hms.harvard.edu/tools/downloads/cospar/LARRY_adata_preprocessed.h5ad).
Downloaded on 2026-10-04: **186,750,814 bytes**, SHA-256
`16607046cd2550dbcdb33c24c58e5b8852c5558f7b52a437e0ae20b0137ef881`.
The importer rejects changed bytes. The full original normalized matrix would
require a 2,065,554,808-byte compressed download; it was not downloaded.

The source license has no explicit data redistribution grant on the inspected
author page. Source data remain under ignored `artifacts/`; this repository
publishes original importer/evaluation code and aggregate results. CoSpar's
software MIT license is not assumed to license the external dataset. The
original clone, gene-name and metadata files were inspected for provenance, but
not used to establish an unverified mapping to the processed row order.

| Audited original supporting file | SHA-256 of downloaded gzip bytes |
|---|---|
| `stateFate_inVitro_clone_matrix.mtx.gz` | `26258702f8ccb7c98884d4f91bdcd1d6f552d7f2dd5125358efe7ade2fdcc50d` |
| `stateFate_inVitro_gene_names.txt.gz` | `5672e137bcac196851b8dc44311b40fe61c428d6fd726dfae320b3fe5ddfffcd` |
| `stateFate_inVitro_metadata.txt.gz` | `c88e02cc512205e7033e28bf512f5d0dd43b98ba547fdd4a196b4bd9b62d7c07` |

## Selection and separation

The [protocol](LARRY_VALIDATION_PLAN.md) was written before any predictive metric.
No gene selection, PCA, epsilon or cohort rule was adjusted in response to scores.
The public processed cohort has **49,116 cells**, **25,289 genes**, **5,857
observed clones** and 5,864 clone columns (seven empty). All its cells have one
clone assignment; the original author matrix contains 130,887 cells, including
unlabelled cells, and is not the same cohort.

| Measured day | Processed source cells | Evaluation source cells | Training preprocessing cells |
|---|---:|---:|---:|
| 2 | 4,585 | 204 | 600 |
| 4 | 14,962 | 415 | 600 |
| 6 | 29,569 | excluded | 600 |

The deterministic 80/20 clone hash split yields **147 eligible test clones**
with days 2 and 4 sampled and at least two terminal cells. All 147 meet the
150-clone cap. Hash-ranked source sampling retains at most four cells per clone
per earlier day. All **1,520 usable day-6 cells** from those clones define the
held-out observed outcome distributions; none enter inference or preprocessing.
The **1,800 preprocessing cells represent 1,366 training clones**. Its **600
day-6 cells from 479 training clones** form the terminal reference population.

Only nonnegative expression `X` enters library-size normalization, `log1p`,
500-gene variance selection and standardized 20-component PCA. All learned
operations fit on training clones. Published precomputed PCA/embeddings,
predicted fates, masks and growth fields are ignored. Clone identities enter
splitting/scoring only, and early state labels are blank in the inference CSV.
The future outcomes of held-out clones are unavailable to the model.

The inference table includes 204 measured day-2 cells, 415 measured day-4 cells
and 600 training reference day-6 cells. Input SHA-256:
`99a1476c846c52ade682f136604a60894b45b7217d77542f8ab96562e6b719dd`.
Each snapshot uses independently normalized equal cell masses. These changing
sample sizes do not imply proliferation, death, survival or absolute yield.

## Results

Fixed balanced transport epsilon 0.5; marginal tolerance 1e-8. For each test
clone, earlier predicted per-cell distributions are averaged and compared with
its observed day-6 annotation frequency across all terminal cells. Each clone
has equal scoring weight. The multiclass frequency Brier score is the sum of
squared errors across 12 state categories, averaged over clones; lower is better.
It differs from the binary branch-B metric in the synthetic demo and should not
be numerically compared with that demo's 0.25 baseline.

| Source day | Transport Brier [95% CI] | Training-prior Brier [95% CI] | Paired improvement, prior minus transport [95% CI] |
|---|---:|---:|---:|
| 2 | 0.5663 [0.5282, 0.6044] | 0.5784 [0.5373, 0.6178] | 0.0120 [0.0065, 0.0181] |
| 4 | 0.4291 [0.3941, 0.4650] | 0.5784 [0.5373, 0.6178] | 0.1493 [0.1174, 0.1840] |

The uniform-12-state baseline scores **0.7008 [0.6610, 0.7385]** at both days.
Mean clone total-variation error is **0.6745 [0.6526, 0.6958]** at day 2 and
**0.5722 [0.5495, 0.5945]** at day 4, so substantial outcome error remains.
Intervals use 2,000 paired clone bootstrap resamples, seed 20261004. Positive
paired improvements support improvement over this fixed baseline conditional
on this cohort and reference; they are not calibrated cell-fate confidence.

| Transport interval | Cells source / target | Iterations | Marginal L1 error |
|---|---:|---:|---:|
| day 2 → 4 | 204 / 415 | 40 | 2.45e-12 |
| day 4 → 6 | 415 / 600 | 50 | 2.14e-15 |

The day-2 prediction composes these maps under the documented Markov assumption.
Numerical conservation is satisfied but does not establish lineage accuracy.
All terminal categories are included in scoring, even `undiff` and `Neu_Mon`.
**No pDC cells occur in the selected terminal reference**, so its predicted
probability is zero and unsupported-outcome error is retained, not removed.

## Practical limits and next validation

- These are sampled clone outcome frequencies, not individual parent/child
  links, complete lifetime fate or proven single-cell fate decisions.
- This is clone holdout within one public experiment, not a separate biological
  experiment. Upstream QC/normalization and state annotations used in the
  author's processed file may have involved the complete cohort.
- Hash sampling and minimum descendant coverage select a limited, clonally
  recoverable population. Rare outcomes and small clones may be underrepresented.
- The processed file lacks sequencing-library, starting-population and well
  metadata. A controlled well/batch analysis requires verified metadata alignment
  or import of the original large matrix. No such alignment is assumed here.
- Clone bootstrap intervals condition on the fixed transforms, reference and
  observed terminal cells. They omit barcode errors, replicate effects,
  parameter sensitivity and uncertainty from sparsely sampled descendants.
- Balanced marginals and transductive day-4 test snapshot information affect
  the composed day-2 predictions; they do not infer biological growth or rates.

A separate lineage-traced experiment with verified batches, sufficient terminal
coverage and a frozen feature/parameter specification is the next evidence step.
Reprogramming and regeneration require their own outcome and safety measurements.

## Reproduction and verification

Use the README command with `.[validation]`. Runs create new output directories.
Local outputs include the full inference maps, per-clone audit, selected IDs and
fitted preprocessing arrays. The tracked [aggregate manifest](validation/larry-2026-10-04.json)
contains source hashes, all reported metrics, selected clone indices and genes.
The external expression matrix and individual cell outcome table are not tracked.

Docker verification passed **19 tests**, including H5AD alignment checks,
train/test clone separation, exclusion of test terminal cells from inference,
frequency-valued scoring, paired bootstrap and all original transport tests.
The unchanged synthetic seed-0 fixture remains deliberately unsuccessful at
day 0: branch-B Brier **0.26258235 > 0.25**; day-1 Brier **0.0000766546**.
