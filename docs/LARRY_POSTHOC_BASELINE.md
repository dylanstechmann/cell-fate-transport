# Post hoc direct-timepoint baseline

The locked LARRY analysis in `LARRY_VALIDATION_PLAN.md` evaluates the existing
balanced transport method against the training terminal-state prior and a
uniform-state baseline. A direct-timepoint nearest-neighbor comparator was
added after that plan and is therefore reported separately as a post hoc
secondary analysis. It does not change, replace, or retune the transport
result in `LARRY_VALIDATION.md`.

For each source day (2 and 4), the implementation averages training-split cell
features within each training clone. It keeps only clones with both a source-day
prototype and measured day-6 reference outcomes, ranks prototypes by squared
Euclidean distance in the PCA space fitted on the training split, and predicts
the held-out clone's terminal-state frequencies as the unweighted mean of the
five nearest eligible training clones. Ties are broken by clone-column index;
if fewer than five eligible training clones exist, all are used and the actual
count is recorded. Test-clone terminal labels are used only for scoring.

Fresh validation output places the metrics and paired clone-bootstrap
comparison under `posthoc_secondary_baselines.direct_timepoint_knn` and writes
neighbor identities and distances to `posthoc-knn-audit.json`. A positive
`estimate_transport_brier_minus_knn_brier` favors kNN. The checked-in primary
report does not contain this later analysis; reproduce it from the pinned
source data before making any claim about its comparative performance.

This is an algorithmic comparator on one processed mouse experiment. It is not
an observed lineage relationship, independent replication, biological
validation, or evidence about rejuvenation.
