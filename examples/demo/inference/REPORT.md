# Cell-state transport

240 cells, 3 snapshots; input SHA-256 `a33430b352c2b8b47968fd49092155cbb0291dec710d47df20f00e08ba727066`.

| Interval | Source / target cells | Iterations | Marginal L1 error | Cost |
|---|---:|---:|---:|---:|
| 0 → 1 | 80 / 80 | 40 | 1.14e-09 | 0.9489 |
| 1 → 2 | 80 / 80 | 830 | 8.93e-09 | 1.0314 |

Class columns: `p_state_0` = `branch_A`, `p_state_1` = `branch_B`.

## Interpretation

- Snapshot couplings are geometric hypotheses, not observed lineages or causal effects.
- Balanced transport fixes the supplied marginals; it cannot estimate cell proliferation or death.
- Fate probabilities depend on features, sampling, terminal labels and epsilon; they are not calibrated confidence.
- Each pair is matched regardless of time interval; no physical velocity or continuous-time rate is inferred.
- No batch correction or raw RNA preprocessing is performed; choose a shared feature space upstream.
