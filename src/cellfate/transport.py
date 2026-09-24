"""Balanced entropic transport with log-domain scaling and explicit diagnostics."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


@dataclass
class TransportResult:
    coupling: np.ndarray
    transition: np.ndarray
    source_mass: np.ndarray
    target_mass: np.ndarray
    iterations: int
    marginal_l1_error: float
    transport_cost: float
    epsilon: float


def probability_mass(values, n):
    weights = np.ones(n) if values is None else np.asarray(values, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("masses must be finite, strictly positive and match the sample count")
    # Scaling before summation avoids overflow of large but finite weights.
    weights = weights / weights.max()
    weights = weights / weights.sum()
    if np.any(weights == 0):
        raise ValueError("mass dynamic range exceeds floating-point precision")
    return weights


def transport(source, target, *, source_mass=None, target_mass=None, epsilon=0.5,
              tolerance=1e-8, max_iterations=20000, max_pairs=4_000_000):
    """Minimize <P,C> + epsilon * sum(P * (log(P)-1)) at fixed marginals.

    C is mean squared feature distance. Input features must already share a
    meaningful metric and scaling. No labels, growth rates or batch corrections
    are inferred. Failure to meet the marginal tolerance raises RuntimeError.
    """
    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (x.ndim != 2 or y.ndim != 2 or not x.size or not y.size or x.shape[1] != y.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("source and target must be nonempty finite matrices with equal feature count")
    if (not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(tolerance)
            or not 0 < tolerance < 1):
        raise ValueError("epsilon must be positive and tolerance must lie in (0, 1)")
    for name, value in [("max_iterations", max_iterations), ("max_pairs", max_pairs)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if len(x) * len(y) > max_pairs:
        raise ValueError("dense pair limit exceeded; subsample transparently or use a scalable OT library")
    a = probability_mass(source_mass, len(x))
    b = probability_mass(target_mass, len(y))
    costs = cdist(x, y, metric="sqeuclidean") / x.shape[1]
    with np.errstate(over="ignore", invalid="ignore"):
        log_kernel = -costs / epsilon
    if not np.isfinite(log_kernel).all():
        raise ValueError("feature scale or epsilon causes overflow; rescale features explicitly")
    log_a, log_b = np.log(a), np.log(b)
    log_v = np.zeros(len(y))
    # Warm the dual potentials at higher entropy, then solve the requested
    # epsilon. This accelerates nearly disconnected branches without relaxing
    # the final marginal tolerance or changing the final objective.
    warm = [8.0, 4.0, 2.0] if max_iterations >= 1000 and epsilon < 1e300 else []
    warm_budget = min(300, max_iterations // 10)
    stages = [(factor, warm_budget) for factor in warm]
    stages.append((1.0, max_iterations - len(warm) * warm_budget))
    iteration = 0
    previous_factor = stages[0][0]
    error = float("inf")
    for factor, budget in stages:
        if factor == 1.0:
            budget = max_iterations - iteration
        log_v *= previous_factor / factor
        previous_factor = factor
        kernel = log_kernel / factor
        for stage_iteration in range(1, budget + 1):
            iteration += 1
            log_u = log_a - logsumexp(kernel + log_v[None, :], axis=1)
            log_v = log_b - logsumexp(kernel + log_u[:, None], axis=0)
            if stage_iteration % 10 == 0 or stage_iteration == budget:
                plan = np.exp(log_u[:, None] + kernel + log_v[None, :])
                row_sum, col_sum = plan.sum(axis=1), plan.sum(axis=0)
                error = max(float(np.abs(row_sum - a).sum()), float(np.abs(col_sum - b).sum()))
                if error <= (tolerance if factor == 1 else 1e-6):
                    if factor != 1:
                        break
                    if np.any(row_sum == 0):
                        raise RuntimeError("a transport row underflowed to zero")
                    conditional = plan / row_sum[:, None]
                    return TransportResult(plan, conditional, a, b, iteration, error,
                                           float(np.sum(plan * costs)), float(epsilon))
    raise RuntimeError(f"Sinkhorn did not converge in {max_iterations} iterations (marginal L1 error {error:.3g})")


def pull_back_fates(transitions, terminal_labels):
    """Compose row-conditional maps backwards under a Markov assumption.

    The caller must align each map's target IDs with the next map's source IDs.
    The table pipeline guarantees this through a shared, explicit sample order.
    Returns one fate-probability matrix per snapshot, including the terminal one.
    """
    labels = np.asarray(terminal_labels, dtype=str)
    if labels.ndim != 1 or not labels.size or any(not x.strip() for x in labels):
        raise ValueError("every terminal sample needs a nonempty state label")
    classes = sorted(set(labels))
    result = [(labels[:, None] == np.asarray(classes)[None, :]).astype(float)]
    for transition in reversed(transitions):
        p = np.asarray(transition, dtype=float)
        if (p.ndim != 2 or not p.shape[0] or p.shape[1] != result[0].shape[0]
                or not np.isfinite(p).all() or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8)):
            raise ValueError("transition must be finite, nonnegative, row-stochastic and dimensionally aligned")
        result.insert(0, p @ result[0])
    return classes, result
