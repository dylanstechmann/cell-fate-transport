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


@dataclass
class UnbalancedTransportResult:
    coupling: np.ndarray
    transition: np.ndarray
    source_mass: np.ndarray
    target_mass: np.ndarray
    realized_source_mass: np.ndarray
    realized_target_mass: np.ndarray
    implied_relative_mass_change: np.ndarray
    iterations: int
    transport_cost: float
    epsilon: float
    growth_relaxation: float


def unbalanced_transport(source, target, *, source_mass=None, target_mass=None, epsilon=0.5,
                         growth_relaxation=1.0, tolerance=1e-8, max_iterations=20000,
                         max_pairs=4_000_000):
    """KL-relaxed entropic transport that lets row and column mass deviate.

    Balanced transport holds both marginals fixed, so every source cell must send
    exactly its own mass onward. When one subpopulation actually proliferated, that
    constraint has to be satisfied by moving mass between unrelated states, and the
    resulting transition can claim a transfer that did not happen. Relaxing the
    marginals with a KL penalty lets a row carry more or less mass than it started
    with.

    The objective adds ``growth_relaxation * KL`` on each marginal. Large values
    approach balanced transport; small values let mass deviate freely at the cost
    of a weaker identification. The scaling iterations gain the standard exponent
    ``tau / (tau + epsilon)`` on each update.

    ``implied_relative_mass_change`` is each source row's realized mass divided by
    its requested mass. It is **model-implied**, not measured: it is whatever the
    relaxed objective found convenient given the features, epsilon and this
    penalty. Changing cell counts between snapshots do not identify proliferation,
    death or sampling depth, and this function does not separate them. Use it to
    test whether a balanced assumption is distorting a transition, not to report a
    growth rate.
    """
    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (x.ndim != 2 or y.ndim != 2 or not x.size or not y.size or x.shape[1] != y.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("source and target must be nonempty finite matrices with equal feature count")
    if (not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(tolerance)
            or not 0 < tolerance < 1):
        raise ValueError("epsilon must be positive and tolerance must lie in (0, 1)")
    if not np.isfinite(growth_relaxation) or growth_relaxation <= 0:
        raise ValueError("growth_relaxation must be positive and finite")
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
    exponent = growth_relaxation / (growth_relaxation + epsilon)
    log_a, log_b = np.log(a), np.log(b)
    log_u = np.zeros(len(x))
    log_v = np.zeros(len(y))
    for iteration in range(1, max_iterations + 1):
        log_u = exponent * (log_a - logsumexp(log_kernel + log_v[None, :], axis=1))
        log_v = exponent * (log_b - logsumexp(log_kernel + log_u[:, None], axis=0))
        if iteration % 10 == 0 or iteration == max_iterations:
            plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
            if not np.isfinite(plan).all():
                raise RuntimeError("unbalanced transport plan left the finite range")
            previous = locals().get("_previous_plan")
            if previous is not None and float(np.abs(plan - previous).sum()) <= tolerance:
                break
            _previous_plan = plan
    else:
        raise RuntimeError(
            f"unbalanced scaling did not settle in {max_iterations} iterations; "
            "raise max_iterations or growth_relaxation")
    plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
    row_sum, col_sum = plan.sum(axis=1), plan.sum(axis=0)
    if np.any(row_sum == 0):
        raise RuntimeError("an unbalanced transport row underflowed to zero")
    return UnbalancedTransportResult(
        plan, plan / row_sum[:, None], a, b, row_sum, col_sum, row_sum / a,
        iteration, float(np.sum(plan * costs)), float(epsilon), float(growth_relaxation))


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
                or not _is_row_stochastic(p)):
            raise ValueError("transition must be finite, nonnegative, row-stochastic and dimensionally aligned")
        result.insert(0, p @ result[0])
    return classes, result


def compose_transitions(transitions):
    """Compose a chain of row-stochastic transitions: T_chain = T_0 @ T_1 @ ... @ T_{n-1}."""
    if not transitions:
        raise ValueError("transitions list cannot be empty")
    chain = np.asarray(transitions[0], dtype=float)
    if not _is_row_stochastic(chain):
        raise ValueError("transition must be a nonempty finite, nonnegative row-stochastic matrix")
    for t in transitions[1:]:
        step = np.asarray(t, dtype=float)
        if not _is_row_stochastic(step) or chain.shape[1] != step.shape[0]:
            raise ValueError("adjacent transitions must be dimensionally aligned and row-stochastic")
        chain = chain @ step
    if not _is_row_stochastic(chain, atol=1e-7):
        raise RuntimeError("composed transition lost stochastic row mass")
    return chain


def _is_row_stochastic(matrix, *, atol=1e-8):
    """Return whether a matrix is a valid conditional-probability map."""
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        return False
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        return False
    row_sums = matrix.sum(axis=1)
    return bool(np.all(row_sums > 0) and np.allclose(row_sums, 1, rtol=0, atol=atol))


def build_sankey_data(couplings, transitions, snapshot_states, snapshot_times, source_masses=None,
                      *, mass_tolerance=1e-7):
    """Generate model-implied allocations without discarding or rounding mass.

    Supplied couplings must agree with the conditional maps. When only maps
    are supplied, propagate the first source distribution through the chain;
    resetting every intermediate snapshot to uniform would invent mass.
    """
    if len(snapshot_states) != len(snapshot_times) or len(snapshot_states) < 2:
        raise ValueError("need at least 2 snapshots with matched states and times")
    if len(transitions) != len(snapshot_times) - 1:
        raise ValueError("transitions count must equal snapshots count - 1")
    if not np.isfinite(mass_tolerance) or not 0 < mass_tolerance < 1:
        raise ValueError("mass tolerance must lie in (0, 1)")
    times = np.asarray(snapshot_times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("snapshot times must be finite and strictly increasing")
    snapshot_times = times.tolist()
    states = [np.asarray(item, dtype=object) for item in snapshot_states]
    if any(item.ndim != 1 or not item.size for item in states):
        raise ValueError("snapshot states must be nonempty one-dimensional sequences")
    n_maps = len(transitions)
    if couplings is not None and len(couplings) != n_maps:
        raise ValueError("provide exactly one coupling per interval or None")
    if source_masses is not None and len(source_masses) != n_maps:
        raise ValueError("provide exactly one source mass per interval or None")
    validated_couplings = []
    previous_target = None
    for k, transition in enumerate(transitions):
        step = np.asarray(transition, dtype=float)
        shape = (len(states[k]), len(states[k + 1]))
        if not _is_row_stochastic(step) or step.shape != shape:
            raise ValueError("transitions must be row-stochastic and aligned with snapshot states")
        supplied_mass = None if source_masses is None else source_masses[k]
        mass = None if supplied_mass is None else probability_mass(supplied_mass, shape[0])
        supplied_coupling = None if couplings is None else couplings[k]
        if supplied_coupling is None:
            if mass is None:
                mass = previous_target if previous_target is not None else probability_mass(None, shape[0])
            coupling = mass[:, None] * step
        else:
            coupling = np.asarray(supplied_coupling, dtype=float)
            if (coupling.shape != shape or not np.isfinite(coupling).all()
                    or np.any(coupling < 0) or not np.isclose(coupling.sum(), 1, rtol=0, atol=mass_tolerance)):
                raise ValueError("couplings must be finite, nonnegative, normalized and aligned")
            row_mass = coupling.sum(axis=1)
            if np.any(row_mass <= 0):
                raise ValueError("every coupling source row must carry positive mass")
            if mass is not None and not np.allclose(row_mass, mass, rtol=0, atol=mass_tolerance):
                raise ValueError("coupling disagrees with supplied source mass")
            if not np.allclose(coupling, row_mass[:, None] * step, rtol=0, atol=1e-7):
                raise ValueError("coupling disagrees with conditional transition")
        if previous_target is not None and not np.allclose(
                coupling.sum(axis=1), previous_target, rtol=0, atol=mass_tolerance):
            raise ValueError("adjacent couplings disagree on intermediate snapshot mass")
        previous_target = coupling.sum(axis=0)
        validated_couplings.append(coupling)

    labeled_states = [
        [str(state).strip() if state is not None and str(state).strip()
         else f"Unlabeled at t={time:g}" for state in states]
        for states, time in zip(snapshot_states, snapshot_times)
    ]

    nodes = []
    node_id_to_idx = {}
    node_idx = 0

    for k, (t, states) in enumerate(zip(snapshot_times, labeled_states)):
        unique_states = sorted(set(states))
        for state in unique_states:
            nid = f"t{k}_{state}"
            node_id_to_idx[nid] = node_idx
            count = sum(1 for s in states if s == state)
            nodes.append({
                "id": nid,
                "node_index": node_idx,
                "label": f"{state} (t={t:g})",
                "time": float(t),
                "time_index": k,
                "state": state,
                "cell_count": count,
            })
            node_idx += 1

    links = []
    for k in range(len(transitions)):
        t_src = float(snapshot_times[k])
        t_dst = float(snapshot_times[k + 1])
        src_states = np.asarray(labeled_states[k])
        dst_states = np.asarray(labeled_states[k + 1])
        coupling = validated_couplings[k]

        u_src = sorted(set(src_states))
        u_dst = sorted(set(dst_states))

        for s_state in u_src:
            src_mask = src_states == s_state
            src_id = f"t{k}_{s_state}"
            for d_state in u_dst:
                dst_mask = dst_states == d_state
                dst_id = f"t{k + 1}_{d_state}"
                flow_val = float(np.sum(coupling[np.ix_(src_mask, dst_mask)]))
                if flow_val > 0:
                    links.append({
                        "source": src_id,
                        "target": dst_id,
                        "source_index": node_id_to_idx[src_id],
                        "target_index": node_id_to_idx[dst_id],
                        "value": flow_val,
                        "source_time": t_src,
                        "target_time": t_dst,
                        "source_state": s_state,
                        "target_state": d_state,
                    })

    return {
        "nodes": nodes,
        "links": links,
        "n_snapshots": len(snapshot_times),
        "snapshot_times": [float(t) for t in snapshot_times],
        "total_mass": 1.0,
        "metadata": {
            "format": "sankey_v1",
            "description": "Model-implied transport allocations across temporal snapshots; links are not observed lineage or cell ancestry.",
            "unlabeled_state_policy": "Blank annotations are represented as a time-specific unlabeled state.",
            "mass_policy": "One normalized distribution per snapshot; all positive allocations are retained at full floating-point precision.",
            "mass_tolerance": mass_tolerance,
        },
    }
