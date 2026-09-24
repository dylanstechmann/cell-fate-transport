# Mathematical specification

For adjacent snapshots with feature vectors x_i and y_j, define

$$C_{ij}=\frac{1}{d}\lVert x_i-y_j\rVert_2^2.$$

Sample weights a and b are strictly positive and each sum to one. The solver
finds a nonnegative coupling P minimizing

$$\langle P,C\rangle+\epsilon\sum_{ij}P_{ij}(\log P_{ij}-1)$$

subject to P1=a and Pᵀ1=b. This is an entropically regularized **balanced**
coupling. The reported transport cost is the unregularized term ⟨P,C⟩; it is
not a debiased Sinkhorn divergence and need not be zero for identical clouds.

With log K = −C/epsilon, alternating updates are

$$\log u_i=\log a_i-\operatorname{LSE}_j(\log K_{ij}+\log v_j),$$
$$\log v_j=\log b_j-\operatorname{LSE}_i(\log K_{ij}+\log u_i).$$

When at least 1,000 updates are permitted, warm stages use 8, 4 and 2 times the
requested epsilon, up to 300 updates each. Dual potentials are rescaled between
stages. Remaining updates solve the requested epsilon; the maximum update
budget includes all stages. Warm stages do not change the final objective.

Both row and column marginal L1 errors must be at most the requested tolerance
(default 1e−8). The reported error is their maximum. No unconverged map is
exported. Log-domain arithmetic prevents many underflow failures but does not
remove poor conditioning at very small epsilon or extreme feature scales.

## Conditional transitions and fate propagation

The row-conditional matrix is T_ij=P_ij/sum_j P_ij. It has row sums of one.
Terminal labels form a one-hot matrix F_T. Recursively compute

$$F_k=T_k F_{k+1}.$$

Each row is a distribution over the observed terminal labels. This composition
assumes the feature state is Markov sufficient; omitted biological history can
invalidate that assumption. It also assumes the observed terminal states cover
the relevant outcomes. Unknown or unsampled fates cannot be discovered by this
label-propagation step.

The pipeline groups rows by numeric time while retaining their original order
within a snapshot. Adjacent maps use the same intermediate cell ordering. Each
saved map contains both ID arrays so an independent consumer can check alignment.

## Verification versus validation

Numerical tests check conservation, analytic special cases, permutation
equivariance and agreement with a separately solved linear program in a
low-entropy example. They do not verify a biological lineage model. The demo's
terminal labels are synthetic; intermediate branch separation is constructed,
and time-0 labels are independent of its features by design.

Scientific context: [Cuturi (2013)](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)
and [Schiebinger et al. (2019)](https://doi.org/10.1016/j.cell.2019.01.006).
The implementation follows the balanced objective above, not the full WOT model.
