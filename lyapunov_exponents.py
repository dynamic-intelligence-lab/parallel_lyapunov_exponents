# coding: utf-8

import math
import torch
import torch.nn.functional as F
import generalized_orders_of_magnitude as goom
import torch_parallel_scan as tps


goom.config.float_dtype = torch.float64  # for precision


# Functions for normalizing vectors over generalized orders of magnitude:

def _log_norm_exp(log_x, dim):
    "Computes log(L2 norm(exp(log_x))) over dim, keeping dim."
    return goom.log_sum_exp(log_x.mul(2), dim=dim).mul(0.5).unsqueeze(dim)


def _log_normalize_exp(log_x, dim):
    "Normalizes exp(log_x) over dim in the log-domain."
    return log_x - _log_norm_exp(log_x, dim)


# Functions for computing QR-decompositions in parallel:

def _broadcastable_householder(x):
    """
    Householder transformation of column vector x, with shape [..., d, 1],
    broadcasting over preceding dimensions. Based on code from [1] and [2].

    [1] https://stackoverflow.com/questions/53489237
    [2] https://rosettacode.org/wiki/QR_decomposition#Python.
    """
    x_norm = x.norm(dim=-2)[..., None, :]
    v = x / (x[..., :1, :] + torch.copysign(x_norm, x[..., :1, :]))
    v[..., 0, :] = 1
    tau = 2 / (v.transpose(-2, -1) @ v)
    return v, tau

def _broadcastable_qr(x):
    """
    QR-decomposition of matrix x, with shape [..., d, n], broadcasting over
    preceding dims. For d <= 64, executes custom code based on [1] and [2],
    because torch.linalg.qr performs poorly with parallel batches of small
    matrices (as of May 2025).[3][4][5] For d > 64, calls torch.linalg.qr.

    [1] https://stackoverflow.com/questions/53489237 and
    [2] https://rosettacode.org/wiki/QR_decomposition#Python
    [3] https://github.com/pytorch/pytorch/issues/64058
    [4] https://github.com/pytorch/pytorch/issues/143281
    [5] https://github.com/pytorch/pytorch/issues/22573
    """
    preceding_dims, (d, n) = (x.shape[:-2], x.shape[-2:])
    if d <= 64:
        I = torch.eye(d, dtype=x.dtype, device=x.device)
        R = x.clone()
        Q = I
        for j in range(0, n):
            v, tau = _broadcastable_householder(R[..., j:, j, None])
            H = I.expand(*preceding_dims, -1, -1).contiguous()
            H[..., j:, j:] = H[..., j:, j:] - tau * (v @ v.transpose(-2, -1))
            R = H @ R
            Q = H @ Q
        return Q[..., :n, :].transpose(-2, -1), torch.triu(R[..., :n, :])
    else:
        return torch.linalg.qr(x)


# Function that executes parallel prefix scan:

@torch.no_grad()
def _prefix_scan(x, prefix_transform, dim, pad_value=0):
    """
    Apply prefix_transform in parallel over sequence x, left to right,
    updating values *in-place* to limit memory footprint. Based on [1].

    Inputs:
        x: tensor of shape [*preceding_dims, seq_len, *operand_dims].
        prefix_transform: broadcastable binary associative transformation.
        dim: dimension over which to compute the parallel scan.
        pad_value: for padding sequences to a power of two. Default: 0.
    Output:
        y: tensor of shape [*preceding_dims, seq_len, *operand_dims].

    [1] https://github.com/glassroom/torch_parallel_scan
    """
    y = x.movedim(dim, -1)  # [*preceding_dims, *operand_dims, seq_len]
    other_dims, seq_len = (y.shape[:-1], y.size(-1))
    n_powers_of_2 = int(math.ceil(math.log2(seq_len)))
    n_pads = 2 ** n_powers_of_2 - seq_len
    y = F.pad(y, (0, n_pads), value=pad_value)
    for n in (2 ** torch.arange(n_powers_of_2)).tolist():
        y = y.view(*other_dims, -1, n * 2)
        last_on_L = y[..., (n - 1):n]
        last_on_L = last_on_L.movedim((-2, -1), (dim - 1, dim))
        all_on_R = y[..., n:]
        all_on_R = all_on_R.movedim((-2, -1), (dim - 1, dim))
        all_on_R = prefix_transform(last_on_L, all_on_R)
        all_on_R = all_on_R.movedim((dim - 1, dim), (-2, -1))
        y[..., n:] = all_on_R  # update *in-place*
    y = y.view(*other_dims, -1)
    y = y[..., :seq_len]
    y = y.movedim(-1, dim)  # [*preceding_dims, seq_len, *operand_dims]
    return y


# Class for binary associative transform:

class _UpdateLogStatesOnRightWithSelectiveResetsOnLeft():
    """
    Class for constructing a binary associative transform, which, when given
    matrices log_A and log_B on the left, both with shape [..., 1, d, d], and
    log_A's and log_B's on the right, with shape [..., <num on right>, d, d],
    computes log(left A @ right A's), log(left B @ right A's + right B's),
    over generalized orders of magnitude. However, *before* the computation,
    if the cosine similarity of any pair of (exponentiated) vectors in a left
    log_A exceeds `max_cos_sim` and the corresponding left log_B is all log-
    zeros, we first modify that left log_A *in-place* to be log-zeros and that
    left log_B *in-place* to be log(orthonormal basis of the left A, obtained
    via QR- decomposition), resetting the sequence with (log-) orthonormal
    biases at that step. Inputs and outputs consist of log_A's stacked atop
    log_B's, i.e., each step's input and output stack has shape [d * 2, d].

    Args:
        d: int, size of d x d square matrices.
        qr_func: function for computing QR-decomposition.
        device: string, torch device.
        max_cos_sim: float, max cosine similarity allowed between states.
        n_above_max: (optional) int, number of pairs with cosine similarity
            above max_cos_sim that trigger a selective reset. Default: 1.
    Inputs:
        stack_on_L: log-tensor of shape [..., 1, d * 2, d].
        stacks_on_R: log-tensor of shape [..., <num>, d * 2, d].
    Output:
        updated_stacks_on_R: log-tensor of shape [..., <num>, d * 2, d].
    """
    def __init__(self, d, qr_func, device, max_cos_sim, n_above_max=1):
        self.d, self.qr_func, self.max_cos_sim, self.n_above_max = \
            (d, qr_func, max_cos_sim, n_above_max)
        self.log_zeros_atop_I = goom.log(torch.cat([
            torch.zeros(d, d, device=device),                                                # [d, d]
            torch.eye(d, device=device),                                                     # [d, d]
        ], dim=-2))                                                                          # [d * 2, d]

    def __call__(self, stack_on_L, stacks_on_R):
        d, max_cos_sim, n_above_max = (self.d, self.max_cos_sim, self.n_above_max)           # for convenience

        # Get log_A's and log_B's from left stack:
        log_A_on_L, log_B_on_L = (stack_on_L[..., :d, :], stack_on_L[..., d:, :])            # [..., 1, d, d] x 2

        # Compute cosine similarities between A's vectors:
        U_on_L = goom.exp(_log_normalize_exp(log_A_on_L, dim=-1))                            # [..., 1, d, d], A's vecs scaled to unit norm
        cos_sim_mat = U_on_L.matmul(U_on_L.transpose(-2, -1))                                # [..., 1, d, d], gram matrix with cosines
        idx_above_diag = torch.ones_like(cos_sim_mat).triu(diagonal=1).bool()                # [..., 1, d, d], True above diag, False elsewhere
        cos_sims = cos_sim_mat.masked_select(idx_above_diag).view(*U_on_L.shape[:-2], -1)    # [..., 1, <num of elements above diag>]

        # Determine which stacks on the left should be modified:
        A_on_L_is_near_colinear = (cos_sims > max_cos_sim).long().sum(dim=-1) >= n_above_max # [..., 1] boolean
        B_on_L_is_still_zeroed = (goom.exp(log_B_on_L) == 0).all(dim=(-2, -1))               # [..., 1] boolean
        should_modify_on_L = A_on_L_is_near_colinear & B_on_L_is_still_zeroed                # [..., 1] boolean

        if torch.any(should_modify_on_L):
            # Get subset of unit-length vecs for stacks to be modified on on left:
            idx = should_modify_on_L[..., None, None].expand_as(U_on_L)                      # [..., 1, d, d]
            subset_on_L = U_on_L[idx].view(*U_on_L.shape[:-4], -1, 1, d, d)                  # [<subset dims>, 1, d, d]

            # Obtain orthonormal bases of that subset:
            Q = self.qr_func(subset_on_L.transpose(-2, -1))[0].transpose(-2, -1)             # [<subset dims>, 1, d, d], L-to-R

            # Replace subset of left inputs with stacks log-zeros and log-ortho bases:
            idx = should_modify_on_L[..., None, None].expand_as(stack_on_L)                  # [..., 1, d * 2, d]
            log_zeros_atop_Q = goom.log(F.pad(Q, (0, 0,  d, 0), value=0))                    # [<subset dims>, d * 2, d]
            stack_on_L[idx] = log_zeros_atop_Q.view(stack_on_L[idx].shape)                   # modify left inputs in-place!

        # Compute log(left A @ right A), log(left B @ right A + right B):
        log_zeros_atop_I = self.log_zeros_atop_I.expand(*stack_on_L.shape[:-2], -1, -1)      # [..., 1, d * 2, d]
        log_factors_on_L = torch.cat([stack_on_L, log_zeros_atop_I], dim=-1)                 # [..., 1, d * 2, d * 2]
        return goom.log_matmul_exp(log_factors_on_L, stacks_on_R)                            # [..., <num on right>, d * 2, d]


# Functions for estimating Lyapunov exponents:

@torch.no_grad()
def estimate_spectrum_in_parallel(jac_vals, dt, max_cos_sim=0.99999, n_above_max=1, qr_func=None, prefix_scan_func=None):
    """
    Estimates spectrum of Lyapunov exponents given a sequence of Jacobian matrix
    values, applying the parallel algorithm proposed in "Generalized Orders of
    Magnitude for Scalable, Parallel, High-Dynamic-Range Computation" (Heinsen
    and Kozachkov, 2025).

    Inputs:
        jac_vals: float tensor, seq of R-to-L Jacobians, [..., n_steps, d, d].
        dt: float scalar, discrete time interval.
        max_cos_sim: (optional) float, max cosine similarity allowed between
            pairs of deviation vectors on a step. Default: 0.99999.
        n_above_max: (optional) int, number of pairs with cosine similarity
            above max_cos_sim that trigger a selective reset. Default: 1.
        qr_func: (optional) function for computing QR-decomposition in parallel.
            If provided, the function must accept torch.float inputs of shape
            [..., d, d] and return a tuple with two outputs of the same shape,
            [..., d, d], equal to the Q and R matrix factors, respectively.
        prefix_scan_func: (optional) function that applies parallel prefix scan.
            If provided, the function must accept three arguments: a *complex*
            tensor with a sequence of matrices, a binary associative function,
            and an integer indicating the dimension over which to apply the
            parallel prefix scan. The function must return a *complex* tensor.
    Output:
        est_LEs: float tensor, estimated spectra of Lyapunov exponents [..., d].
    """
    d = jac_vals.size(-1)
    if qr_func is None:
        qr_func = _broadcastable_qr  # use built-in QR-decomposition function
    if prefix_scan_func is None:
        prefix_scan_func = _prefix_scan  # use built-in prefix scan function

    # Stack log-Jacobians atop log-zeroed-biases, excluding last step:
    transp_jac_vals_except_last = jac_vals[..., :-1, :, :].transpose(-2, -1)              # [..., n_steps - 1, d, d], L-to-R
    S0 = torch.nn.init.orthogonal_(torch.empty_like(jac_vals[..., :1, :, :]))             # [..., 1, d, d], initial orthonormal state
    S0_and_transp_jac_vals = torch.cat([S0, transp_jac_vals_except_last], dim=-3)         # [..., n_steps, d, d]
    stacks = goom.log(F.pad(S0_and_transp_jac_vals, (0, 0,  0, d), value=0))              # [..., n_steps, d * 2, d]

    # Apply prefix transform over sequence of stacks with a parallel scan:
    prefix_transform = _UpdateLogStatesOnRightWithSelectiveResetsOnLeft(
        d, qr_func, jac_vals.device, max_cos_sim, n_above_max)                            # instance of callable prefix transform
    cum_stacks = prefix_scan_func(stacks, prefix_transform, dim=-3)                       # [..., n_steps, d * 2, d]
    log_S = goom.log_add_exp(cum_stacks[..., :d, :], cum_stacks[..., d:, :])              # [..., n_steps, d, d], log-states

    # Get ortho bases of exponentiated log-states:
    U = goom.exp(_log_normalize_exp(log_S, dim=-1))                                       # [..., n_steps, d, d], state vecs scaled to unit norm
    inp_Q, _ = qr_func(U.transpose(-2, -1))                                               # [..., n_steps, d, d], R-to-L orthonormal bases (Q's)

    # Apply jac_vals to prev steps' ortho bases and estimate exponents:
    out_states = jac_vals @ inp_Q.to(jac_vals.dtype)                                      # [..., n_steps, d, d]
    _, out_R = qr_func(out_states)                                                        # [..., n_steps, d, d]
    est_LEs = out_R.diagonal(dim1=-2, dim2=-1).abs().log().mean(dim=-2) / dt              # [..., d]
    return est_LEs


@torch.no_grad()
def estimate_largest_in_parallel(jac_vals, dt, reduce_scan_func=None):
    """
    Estimates largest Lyapunov exponent given a sequence of Jacobian matrix
    values, applying the parallel expression proposed in "Generalized Orders
    of Magnitude for Scalable, Parallel, High-Dynamic-Range Computation"
    (Heinsen and Kozachkov, 2025).

    Inputs:
        jac_vals: float tensor, seq of R-to-L Jacobians, [..., n_steps, d, d].
        dt: float scalar, discrete time interval.
        reduce_scan_func: (optional) function that applies parallel reduce scan.
            If provided, the function must accept three arguments: a *complex*
            tensor with a sequence of matrices, a binary associative function,
            and an integer indicating the dimension over which to apply the
            parallel scan. The function must return a *complex* tensor.
    Output:
        est_LLE: float tensor, estimated largest Lyapunov exponent, [...].
    """
    if reduce_scan_func is None:
        reduce_scan_func = tps.reduce_scan  # use default parallel reduce scan
    n_steps, d = (jac_vals.size(-3), jac_vals.size(-1))
    log_J = goom.log(jac_vals.transpose(-2, -1))                                     # transposed L-to-R log-Jacobians
    u0 = F.normalize(torch.randn_like(jac_vals[..., 0, :1, :]), dim=-1)              # [..., 1, d], initial state
    log_end_state = goom.log_matmul_exp(
        goom.log(u0),                                                                # [..., 1, d], initial log-state
        reduce_scan_func(log_J, goom.log_matmul_exp, dim=-3),                        # [..., d, d], compounded L-to-R
    )                                                                                # [..., 1, d], ending log-state
    est_LLE = goom.log_sum_exp(log_end_state * 2, dim=-1).real / (2 * n_steps * dt)  # see Appendix B in paper
    return est_LLE


@torch.no_grad()
def estimate_spectrum_sequentially(jac_vals, dt, qr_func=None):
    """
    Estimates spectrum of Lyapunov exponents given a sequence of Jacobian matrix
    values, applying the standard method with sequential QR-decompositions.

    Input:
        jac_vals: float tensor, seq of R-to-L Jacobians, [..., n_steps, d, d].
        dt: float scalar, discrete time interval.
        qr_func: (optional) function for broadcastable QR-decomposition.
            If provided, the function must accept torch.float inputs of shape
            [..., d, d] and return a tuple with two outputs of the same shape,
            [..., d, d], equal to the Q and R matrix factors, respectively.
    Output:
        est_LEs: float tensor, estimated spectra of Lyapunov exponents [..., d].
    """
    if qr_func is None:
        qr_func = torch.linalg.qr
    n_steps, d = jac_vals.shape[-3:-1]
    L = jac_vals.new_zeros(d)
    Q = F.normalize(torch.randn_like(jac_vals[..., :1, :, :]), dim=-2)
    for J in jac_vals:
        U = J @ Q
        Q, R = qr_func(U)
        L = L + R.diagonal(dim1=-2, dim2=-1).abs().log()
    est_LEs = (L / n_steps).flatten() / dt
    return est_LEs
