"""EVON: Eigenspace VON optimizer combining a Shampoo preconditioner with IVON-style diagonal Hessian estimation in the eigenspace, resulting in a non-diagonal Hessian in parameter space."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from itertools import chain

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer


__all__ = ['EVON']

# Newton-Schulz constants from Keller Jordan's Muon post:
# https://kellerjordan.github.io/posts/muon/
_NS_A: float = 3.4445
_NS_B: float = -4.7750
_NS_C: float = 2.0315
_NS_STEPS: int = 5
_NS_EPS: float = 1e-7


class EVON(Optimizer):
    """EVON: Eigenspace VON optimizer.

    Runs IVON-style diagonal Hessian estimation (Price estimator) in a
    Shampoo eigenbasis with optional Newton-Schulz whitening on the
    preconditioned gradient.
    """

    def __init__(
        self,
        params,
        ess: float,
        hess_init: float,
        # Shampoo hyperparameters
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.9999),
        shampoo_beta: float = -1,
        eps: float = 1e-10,
        weight_decay: float = 1e-6,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        cast_dtype=torch.float32,
        # EVON hyperparameters
        *,
        mc_samples: int = 1,
        phasing: bool = False,
        price_clip_ratio: float | None = None,
        sync: bool = True,
        whiten_prec_grad: bool = True,
        debias_beta2: bool = False,
    ) -> None:
        """Initialize EVON.

        Args:
            params: Parameters to optimize or list of parameter-group dicts.
            ess: Effective sample size controlling posterior noise scale.
            hess_init: Initial diagonal Hessian estimate (must be > 0).
            lr: Base learning rate.
            betas: ``(beta1, beta2)`` EMA coefficients for momentum and Hessian.
            shampoo_beta: EMA for Shampoo statistics;
                falls back to ``betas[1]`` when -1.
            eps: Numerical stabilizer.
            weight_decay: L2 regularization / prior precision (coupled).
            precondition_frequency: Steps between Shampoo eigenspace updates.
            max_precond_dim: Max tensor dimension for Shampoo blocks.
            merge_dims: Merge tensor dimensions before preconditioning.
            precondition_1d: Precondition 1-D tensors.
            data_format: Tensor layout hint (``"channels_first"`` or ``"channels_last"``).
            correct_bias: Apply Adam-style bias correction.
            cast_dtype: Optional dtype for preconditioner math.
            mc_samples: Monte Carlo samples per optimizer step.
            hess_clip_ratio: Clip ratio for the Hessian estimator residual.
                When set, the raw Hessian estimate ``f`` is clamped to
                ``[-hess_clip_ratio * (h + eps), hess_clip_ratio * (h + eps)]``
                before the EMA update, preventing large gradient samples from
                corrupting the Hessian. Set to ``None`` to disable clipping.
            phasing: Enables clean/noisy alternating phases to decrease
                noise in left and right preconditioner updates.
            sync: All-reduce MC sample statistics across distributed workers.
            whiten_prec_grad: Apply Newton-Schulz whitening to the
                preconditioned gradient.
        """
        if not 0.0 < ess:
            raise ValueError(f'Invalid effective sampling size: {ess}')
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not (0.0 <= shampoo_beta < 1.0 or shampoo_beta == -1):
            raise ValueError(f'Invalid shampoo_beta must be in [0.0, 1.0) or -1.0, got {shampoo_beta}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if not 1 <= mc_samples:
            raise ValueError(f'Invalid mc_samples value: {mc_samples}')

        defaults = {
            'lr': lr,
            'betas': betas,
            'shampoo_beta': shampoo_beta,
            'eps': eps,
            'weight_decay': weight_decay,
            'precondition_frequency': precondition_frequency,
            'max_precond_dim': max_precond_dim,
            'merge_dims': merge_dims,
            'precondition_1d': precondition_1d,
            'correct_bias': correct_bias,
            'ess': ess,
            'hess_init': hess_init,
            'phasing': phasing,
            'price_clip_ratio': price_clip_ratio,
            'whiten_prec_grad': whiten_prec_grad,
            'debias_beta2': debias_beta2,
        }
        super().__init__(params, defaults)

        self.cast_dtype = cast_dtype
        self._data_format = data_format

        self._ess = ess
        self._hess_init = hess_init
        self._mc_samples = mc_samples
        self._phasing = phasing
        self._price_clip_ratio = price_clip_ratio
        self.sync = sync
        self._whiten_prec_grad = whiten_prec_grad
        self._debias_beta2 = debias_beta2

        if self._ess == float('inf'):
            self._sampling_enabled = False
        else:
            self._sampling_enabled = True

        # Per-step accumulators (reset after each step).
        # Stored as instance attributes rather than param state because they
        # are keyed by Python id(p) and are not needed for checkpointing.
        # NOTE: for FSDP / tensor-parallel you must call sync_samples() before
        # step() so that these are consistent across ranks.
        self._reset_samples()

    def disable_sampling(self) -> None:
        """Switch to deterministic (MAP) mode."""
        self._sampling_enabled = False

    def enable_sampling(self) -> None:
        """Re-enable variational sampling."""
        self._sampling_enabled = True

    @contextmanager
    def sampled_params(self, train: bool = True):
        """Context manager for a sampled forward pass.

        Perturbs parameters with noise drawn from the approximate posterior,
        runs the user's forward pass, then restores the posterior means and
        accumulates gradient statistics.

        Usage::

            for batch in loader:
                with optimizer.sampled_params():
                    loss = criterion(model(batch), targets)
                loss.backward()
            optimizer.step()
        """
        if not self._sampling_enabled:
            yield
            return
        self._sample_params(train=train)
        try:
            yield
        finally:
            self._restore_param_means(train=train)

    @torch.no_grad()
    def step(self, closure=None) -> Tensor | None:
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is not None:
            losses = []
            for _ in range(self._mc_samples):
                with torch.enable_grad():
                    losses.append(closure())
            loss = sum(losses) / self._mc_samples
        else:
            loss = None

        if self.sync and dist.is_available() and dist.is_initialized():
            self.sync_samples()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                p_id = id(p)

                # determine gradient source
                if p_id in self._avg_raw_grads:
                    grad = self._avg_raw_grads[p_id].to(
                        self.cast_dtype if self.cast_dtype is not None else p.dtype
                    )
                elif p.grad is not None:
                    grad = p.grad.to(
                        self.cast_dtype if self.cast_dtype is not None else p.dtype
                    )
                else:
                    continue

                state = self.state[p]

                # lazy state initialisation
                if "step" not in state:
                    state["step"] = 0
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad, dtype=self.cast_dtype or p.dtype)
                if "h_mom" not in state:
                    state["h_mom"] = torch.full_like(
                        grad, self._hess_init, dtype=self.cast_dtype or p.dtype
                    )

                # initialise Shampoo preconditioner on first encounter
                if "Q" not in state:
                    self._init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(
                            group["shampoo_beta"]
                            if group["shampoo_beta"] >= 0
                            else group["betas"][1]
                        ),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                    self._update_preconditioner(
                        grad,
                        state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"],
                    )
                    # skip parameter update on the very first step
                    if group["max_precond_dim"] > 0:
                        continue

                # project gradient into Shampoo eigenspace
                if p_id in self._avg_grads:
                    grad_proj = self._avg_grads[p_id]
                else:
                    grad_proj = self._project(
                        grad,
                        state,
                        merge_dims=group["merge_dims"],
                        max_precond_dim=group["max_precond_dim"],
                    )

                exp_avg = state["exp_avg"]
                h_mom = state["h_mom"]
                state["step"] += 1

                # Phase detection based on step count
                # Clean Phase (Odd steps after increment):
                # Noise=0, Precond Update=YES, Hessian Update=NO
                # Noisy Phase (Even steps after increment):
                # Noise=Normal, Precond Update=NO, Hessian Update=YES
                current_step = state["step"]
                is_clean_phase = (
                    current_step % 2 != 0 or not self._sampling_enabled
                ) and group.get("phasing", self._phasing)

                # momentum update
                exp_avg.mul_(beta1).add_(grad_proj, alpha=1.0 - beta1)

                # Hessian update (Price estimator or grad² fallback)
                if self._sampling_enabled:
                    if p_id in self._avg_nxgs and not is_clean_phase:
                        EVON._price_hess_update(
                            h_mom,
                            self._avg_nxgs[p_id],
                            ess=group.get("ess", self._ess),
                            wd=group["weight_decay"],
                            eps=group["eps"],
                            beta2=beta2,
                            clip_ratio=group.get("price_clip_ratio", self._price_clip_ratio),
                        )
                    # if no nxg accumulated yet (e.g. first step without
                    # sampled_params), leave h_mom unchanged
                else:
                    # deterministic fallback: exponential moving average of g²
                    h_mom.mul_(beta2).add_(grad_proj.square(), alpha=1.0 - beta2)

                # bias correction
                step = state["step"]
                debias1 = (1.0 - beta1 ** step) if group["correct_bias"] else 1.0
                debias2 = (1.0 - beta2 ** step) if group["correct_bias"] and self._debias_beta2 else 1.0
                step_size = group["lr"] / debias2

                # IVON-style: h + wd (full Newton step denominator)
                preconditioner = h_mom + group["weight_decay"] + group["eps"]

                # coupled weight decay: project p into eigenspace
                p_proj = self._project(
                    p.data.to(self.cast_dtype if self.cast_dtype is not None else p.dtype),
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                # preconditioning step
                precond_grad = (exp_avg / debias1).add_(
                    p_proj, alpha=group["weight_decay"]).div_(preconditioner)

                # project back to parameter space
                update = self._project_back(
                    precond_grad.to(grad.dtype),
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                # optional Newton-Schulz whitening
                if group.get("whiten_prec_grad", self._whiten_prec_grad):
                    update = _whiten(update, eps=group["eps"])

                # parameter update
                p.add_(update.to(p.dtype), alpha=-step_size)

                # update Shampoo preconditioner
                if is_clean_phase or not group.get("phasing", self._phasing)  or step % group.get("precondition_frequency", 10) == 0:
                    self._update_preconditioner(
                        grad,
                        state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"],
                    )

        self._reset_samples()
        return loss

    # -----------------------------------------------------------------------
    # Shampoo preconditioner
    # -----------------------------------------------------------------------

    def _merge_dims(self, grad: Tensor, max_precond_dim: int) -> Tensor:
        """Merge leading dimensions until the product fits within ``max_precond_dim``."""
        assert self._data_format in ("channels_first", "channels_last")
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []
        curr = 1
        for s in shape:
            tmp = curr * s
            if tmp > max_precond_dim:
                if curr > 1:
                    new_shape.append(curr)
                    curr = s
                else:
                    new_shape.append(s)
                    curr = 1
            else:
                curr = tmp
        if curr > 1 or len(new_shape) == 0:
            new_shape.append(curr)
        return grad.reshape(new_shape)

    def _init_preconditioner(
        self,
        grad: Tensor,
        state: dict,
        precondition_frequency: int = 10,
        shampoo_beta: float = 0.95,
        max_precond_dim: int = 10000,
        precondition_1d: bool = False,
        merge_dims: bool = False,
    ) -> None:
        """Initialise the GG accumulator matrices and Q placeholder."""
        state["GG"] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype)
                )
        else:
            if merge_dims:
                grad = self._merge_dims(grad, max_precond_dim)
            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(
                        torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype)
                    )
        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def _update_preconditioner(
        self,
        grad: Tensor,
        state: dict,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
    ) -> None:
        """Update GG accumulators and recompute eigenbases when due."""
        if (
            state["Q"] is not None
            and state["step"] > 0
            and state["step"] % state["precondition_frequency"] == 0
        ):
            state["exp_avg"] = self._project_back(
                state["exp_avg"], state,
                merge_dims=merge_dims, max_precond_dim=max_precond_dim,
            )

        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"])
        else:
            if merge_dims:
                new_grad = self._merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer = torch.tensordot(
                            new_grad, new_grad,
                            dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2,
                        )
                        state["GG"][idx].lerp_(outer, 1 - state["shampoo_beta"])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer = torch.tensordot(
                            grad, grad,
                            dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                        )
                        state["GG"][idx].lerp_(outer, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = _get_orthogonal_matrix(state["GG"])

        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = _get_orthogonal_matrix_qr(state, max_precond_dim)
            state["exp_avg"] = self._project(
                state["exp_avg"], state,
                merge_dims=merge_dims, max_precond_dim=max_precond_dim,
            )

    def _project(
        self,
        grad: Tensor,
        state: dict,
        merge_dims: bool = False,
        max_precond_dim: int = 10000,
    ) -> Tensor:
        """Project ``grad`` into the Shampoo eigenbasis."""
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape  # noqa: F841
            grad = self._merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                mat_use = mat if mat.dtype == grad.dtype else mat.to(grad.dtype)
                grad = torch.tensordot(grad, mat_use, dims=[[0], [0]])
            else:
                grad = grad.permute(list(range(1, grad.ndim)) + [0])

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _project_back(
        self,
        grad: Tensor,
        state: dict,
        merge_dims: bool = False,
        max_precond_dim: int = 10000,
    ) -> Tensor:
        """Project ``grad`` back from the Shampoo eigenbasis to parameter space."""
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape  # noqa: F841
            grad = self._merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                mat_use = mat if mat.dtype == grad.dtype else mat.to(grad.dtype)
                grad = torch.tensordot(grad, mat_use, dims=[[0], [1]])
            else:
                grad = grad.permute(list(range(1, grad.ndim)) + [0])

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    # -----------------------------------------------------------------------
    # Sampling helpers
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _sample_params(self, train: bool = True) -> None:
        """Perturb each parameter with noise sampled from the approximate posterior.

        Noise is drawn as ``epsilon ~ N(0, 1 / (ess * (h + wd)))``.

        Persistent state added by this method:

        * ``state["_mean_buf"]``: one tensor per parameter, shape/dtype matching ``p.data``. Filled with the posterior mean on every call and read back by ``_restore_param_means``.

        Transient state (freed by ``_reset_samples`` at the end of each step):

        * ``self._noises[id(p)]``: eigenspace-basis noise used by the Price estimator in ``_restore_param_means``. Only populated when ``train=True`` and ``ess != inf``.
        """
        for group in self.param_groups:
            ess = group.get("ess", self._ess)
            wd = group["weight_decay"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                state = self.state[p]
                h_mom = state.get(
                    "h_mom",
                    torch.full_like(p, self._hess_init, dtype=self.cast_dtype or p.dtype),
                )

                # allocate _mean_buf once, then always write in-place
                if "_mean_buf" not in state:
                    state["_mean_buf"] = torch.empty_like(p.data)
                state["_mean_buf"].copy_(p.data)

                # Clean Phase Detection:
                # If step is missing, assume 0 (Clean).
                # If step is even, it is Clean Phase -> Noise = 0.
                current_step = state.get("step", 0)
                is_clean_phase = (current_step % 2 == 0) and group.get(
                    "phasing", self._phasing
                )

                if is_clean_phase:
                    noise = torch.zeros_like(h_mom)
                else:
                    # compute noise on-the-fly (temporary tensors, not stored in state)
                    # h_mom.add(wd) returns a new tensor, so h_mom is never mutated
                    denom = h_mom.add(wd).mul_(ess).sqrt_()
                    noise = torch.empty_like(h_mom).normal_().div_(denom)

                if train:
                    self._noises[id(p)] = noise

                if "Q" in state:
                    full_noise = self._project_back(
                        noise.to(self.cast_dtype if self.cast_dtype is not None else p.dtype),
                        state,
                        merge_dims=group["merge_dims"],
                        max_precond_dim=group["max_precond_dim"],
                    )
                else:
                    full_noise = noise

                p.data.add_(full_noise)

    @torch.no_grad()
    def _restore_param_means(self, train: bool = True) -> None:
        """Restore the posterior means and accumulate gradient statistics.

        For each parameter that was perturbed, restores ``p.data`` from
        ``state["_mean_buf"]`` and, when ``train=True``, accumulates:

        * ``_avg_raw_grads[p_id]``: Welford mean of raw gradients (for
          Shampoo updates and momentum initialisation).
        * ``_avg_grads[p_id]``: Welford mean of eigenspace-projected gradients.
        * ``_avg_nxgs[p_id]``: Welford mean of ``noise * grad_proj`` (the
          Price estimator numerator).
        """
        if train:
            self._sample_count += 1
        count = self._sample_count if train else None

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue

                state = self.state[p]

                # restore posterior mean directly from _mean_buf
                if "_mean_buf" in state:
                    p.data.copy_(state["_mean_buf"])

                if not (train and p.grad is not None):
                    self._noises.pop(id(p), None)
                    continue

                p_id = id(p)
                grad = p.grad

                # raw gradient accumulation (for Shampoo / momentum)
                grad_cast = grad.to(
                    self.cast_dtype if self.cast_dtype is not None else p.dtype
                )
                self._avg_raw_grads[p_id] = _welford_mean(
                    self._avg_raw_grads.get(p_id),
                    grad_cast.clone(),
                    count,
                )

                # eigenspace gradient accumulation
                # when Q is absent the projection is the identity, so grad_proj = grad_cast
                if "Q" in state:
                    grad_proj = self._project(
                        grad_cast,
                        state,
                        merge_dims=group["merge_dims"],
                        max_precond_dim=group["max_precond_dim"],
                    )
                else:
                    grad_proj = grad_cast

                self._avg_grads[p_id] = _welford_mean(
                    self._avg_grads.get(p_id), grad_proj.clone(), count
                )

                # Price estimator: accumulate noise * grad_proj
                noise = self._noises.pop(p_id, None)
                if noise is not None:
                    self._avg_nxgs[p_id] = _welford_mean(
                        self._avg_nxgs.get(p_id), noise * grad_proj, count
                    )

                # set grad to None to free memory; it is only valid within the sampled_params context
                p.grad = None

    def _reset_samples(self) -> None:
        """Reset per-step MC accumulators."""
        self._sample_count: int = 0
        self._avg_grads: dict[int, Tensor] = {}
        self._avg_raw_grads: dict[int, Tensor] = {}
        self._avg_nxgs: dict[int, Tensor] = {}
        self._noises: dict[int, Tensor] = {}

    # -----------------------------------------------------------------------
    # Distributed synchronisation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sync_samples(self) -> None:
        """All-reduce MC sample statistics across distributed workers.

        Call this before ``step()`` when each rank collects independent
        MC samples and you want to combine them into a globally consistent
        gradient / Hessian estimate.
        """
        local_count = float(self._sample_count)
        if local_count <= 0 or not dist.is_initialized():
            return

        # discover a reference device from the first parameter
        ref_device = None
        for group in self.param_groups:
            for p in group["params"]:
                ref_device = p.device
                break
            if ref_device is not None:
                break
        if ref_device is None:
            return

        count_t = torch.tensor(local_count, device=ref_device)
        dist.all_reduce(count_t)
        global_count = int(count_t.item())
        if global_count <= 0:
            return

        world_size = dist.get_world_size()

        bufs = (self._avg_grads, self._avg_raw_grads, self._avg_nxgs)

        # collect (tensor, buf_index, p_id) triples that need all-reducing
        entries: list[tuple[Tensor, int, int]] = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                p_id = id(p)
                for i, buf in enumerate(bufs):
                    if p_id in buf:
                        entries.append((buf[p_id], i, p_id))

        # batch by (device, dtype) for one all_reduce per pair instead of one
        # per tensor, cutting NCCL launch overhead from O(3 * params) to O(1)
        # for the common single-dtype case
        dgroups: dict[tuple, list[tuple[Tensor, int, int]]] = defaultdict(list)
        for t, i, p_id in entries:
            dgroups[(t.device, t.dtype)].append((t, i, p_id))

        for (_, _dtype), dgroup_entries in dgroups.items():
            tensors = [t for t, _, _ in dgroup_entries]
            splits = [t.numel() for t in tensors]
            flat = torch.cat([t.flatten().mul(local_count) for t in tensors])
            dist.all_reduce(flat)
            flat.div_(global_count)
            offset = 0
            for (_, i, p_id), n in zip(dgroup_entries, splits):
                bufs[i][p_id] = flat[offset : offset + n].view_as(bufs[i][p_id])
                offset += n

        # keep local count consistent with IVON's sync convention
        self._sample_count *= world_size

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _price_hess_update(
        h: Tensor,
        avg_nxg: Tensor,
        ess: float,
        wd: float,
        eps: float,
        beta2: float,
        clip_ratio: float | None = None,
    ) -> None:
        """One EMA step of the Price Hessian estimator.

        ``h <- beta2 h + (1-beta2) f + 0.5*(1-beta2)**2 (h-f)**2 / (h + wd)``

        where ``f = avg_nxg * (h + wd) * ess`` is the raw Price estimate.

        When ``clip_ratio`` is set, ``f`` is clamped element-wise to
        ``±clip_ratio * (h + eps)`` before the EMA update, preventing
        outlier gradient samples from corrupting the Hessian estimate.
        """
        f = avg_nxg * (h + wd) * ess
        if clip_ratio is not None and clip_ratio > 0.0:
            limit = clip_ratio * (h + eps)
            f = torch.clamp(f, min=-limit, max=limit)
        correction = 0.5 * (1.0 - beta2) ** 2 * (h - f).square() / (h + wd + eps)
        h.mul_(beta2).add_(f, alpha=1.0 - beta2).add_(correction)


# ---------------------------------------------------------------------------
# Online statistics
# ---------------------------------------------------------------------------


def _welford_mean(avg: Tensor | None, newval: Tensor, count: int) -> Tensor:
    """Welford online mean update (in-place on ``avg`` when possible)."""
    if avg is None:
        return newval.clone()
    return avg.add_(newval.sub_(avg).div_(count))


# ---------------------------------------------------------------------------
# Newton-Schulz whitening
# ---------------------------------------------------------------------------


def _normalize_1d(g: Tensor, eps: float = _NS_EPS) -> Tensor:
    """L2-normalize a 1-D gradient vector."""
    if g.ndim != 1:
        raise ValueError("Expected a 1-D tensor.")
    return g / g.norm().clamp(min=eps)


@torch.compile
def _zeropower_via_newtonschulz(
    g: Tensor,
    ns_steps: int = _NS_STEPS,
    ns_coeffs: tuple[float, float, float] = (_NS_A, _NS_B, _NS_C),
    eps: float = _NS_EPS,
) -> Tensor:
    """
    Newton-Schulz orthogonalization of a 2-D matrix G.

    Computes (approximately) the matrix sign / zeroth-power G(G^T G)^{-1/2},
    which maps G → U S' V^T where S'_ii ≈ Uniform(0.5, 1.5).

    Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    """
    if g.ndim != 2:
        raise ValueError("Expected a 2-D tensor.")
    if len(ns_coeffs) != 3:
        raise ValueError("ns_coeffs must be a 3-tuple (a, b, c).")
    if ns_steps >= 100:
        raise ValueError("ns_steps must be < 100 for efficiency.")

    a, b, c = ns_coeffs
    x = g.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / x.norm().clamp(min=eps)
    for _ in range(ns_steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x


def _whiten(g: Tensor, eps: float = _NS_EPS) -> Tensor:
    """Apply Newton-Schulz whitening to a gradient of any shape."""
    if g.ndim == 1:
        return _normalize_1d(g, eps=eps)
    orig_shape = g.shape
    return _zeropower_via_newtonschulz(g.view(orig_shape[0], -1)).view(orig_shape)


# ---------------------------------------------------------------------------
# Shampoo preconditioner helpers
# ---------------------------------------------------------------------------


def _get_orthogonal_matrix(mat: list) -> list:
    """Compute eigenbases via ``torch.linalg.eigh``."""
    matrix = []
    for m in mat:
        if len(m) == 0:
            matrix.append([])
            continue
        if m.data.dtype != torch.float:
            matrix.append(m.data.float())
        else:
            matrix.append(m.data)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue
        try:
            _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
        except Exception:
            _, Q = torch.linalg.eigh(
                m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device)
            )
            Q = Q.to(m.dtype)
        Q = torch.flip(Q, [1])
        final.append(Q)
    return final


def _get_orthogonal_matrix_qr(
    state: dict,
    max_precond_dim: int = 10000,
) -> list:
    """Compute eigenbases via one power iteration step + ``torch.linalg.qr``."""
    precond_list = state["GG"]
    orth_list = state["Q"]

    matrix, orth_matrix = [], []
    original_types, original_devices = [], []
    float_data_flags = []
    for m, o in zip(precond_list, orth_list):
        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            float_data_flags.append(True)
            original_types.append(None)
            original_devices.append(None)
            continue
        if m.data.dtype != torch.float:
            float_data_flags.append(False)
            original_types.append(m.data.dtype)
            original_devices.append(m.data.device)
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())
        else:
            float_data_flags.append(True)
            original_types.append(None)
            original_devices.append(None)
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())

    final = []
    for float_data, orig_type, orig_device, m, o in zip(
        float_data_flags, original_types, original_devices, matrix, orth_matrix
    ):
        if len(m) == 0:
            final.append([])
            continue
        power_iter = m @ o
        Q, _ = torch.linalg.qr(power_iter)
        if not float_data:
            Q = Q.to(device=orig_device, dtype=orig_type)
        final.append(Q)
    return final