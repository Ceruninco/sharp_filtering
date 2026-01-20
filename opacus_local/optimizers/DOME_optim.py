"""
DOME_optim.py

Refactor summary (requested):
- Gradient-subspace-fractions logging is no longer hard-coded to dataset-specific filenames.
- Logging depends on the same run parameters as the launcher (dataset, arch, seed, remove_dom, debias,
  compression_rate, nb_dims_pca, DP knobs, etc.).
- The launcher injects logging config into the optimizer via AdamCorr.set_logging(...).
- No import of launcher DATASET_CONFIG inside this optimizer (avoids circular imports).
"""

import os.path
import numpy
import pandas as pd
import torch
import math
from torch import Tensor
from typing import List, Optional, Dict
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy
import pickle
import numpy as np
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

import os
import csv


# ------------------------------------------------------------------------
# Logging helpers (NEW)
# ------------------------------------------------------------------------

FRACTION_FIELDS = [
    "dataset",
    "arch",
    "seed",
    "remove_dom",
    "debias",
    "compression_rate",
    "nb_dims_pca",
    "disable_dp",
    "sigma",
    "max_per_sample_grad_norm",
    "step",
    "fraction",
    "fraction_mean",
]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _append_row_csv(csv_path: str, fieldnames: List[str], row: Dict) -> None:
    _ensure_parent_dir(csv_path)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def log_gradient_subspace_fractions(
    *,
    grads: torch.Tensor,          # (B, d)
    eigenvectors: torch.Tensor,   # (d, k)
    step: int,
    csv_path: str,
    run_meta: Dict,
) -> None:
    """
    Log per-step gradient energy fractions inside the subspace spanned by eigenvectors.
    Each row includes metadata so many runs can share one CSV.
    """
    if grads is None or grads.numel() == 0:
        return
    if eigenvectors is None or eigenvectors.numel() == 0:
        return

    grads_mean = grads.mean(dim=0)

    # Project onto subspace
    proj_grads = (grads @ eigenvectors) @ eigenvectors.t()
    proj_mean = (grads_mean @ eigenvectors) @ eigenvectors.t()

    grad_norms = grads.norm(dim=1).clamp_min(1e-12)
    proj_norms = proj_grads.norm(dim=1)

    fraction = (proj_norms / grad_norms).mean().item()
    fraction_mean = (proj_mean.norm() / grads_mean.norm().clamp_min(1e-12)).item()

    row = dict(run_meta) if run_meta is not None else {}
    row.update(
        {
            "step": int(step),
            "fraction": float(fraction),
            "fraction_mean": float(fraction_mean),
        }
    )
    _append_row_csv(csv_path, FRACTION_FIELDS, row)


# ------------------------------------------------------------------------
# Existing helpers (unchanged except for style)
# ------------------------------------------------------------------------

def _tmp_get_summary_stats(vec):
    return {
        "min": torch.min(vec), "max": torch.max(vec),
        "mean": torch.mean(vec), "median": torch.quantile(vec, 0.50),
        "q1": torch.quantile(vec, 0.25), "q3": torch.quantile(vec, 0.75)
    }


def plot_eigenvalues(eigenvalues):
    eigenvalues = eigenvalues[:200]
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 14,
        "axes.titlesize": 10,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 4.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
    })

    if hasattr(eigenvalues, "detach"):
        lamb = eigenvalues.detach().cpu().numpy().ravel()
    else:
        lamb = np.asarray(eigenvalues).ravel()

    lamb_sorted = np.sort(lamb[np.isfinite(lamb)])[::-1]
    lamb_sorted = lamb_sorted[lamb_sorted > 0]

    df = pd.DataFrame({
        "Rank": np.arange(1, len(lamb_sorted) + 1),
        "Eigenvalue": lamb_sorted
    })

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Rank", y="Eigenvalue", ax=ax)
    ax.set_yscale("log")

    ax.axvline(10, linestyle=":", linewidth=2.5)
    ax.axvline(100, linestyle=":", linewidth=2.5)

    ax.set_xlabel("Rank", labelpad=2)
    ax.set_ylabel("Eigenvalue", labelpad=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linewidth=0.6)
    ax.grid(True, which="minor", linewidth=0.4)

    plt.tight_layout()
    plt.savefig("mnist_spectrum.pdf")
    plt.show()


def clip_aggregate_noise_fixed(gradients, dp_scale):
    norms = torch.norm(gradients, dim=1)
    eps = 1e-6
    C = 1.0
    clip_coef = C / (norms + eps)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    gradients = gradients * clip_coef.unsqueeze(1)
    b_size = gradients.shape[0]
    gradients = gradients.mean(dim=0)

    z = dp_scale * C / b_size
    dp_noise = z * torch.randn_like(gradients)
    gradients += dp_noise
    return gradients, z


def clip_fixed(gradients, dp_scale):
    norms = torch.norm(gradients, dim=1)
    eps = 1e-6
    C = 1.0
    clip_coef = C / (norms + eps)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    gradients = gradients * clip_coef.unsqueeze(1)
    return gradients


def unflatten_grads(flat, metadata):
    grads_reconstructed = []
    idx = 0
    for shape, numel in metadata:
        chunk = flat[idx: idx + numel]
        grads_reconstructed.append(chunk.view(shape))
        idx += numel
    return grads_reconstructed


# ------------------------------------------------------------------------
# Functional AdamCorr core (logging removed from here)
# ------------------------------------------------------------------------

def adam_corr(
    params: List[Tensor],
    grads,
    meta,
    grads_clean: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avgs_clean: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    *,
    remove_dom=True,
    norm_ema=0,
    eigenvectors=0,
    eigenvalues=0,
    current_norm=0,
    compression_rate,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    eps: float,
):
    """
    NOTE: All CSV logging has been removed from this function.
    Logging is now performed inside AdamCorr.step(), where run metadata is available.
    """
    use_weird_prec = False

    if remove_dom and state_steps[0] > 1:
        if use_weird_prec:
            in_space = grads @ eigenvectors
            in_space = in_space / eigenvalues.sqrt().clamp(min=1e-8)
            in_space = in_space @ eigenvectors.t()
            out_space = grads - ((grads @ eigenvectors) @ eigenvectors.t())
            current_norm = out_space.norm(dim=1).pow(2).mean()
            past_sq_norm = norm_ema[-1].mean() / (1 - beta2 ** state_steps[-1])
            parameter_wise = past_sq_norm / grads.shape[1]
            out_space = out_space / parameter_wise.sqrt()
            flat_vec_filtered = in_space + out_space
        else:
            flat_vec_filtered = grads - ((grads @ eigenvectors) @ eigenvectors.t())
            current_norm = grads.norm(dim=1).pow(2).mean()
    else:
        if use_weird_prec:
            current_norm = grads.norm(dim=1).pow(2).mean()
            flat_vec_filtered = grads
        else:
            current_norm = grads.norm(dim=1).pow(2).mean()
            flat_vec_filtered = grads

    grads_aggregated = flat_vec_filtered.mean(dim=0)
    if compression_rate > 1:
        nb_new_sketches = grads_aggregated.shape[0] // compression_rate
        random_probes = (1.0 / math.sqrt(nb_new_sketches)) * torch.randn(
            (grads_aggregated.shape[0], nb_new_sketches), device="cuda:0"
        )
        grads_aggregated = grads_aggregated @ random_probes
        grads_aggregated = grads_aggregated @ random_probes.t()

    restored = unflatten_grads(grads_aggregated, meta)

    for i, param in enumerate(params):
        step = state_steps[i]
        grad = restored[i]
        grad_clean = grads_clean[i]

        exp_avg = exp_avgs[i]
        exp_avg_clean = exp_avgs_clean[i]

        exp_avg_sq = exp_avg_sqs[i]
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_clean.mul_(beta1).add_(grad_clean, alpha=1 - beta1)

        list_element = norm_ema[i]
        list_element.mul_(beta2).add_((1 - beta2) * current_norm)

        use_belief = False
        if not use_belief:
            iter_grad = (1 - beta2) * (grad * grad.conj())
        else:
            bias_correction2 = 1 - beta2 ** step
            exp_avg_hat = torch.divide(exp_avg_sq, bias_correction2)
            iter_grad = (1 - beta2) * (((grad - exp_avg_hat) * (grad - exp_avg_hat).conj())) + 1e-8

        exp_avg_sq.mul_(beta2).add_(iter_grad)

    for i, param in enumerate(params):
        step = state_steps[i]
        exp_avg = exp_avgs[i]
        grad = restored[i]
        exp_avg_sq = exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        exp_avg_sq_hat = torch.divide(exp_avg_sq, bias_correction2)
        exp_avg_hat = exp_avg / bias_correction1

        if amsgrad:
            raise NotImplementedError
        else:
            denom = (exp_avg_sq_hat.sqrt()).add_(eps)

        param.addcdiv_(exp_avg_hat, denom, value=-lr)

    mt_norm, vt_norm, vt_corr_norm, mt_clean_norm, vt_clean_norm = [torch.nan] * 5
    hist_dict = {}
    summary_stats_dict = {}

    return (
        mt_norm, vt_norm, vt_corr_norm, mt_clean_norm, vt_clean_norm,
        0, 0, hist_dict, summary_stats_dict, step
    )


# ------------------------------------------------------------------------
# Optimizer class
# ------------------------------------------------------------------------

class AdamCorr(Optimizer):
    """Modified from torch's version of Adam"""

    def __init__(
        self,
        params,
        dp_batch_size,
        dp_noise_multiplier,
        dp_l2_norm_clip,
        eps_root,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        gamma_decay=1,
        weight_decay=0,
        amsgrad=False,
        numel=0,
        rank=0,
        total_steps=10,
        random_sketch=False,
        use_sketching=True,
        debias=True,
        compression_rate=1,
        use_preconditioning=True,
        seed=0,
        use_momentum=False,
        use_rescaling=True,
        remove_dom=False,
        nb_dims_pca=100,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamCorr, self).__init__(params, defaults)

        self.dp_batch_size = dp_batch_size
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_l2_norm_clip = dp_l2_norm_clip
        self.eps_root = eps_root
        self.gamma_decay = gamma_decay

        self.numel = numel
        self.lamb_ordered = 0

        self.nb_dims_pca = nb_dims_pca
        self.lamb = torch.zeros((self.nb_dims_pca), device="cuda:0")
        self.remove_dom = remove_dom

        self.variances = torch.ones((1, rank)).cuda() if rank > 0 else torch.ones((1, 1)).cuda()
        self.total_steps = total_steps
        self.useful_dims = 0
        self.beta = 0.5
        self.random_sketching = random_sketch
        self.running_var_random = 0.0
        self.use_sketching = use_sketching
        self.debias = debias
        self.compression_rate = compression_rate
        self.use_preconditioning = use_preconditioning
        self.seed = seed
        self.use_momentum = use_momentum
        self.use_rescaling = use_rescaling

        first_dims = torch.linalg.qr(torch.randn((numel, self.nb_dims_pca), device="cuda:0")).Q
        self.u = first_dims
        self.initial_basis = first_dims

        # --- logging config injected by launcher ---
        self._fractions_csv_path: Optional[str] = None
        self._run_meta: Dict = {}
        self._log_fractions_every: int = 1
        self._enable_fraction_logging: bool = True

    def set_logging(
        self,
        *,
        fractions_csv_path: Optional[str],
        run_meta: Dict,
        log_fractions_every: int = 1,
        enable_fraction_logging: bool = True,
    ) -> None:
        """
        Called by the launcher to ensure optimizer-side logging uses the same run identifiers.
        """
        self._fractions_csv_path = fractions_csv_path
        self._run_meta = dict(run_meta) if run_meta is not None else {}
        self._log_fractions_every = max(1, int(log_fractions_every))
        self._enable_fraction_logging = bool(enable_fraction_logging)

    def __setstate__(self, state):
        super(AdamCorr, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def flatten_grads(self, grads):
        metadata = [(g.shape, g.numel()) for g in grads]
        flat = torch.cat([g.contiguous().view(-1) for g in grads], dim=0)
        return flat, metadata

    def flatten_batch_grads(self, grads):
        B = grads[0].shape[0]
        metadata = [(g.shape[1:], g[0].numel()) for g in grads]
        flat_per_layer = [g.contiguous().view(B, -1) for g in grads]
        flat = torch.cat(flat_per_layer, dim=1)
        return flat, metadata

    def unflatten_batch_grads(self, flat, metadata):
        B, D = flat.shape
        grads = []
        idx = 0
        for shape, numel in metadata:
            seg = flat[:, idx:idx + numel]
            grads.append(seg.view(B, *shape))
            idx += numel
        return grads

    # -------------------------- subspace updates --------------------------

    def update_dom(self, flat_vec, u, lamb, step):
        cov_vector = flat_vec
        standard_version = True
        if standard_version:
            remove_mean = True
            if remove_mean:
                mean = cov_vector.mean(dim=0)
                cov_vector = cov_vector - mean
            Y_curr = torch.matmul(cov_vector.t(), cov_vector @ u)
        else:
            mean = cov_vector.mean(dim=0)
            Y_curr = torch.outer(mean, mean @ u)
            Y_curr = u - 0.001 * Y_curr

        y = ((step - 1) / step) * (u * lamb) + (1 / step) * Y_curr / flat_vec.shape[0]
        q, r = torch.linalg.qr(y)
        squared = torch.norm(y, dim=0)
        sorted_eigenvalues = torch.sort(squared, descending=True).values
        columns_indices_full = torch.sort(squared, descending=True).indices[: self.nb_dims_pca]
        sorted_u = q[:, columns_indices_full]
        return sorted_u, sorted_eigenvalues

    # -------------------------- clipping helpers --------------------------

    def clip_aggregate_fixed(self, gradients):
        norms = torch.norm(gradients, dim=1)
        eps = 1e-6
        C = 1.0
        clip_coef = C / (norms + eps)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        gradients = gradients * clip_coef.unsqueeze(1)
        gradients = gradients.mean(dim=0)
        return gradients

    def clip_fixed(self, gradients):
        norms = torch.norm(gradients, dim=1)
        eps = 1e-6
        C = 1.0
        clip_coef = C / (norms + eps)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        gradients = gradients * clip_coef.unsqueeze(1)
        return gradients

    def aggregate_noise_fixed(self, gradients):
        norms = torch.norm(gradients, dim=1)
        eps = 1e-6
        C = 1.0
        b_size = gradients.shape[0]
        gradients = gradients.mean(dim=0)

        z = self.dp_scale * C / b_size
        dp_noise = z * torch.randn_like(gradients)
        gradients += dp_noise
        return gradients, z

    # ------------------------------ step ------------------------------

    @torch.no_grad()
    def step(self, grad_samples, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            mean_clipped_grads = []
            indiv_grad_list = []
            exp_avgs = []
            exp_avgs_clean = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            norm_ema = []
            state_steps = []

            beta1, beta2 = group["betas"]

            for p, indiv_grads in zip(group["params"], grad_samples):
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_clean"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["norm_ema"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state["exp_avg"])
                    exp_avgs_clean.append(state["exp_avg_clean"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    norm_ema.append(state["norm_ema"])

                    mean_clipped_grad = indiv_grads.mean(dim=0)
                    indiv_grad_list.append(indiv_grads)
                    mean_clipped_grads.append(mean_clipped_grad)

                    state["step"] += 1
                    state_steps.append(state["step"])

            if len(indiv_grad_list) == 0:
                continue

            flat_vec, meta = self.flatten_batch_grads(indiv_grad_list)
            flat_mean = flat_vec.mean(dim=0)
            mean_clean = unflatten_grads(flat_mean, meta)

            # (Optional) update subspace estimator (kept consistent with your original behavior)
            old_eigenvalues, old_eigenvectors = self.lamb, self.u
            self.u, self.lamb = self.update_dom(flat_vec, self.u, self.lamb, state["step"])

            # NEW: metadata-aware logging (no hard-coded filename)
            if (
                self._enable_fraction_logging
                and self._fractions_csv_path is not None
                and state["step"] > 1
                and (state["step"] % self._log_fractions_every == 0)
            ):
                # Merge run meta with optimizer-truth fields
                run_meta = dict(self._run_meta) if self._run_meta is not None else {}
                run_meta.update(
                    {
                        "remove_dom": bool(self.remove_dom),
                        "debias": bool(self.debias),
                        "compression_rate": int(self.compression_rate),
                        "nb_dims_pca": int(self.nb_dims_pca),
                    }
                )
                # Log energy in the current estimated nuisance subspace (self.u)
                log_gradient_subspace_fractions(
                    grads=flat_vec,
                    eigenvectors=self.u,
                    step=state["step"],
                    csv_path=self._fractions_csv_path,
                    run_meta=run_meta,
                )

            # Use functional AdamCorr update
            _ = adam_corr(
                params_with_grad,
                flat_vec,
                meta,
                mean_clean,
                exp_avgs,
                exp_avgs_clean,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                remove_dom=self.remove_dom,
                norm_ema=norm_ema,
                eigenvectors=old_eigenvectors,
                eigenvalues=old_eigenvalues,
                current_norm=0,
                compression_rate=self.compression_rate,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                eps=group["eps"],
            )

        return torch.nan, torch.nan, torch.nan, torch.nan, torch.nan
