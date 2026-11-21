# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Execution-Policy Robust Optimization (EPRO): transport, score centering, and SSC.

Policies on one prefix: ``q`` = mu_0 (quantized sampler that produced the rollout), ``p0`` = pi_0
(BF16 trainer at the rollout checkpoint), ``u`` = pi_theta (current BF16 trainer). The current
quantized execution policy is predicted by transport, ``mhat ∝ q * u / p0``. Only the sampler's
top-k head H is recorded; its tail is proxied by the old trainer tail, ``q_tail(v) = rho0 * p0(v)``
with ``rho0 = (1 - q(H)) / (1 - p0(H))``, which gives

    Z     = sum_H q u / p0 + rho0 * (1 - u(H))
    mhat  = q u / (p0 Z) on H,  rho0 u / Z off H
    E_mhat[s] = sum_H (mhat_v - (rho0 / Z) u_v) s_v          (uses E_u[s] = 0)

so ``correction = sum_H sg(mhat_v - rho0 u_v / Z) log u_v`` has gradient ``E_mhat[s]``. Under the
same proxy, a sampled token outside H has ``mhat(a) = rho0 u_a / Z`` and ``q~(a) = rho0 p0_a``, so
``log w = log u_a - log p0_a - log Z`` and ``grad log mhat(a) = s(a) - E_mhat[s]`` hold for every a.

Loss (semi-gradient, per token): ``-sg(A * min(w, C) * kappa) * (log u_a - correction)``.
SSC attenuates outward updates using Policy Shift: ``kappa = exp(-relu(D - tau) / sigma)``;
non-outward updates retain ``kappa = 1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

TSC_TAIL_FLOOR = 1e-8


def transported_head_terms(
    train_head_log_probs: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    old_head_log_probs: torch.Tensor,
    tail_floor: float = TSC_TAIL_FLOOR,
    head_complete: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-row transport quantities on an incomplete recorded head.

    Args:
        train_head_log_probs: log u on H, full-vocab normalized, shape (N, k), with gradient.
        sampler_head_log_probs: log q on H (rollout sampler), shape (N, k).
        old_head_log_probs: log p0 on H (old BF16 trainer, same rollout checkpoint), shape (N, k).
        tail_floor: Numerical floor for tail masses.
        head_complete: True only when the head covers the entire vocabulary; omit the tail exactly.

    Returns:
        Dict of (N,) tensors: ``correction`` (gradient = E_mhat[s]), ``log_z`` (detached),
        ``rho`` (detached), ``q_head_mass``, ``p0_head_mass``, ``u_head_mass`` (detached).
    """
    log_u = train_head_log_probs.float()
    log_q = sampler_head_log_probs.detach().float()
    log_p0 = old_head_log_probs.detach().float()
    with torch.no_grad():
        u = log_u.exp()
        q_mass = log_q.exp().sum(-1)
        p0_mass = log_p0.exp().sum(-1)
        u_mass = u.sum(-1)
        rho = (1.0 - q_mass).clamp_min(tail_floor) / (1.0 - p0_mass).clamp_min(tail_floor)
        if head_complete:
            rho = torch.zeros_like(rho)
        head_terms = log_q + log_u - log_p0
        z = head_terms.exp().sum(-1) + rho * (1.0 - u_mass).clamp_min(tail_floor)
        log_z = z.log()
        mhat_head = (head_terms - log_z.unsqueeze(-1)).exp()
        coeff = mhat_head - (rho / z).unsqueeze(-1) * u
    correction = (coeff * log_u).sum(-1)
    return {
        "correction": correction,
        "log_z": log_z,
        "rho": rho,
        "q_head_mass": q_mass,
        "p0_head_mass": p0_mass,
        "u_head_mass": u_mass,
    }


DIV_KINDS = ("tv", "kl", "lrms", "ln2", "lninf")
TSC_NUCLEUS_FLOOR = 1e-3  # unweighted logit-norm support: head tokens with p0 >= floor (plus the top-2 of p0)


def head_divergences(
    train_head_log_probs: torch.Tensor,
    old_head_log_probs: torch.Tensor,
    tail_floor: float = TSC_TAIL_FLOOR,
    nucleus_floor: float = TSC_NUCLEUS_FLOOR,
) -> dict[str, torch.Tensor]:
    """Per-position divergence between the old bf16 policy p0 and the current u.

    Support = recorded sampler head H plus one "tail" bucket holding the rest of the mass (DPPO top-K style).
    Returns detached (N,) tensors: ``tv`` = 1/2 sum |u - p0|, ``kl`` = KL(p0 || u), and ``lrms`` = the p0-weighted
    RMS of the centered logit shift, sqrt(Var_p0[log u - log p0]) (~ sqrt(2 KL) for small steps; units: nats).
    Additional unweighted logit-vector diagnostics on the support S = {v in H: p0_v >= nucleus_floor} plus the
    top-2 of p0: ``ln2`` = uniform RMS and ``lninf`` = max of |dl_v - mean_S dl| with dl = log u - log p0.
    """
    with torch.no_grad():
        log_u = train_head_log_probs.detach().float()
        log_p0 = old_head_log_probs.detach().float()
        u, p0 = log_u.exp(), log_p0.exp()
        u_t = (1.0 - u.sum(-1)).clamp_min(tail_floor)
        p0_t = (1.0 - p0.sum(-1)).clamp_min(tail_floor)
        tv = 0.5 * ((u - p0).abs().sum(-1) + (u_t - p0_t).abs())
        kl = (p0 * (log_p0 - log_u)).sum(-1) + p0_t * (p0_t.log() - u_t.log())
        dl = torch.cat([log_u - log_p0, (u_t.log() - p0_t.log()).unsqueeze(-1)], -1)
        w = torch.cat([p0, p0_t.unsqueeze(-1)], -1)
        w = w / w.sum(-1, keepdim=True)
        mean = (w * dl).sum(-1, keepdim=True)
        lrms = (w * (dl - mean).square()).sum(-1).clamp_min(0.0).sqrt()
        dlh = log_u - log_p0
        sup = p0 >= nucleus_floor
        top2 = p0.topk(min(2, p0.shape[-1]), -1).indices
        sup = sup.scatter(-1, top2, True)
        ns = sup.sum(-1, keepdim=True).float()
        mu = (dlh * sup).sum(-1, keepdim=True) / ns
        dev = (dlh - mu).abs() * sup
        ln2 = (dev.square().sum(-1) / ns.squeeze(-1)).sqrt()
        lninf = dev.amax(-1)
    return {"tv": tv, "kl": kl.clamp_min(0.0), "lrms": lrms, "ln2": ln2, "lninf": lninf}


def transported_token_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    correction: torch.Tensor,
    log_z: torch.Tensor,
    advantages: torch.Tensor,
    tis_cap: float | None = 2.0,
    div: torch.Tensor | None = None,
    div_delta: float = 0.125,
    div_scale: float = 0.062,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """EPRO's transported-score surrogate with Shift-Aware Soft Clipping.

    ``div`` is bucketed Policy Shift, sqrt(Var_p0[log u - log p0]). SSC uses
    exp(-relu(div - tau)/sigma) only when A * log(u(a)/p0(a)) > 0.
    The advantage, importance weight, and attenuation are held fixed during differentiation.
    """
    if tis_cap is not None and tis_cap <= 0:
        raise ValueError("tis_cap must be positive or None")
    if div is None:
        raise RuntimeError("EPRO needs Policy Shift (tsc_div_lrms) from the logits processor.")
    if div_scale <= 0:
        raise ValueError("SSC scale must be positive")
    with torch.no_grad():
        log_r = log_prob.detach().float() - old_log_prob.float()
        log_w = log_r - log_z.float()
        w = log_w.clamp(max=20.0).exp()
        w_capped = w.clamp(max=tis_cap) if tis_cap is not None else w
        a = advantages.detach().float()
        outward = ((a > 0) & (log_r > 0)) | ((a < 0) & (log_r < 0))
        excess = (div.float() - div_delta).clamp_min(0.0)
        gate = torch.where(outward, (-excess / div_scale).exp(), torch.ones_like(excess))
        attenuated = outward & (excess > 0)
        scale = a * w_capped * gate
    token_loss = -scale * (log_prob.float() - correction.float())
    return token_loss, {
        "log_r": log_r, "w": w, "w_capped": w_capped,
        "blocked": attenuated.float(), "gate": gate,
    }


TSC_COLLECT_FLAG = "tsc_collect_old_head"
TSC_OLD_HEAD_KEY = "tsc_old_topk_log_probs"


def transported_sc_logits_processor(
    student_logits: torch.Tensor,
    data: TensorDict,
    data_format: str = "thd",
) -> dict[str, torch.Tensor]:
    """Engine logits-processor slot (thd layout, logits rows aligned with the rollout head rows).

    Collect mode (old BF16 forward, before any update of this batch): returns the old trainer's
    full-vocab log-probs on the sampler head, ``tsc_old_topk_log_probs`` of shape (1, nnz/sp, k).
    Train mode: returns per-token transport quantities, each of shape (1, nnz/sp).
    """
    from verl.utils import tensordict_utils as tu
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor
    from verl.trainer.ppo.score_centering import topk_log_probs_from_logits

    if data_format != "thd":
        raise NotImplementedError("transported SC supports the thd (remove padding) format only.")
    topk_ids = data["rollout_topk_ids"].values().unsqueeze(0)
    sp = get_ulysses_sequence_parallel_world_size() > 1
    if sp:
        topk_ids = slice_input_tensor(topk_ids, dim=1)
    head = topk_log_probs_from_logits(student_logits.squeeze(0), topk_ids.squeeze(0))
    if tu.get_non_tensor_data(data=data, key=TSC_COLLECT_FLAG, default=False):
        return {TSC_OLD_HEAD_KEY: head.detach().unsqueeze(0)}

    # v1 handles incomplete heads only (full-vocab sampling). Synthetic padding rows carry a dummy uniform
    # head marked support-complete; they are masked out of the loss, and the tail floors keep them finite.
    q_head = data["rollout_topk_log_probs"].values().unsqueeze(0)
    if sp:
        q_head = slice_input_tensor(q_head, dim=1)
    p0_head = data[TSC_OLD_HEAD_KEY].values().unsqueeze(0)
    if sp:
        p0_head = slice_input_tensor(p0_head, dim=1)
    t = transported_head_terms(
        head, q_head.squeeze(0), p0_head.squeeze(0), head_complete=head.shape[-1] == student_logits.shape[-1]
    )
    dv = head_divergences(head, p0_head.squeeze(0))
    extra = {f"tsc_div_{kind}": dv[kind].unsqueeze(0) for kind in DIV_KINDS}
    return {
        **extra,
        "tsc_correction": t["correction"].unsqueeze(0),
        "tsc_log_z": t["log_z"].unsqueeze(0),
        "tsc_rho": t["rho"].unsqueeze(0),
        "tsc_q_head_mass": t["q_head_mass"].unsqueeze(0),
        "tsc_p0_head_mass": t["p0_head_mass"].unsqueeze(0),
        "tsc_u_head_mass": t["u_head_mass"].unsqueeze(0),
    }


def transported_sc_ppo_loss(
    config, model_output: dict = None, data: TensorDict = None, dp_group=None, student_logits=None, data_format="thd"
):
    """Loss entry for the engine: logits processor when ``student_logits`` is given, else the loss.

    Collect mode returns a zero loss (forward-only old-log-prob pass); train mode defers to ``ppo_loss``,
    which passes ``tsc_correction`` / ``tsc_log_z`` to the ``transported_sc`` policy loss.
    """
    from verl.utils import tensordict_utils as tu

    if student_logits is not None:
        return transported_sc_logits_processor(student_logits, data, data_format)
    if tu.get_non_tensor_data(data=data, key=TSC_COLLECT_FLAG, default=False):
        return torch.zeros((), device=model_output["log_probs"].values().device), {}
    if "tsc_correction" not in model_output:
        raise RuntimeError("transported SC enabled but the logits processor did not run (score_centering flag unset).")
    from verl.workers.utils.losses import ppo_loss

    return ppo_loss(config, model_output, data, dp_group)


def compute_policy_loss_transported_sc(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config=None,
    rollout_is_weights: torch.Tensor | None = None,
    tsc_correction: torch.Tensor | None = None,
    tsc_log_z: torch.Tensor | None = None,
    **kwargs,
):
    """Transported SC policy loss. ``old_log_prob`` must be the recomputed BF16 pi_0 (decoupled mode)."""
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.utils import torch_functional as verl_F

    if tsc_correction is None or tsc_log_z is None:
        raise RuntimeError("transported_sc loss needs tsc_correction and tsc_log_z from the logits processor.")
    if rollout_is_weights is not None:
        raise ValueError("transported_sc applies its own TIS on mhat/mu0; set algorithm.rollout_correction.rollout_is=null.")
    rc = config.policy_loss.get("rollout_correction", None) or {}
    cap = rc.get("tsc_tis_cap", 2.0)
    token_loss, d = transported_token_loss(
        log_prob, old_log_prob, tsc_correction, tsc_log_z, advantages, tis_cap=cap,
        div=kwargs.get("tsc_div_lrms"), div_delta=rc.get("tsc_div_delta", 0.125),
        div_scale=rc.get("tsc_div_scale", 0.062),
    )
    pg_loss = agg_loss(loss_mat=token_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    m = response_mask
    pos, neg = m & (advantages > 0), m & (advantages < 0)
    metrics = {
        "actor/tsc_log_z": verl_F.masked_mean(tsc_log_z.float(), m).item(),
        "actor/tsc_w_mean": verl_F.masked_mean(d["w"], m).item(),
        "actor/tsc_w_capped_frac": verl_F.masked_mean((d["w"] > cap).float(), m).item() if cap is not None else 0.0,
        "actor/tsc_gate_blocked_frac": verl_F.masked_mean(d["blocked"], m).item(),
        "actor/tsc_gate_blocked_pos": verl_F.masked_mean(d["blocked"], pos).item() if pos.any() else 0.0,
        "actor/tsc_gate_blocked_neg": verl_F.masked_mean(d["blocked"], neg).item() if neg.any() else 0.0,
        "actor/ppo_kl": verl_F.masked_mean(-d["log_r"], m).item(),
        "actor/pg_clipfrac": verl_F.masked_mean(d["blocked"], m).item(),
        "actor/pg_clipfrac_lower": verl_F.masked_mean(d["blocked"], neg).item() if neg.any() else 0.0,
    }
    metrics["actor/tsc_gate_weight_mean"] = verl_F.masked_mean(d["gate"], m).item()
    for kind in DIV_KINDS:
        value = kwargs.get(f"tsc_div_{kind}")
        if value is not None:
            metrics[f"actor/policy_shift_{kind}"] = verl_F.masked_mean(value.float(), m).item()
    return pg_loss, metrics


def register_transport_loss():
    """Register once when verl constructs its actor worker; math helpers need only PyTorch."""
    from verl.trainer.ppo.core_algos import register_policy_loss

    register_policy_loss("transported_sc")(compute_policy_loss_transported_sc)
