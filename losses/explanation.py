"""
losses/explanation.py — Explanation losses:

  ExplanationLoss   : alpha*Entropy(M_t) + beta*TV(M_t) + diversity_weight*JSD
  DiversityLoss     : Phase 20 Fix 2 — penalises pairwise cosine similarity of M_t
                      across samples in the batch (trains inter_sample_cosine down)
  FaithfulnessLoss  : Phase 20 Fix 3 — MSE between M_t and gradient saliency map
                      (trains M_t to align with what actually affects the logit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExplanationLossOutput:
    loss:            torch.Tensor
    l_h:             float   # entropy term
    l_tv:            float   # total-variation term
    l_div:           float   # inter-sample diversity term
    inter_sample_sim: float  # mean pairwise cosine similarity (diagnostic)


class ExplanationLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, beta: float = 0.5,
                 diversity_weight: float = 2.5):
        super().__init__()
        self.alpha            = alpha
        self.beta             = beta
        self.diversity_weight = diversity_weight

    def forward(
        self,
        M_t: torch.Tensor,   # (B, T, h, w)  normalised to [0,1]
    ) -> ExplanationLossOutput:
        B, T, h, w = M_t.shape
        loss = M_t.new_zeros(1).squeeze()

        l_h_acc  = 0.0
        l_tv_acc = 0.0

        for i in range(B):
            m_avg = M_t[i].mean(0)   # (h, w)

            # Sparsity via entropy
            m_flat  = m_avg.clamp(1e-8, 1 - 1e-8).flatten()
            entropy = -(m_flat * m_flat.log()).sum()

            # Smoothness via total variation
            tv_h = (M_t[i, :, :, 1:] - M_t[i, :, :, :-1]).abs().mean()
            tv_w = (M_t[i, :, 1:, :] - M_t[i, :, :-1, :]).abs().mean()
            tv   = tv_h + tv_w

            loss     = loss + (self.alpha * entropy + self.beta * tv)
            l_h_acc  += entropy.item()
            l_tv_acc += tv.item()

        loss = loss / B

        # Inter-sample diversity — Jensen-Shannon divergence.
        import math as _math
        N   = B * T
        eye = torch.eye(N, dtype=torch.bool, device=M_t.device)
        n_pairs = N * (N - 1)

        eps = 1e-8
        P = M_t.reshape(N, h * w) + eps
        P = P / P.sum(dim=-1, keepdim=True)

        log_P = P.log()
        P_i   = P.unsqueeze(1)
        P_j   = P.unsqueeze(0)
        M_mix = 0.5 * (P_i + P_j)
        log_M = M_mix.log()
        log_P_i = log_P.unsqueeze(1)
        log_P_j = log_P.unsqueeze(0)
        kl_im = (P_i * (log_P_i - log_M)).sum(dim=-1)
        kl_jm = (P_j * (log_P_j - log_M)).sum(dim=-1)
        js_matrix = 0.5 * (kl_im + kl_jm)

        js_off = js_matrix.masked_fill(eye, 0.0)
        mean_js_tensor = js_off.sum() / max(n_pairs, 1)
        log2 = _math.log(2.0)
        l_div_tensor = (log2 - mean_js_tensor).clamp_min(0.0)
        loss = loss + self.diversity_weight * l_div_tensor

        # Cosine similarity kept as diagnostic.
        flat = M_t.reshape(N, h * w)
        flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        cos_matrix = flat @ flat.T
        inter_sample_sim = float(
            cos_matrix.masked_fill(eye, 0.0).sum().item() / (n_pairs + 1e-8)
        )

        return ExplanationLossOutput(
            loss=loss,
            l_h=l_h_acc / max(B, 1),
            l_tv=l_tv_acc / max(B, 1),
            l_div=float(l_div_tensor.item()),
            inter_sample_sim=inter_sample_sim,
        )


# ── Phase 20 Fix 2: Inter-Sample Diversity Loss ───────────────────────────────

class DiversityLoss(nn.Module):
    """
    Penalises pairwise cosine similarity between per-sample explanation maps.

    This is the training-time analogue of the eval metric inter_sample_cosine_mean.
    The loss directly targets the Phase 20 Diagnosed Problem (1): inter_sample_cos
    of 0.9996-0.9999 across all forks.

    Loss = mean of off-diagonal entries of the (B, B) cosine-similarity matrix.
    We want this SMALL (maps should be diverse across samples).

    With the content-adaptive query from Fix 1, this loss provides additional
    gradient pressure to keep the per-sample queries input-specific.
    """

    def forward(self, M_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M_t : (B, T, h, w) — explanation maps
        Returns:
            scalar diversity loss (mean off-diagonal cosine similarity)
        """
        B = M_t.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=M_t.device, requires_grad=False)

        # Flatten each sample's full T*h*w map into a single vector
        flat = M_t.view(B, -1).float()           # (B, T*h*w)

        # L2-normalise per sample
        flat_norm = F.normalize(flat, dim=1, eps=1e-8)  # (B, T*h*w)

        # Pairwise cosine similarity matrix (B, B)
        sim_matrix = flat_norm @ flat_norm.T      # (B, B)

        # Mask the diagonal (self-similarity = 1.0, not informative)
        mask = ~torch.eye(B, dtype=torch.bool, device=M_t.device)

        # Loss = mean off-diagonal similarity (minimise -> maps stay diverse)
        loss = sim_matrix[mask].mean()
        return loss


# ── Phase 20 Fix 3: Gradient-Faithfulness Loss ───────────────────────────────

class FaithfulnessLoss(nn.Module):
    """
    Forces M_t to align with gradient-based saliency during training.

    Closes the gap between "where the attention looks" (M_t) and "what actually
    affects the classification output" (gradient of logit w.r.t. spatial features).

    Phase 19 faithfulness_corr was 0.026-0.088 (near-zero); target > 0.15.

    Algorithm:
      1. Compute grad_S = d(logit.sum())/d(S)  via torch.autograd.grad
         create_graph=False: gradient is a fixed target, no higher-order graph.
         retain_graph=True:  main backward still needs the forward graph.
      2. Aggregate gradient magnitude under torch.no_grad():
         grad_map = grad_S.abs().mean(feature_dim) -> (B, T, h, w)
      3. Normalise grad_map to [0,1] per sample (min-max) — detached target
      4. Normalise M_t to [0,1] per sample (min-max) — gradient flows through M_t
      5. MSE(M_t_norm, g_norm) — only M_t receives gradients

    Requirements:
      - S must have requires_grad=True.  Set S.requires_grad_(True) in EAHN.forward()
        when use_faithfulness_loss=True and model.training.
      - The main backward call should use retain_graph=True when this loss is
        active (FaithfulnessLoss uses retain_graph=True internally, so the graph
        is preserved for the subsequent loss.backward()).
    """

    def forward(
        self,
        M_t:   torch.Tensor,   # (B, T, h, w)  — explanation maps
        logit: torch.Tensor,   # (B,)           — classification logit (keep grad)
        S:     torch.Tensor,   # (B, T, hw, d)  — spatial features (requires_grad=True)
    ) -> torch.Tensor:
        B, T, h, w = M_t.shape

        # Guard: S must have requires_grad=True (set explicitly in EAHN.forward)
        if not S.requires_grad:
            return torch.tensor(0.0, device=M_t.device)

        try:
            # ── Step 1: compute first-order gradient (no higher-order graph) ──
            # create_graph=False  → CRITICAL: prevents second-order graph materialisation
            #                       that caused CUDA OOM on T4 at epoch 6.
            # retain_graph=True   → main loss.backward() needs the forward graph.
            # only_inputs=True    → don't accumulate .grad on leaf params (side-effect free).
            grad_S, = torch.autograd.grad(
                outputs=logit.sum(),
                inputs=S,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )

            if grad_S is None:
                return torch.tensor(0.0, device=M_t.device)

            # ── Step 2: build normalised saliency target (no grad needed) ────
            with torch.no_grad():
                grad_map = grad_S.abs().mean(dim=-1).view(B, T, h, w)
                g_flat   = grad_map.view(B, T, -1)
                g_min    = g_flat.min(dim=-1, keepdim=True)[0]
                g_max    = g_flat.max(dim=-1, keepdim=True)[0]
                g_norm   = ((g_flat - g_min) / (g_max - g_min + 1e-8)).view(B, T, h, w)

            # ── Step 3: normalise M_t — gradient flows through this branch ───
            m_flat = M_t.view(B, T, -1)
            m_min  = m_flat.min(dim=-1, keepdim=True)[0]
            m_max  = m_flat.max(dim=-1, keepdim=True)[0]
            m_norm = ((m_flat - m_min) / (m_max - m_min + 1e-8)).view(B, T, h, w)

            # ── Step 4: MSE teaches M_t to agree with gradient saliency ──────
            return F.mse_loss(m_norm, g_norm)

        except RuntimeError as _e:
            print(f"[FaithfulnessLoss] WARNING: skipped this step ({_e})")
            return torch.tensor(0.0, device=M_t.device)
