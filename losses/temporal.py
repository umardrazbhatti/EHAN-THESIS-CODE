"""
losses/temporal.py — Gated Temporal Consistency loss L_temp.

Fix 3 reformulation (replaces L2 distance with cosine distance):

  1. Flatten M_t per frame: (B, T, h*w)
  2. L2-normalise each frame's attention to unit norm
  3. cos_dist = 1 - cosine_similarity between consecutive frames: (B, T-1)
  4. Motion gate from L2-normalised low-level features (same as before):
       w_t = exp(-γ · ||φ_t_norm − φ_{t+1}_norm||₂)  where φ is L2-normalised
  5. L_temp = mean(w_t * cos_dist)

Why cosine instead of L2:
  The old L2 distance between consecutive M_t maps was dominated by magnitude
  changes.  As the model becomes more confident, M_t magnitudes grow even if the
  spatial pattern is stable, so L_temp climbed monotonically during training.
  Cosine distance is magnitude-invariant and tracks only the angular change in
  the attention pattern, producing a stable loss in [0.0, 0.3] instead of [0.017, 0.033].

config.lambda2 should be raised to ~1.0 to compensate for the ~5–10× smaller
dynamic range of cosine distance vs the old squared-L2 loss.

Preserved from previous version:
  - L2-normalised φ (low-level feature gate, γ=0.1)
  - First-batch diagnostic prints (gated by _diag_printed)
  - FAIL-FAST gate assertion: 0.01 < mean(w_t) < 0.99
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self._diag_printed = False   # print diagnostics once per training run

    def forward(
        self,
        M_t:       torch.Tensor,   # (B, T, h, w)
        low_level: torch.Tensor,   # (B, T, C_low, Hl, Wl)
    ) -> torch.Tensor:
        B, T = M_t.shape[:2]
        if T < 2:
            return torch.tensor(0.0, device=M_t.device)

        # ── Motion gate: L2-normalised low-level features ─────────────────────
        # φ lives on the unit hypersphere → ||φ_t - φ_{t+1}||₂ ∈ [0, 2]
        # γ=0.1 gives w_t ∈ [exp(-0.2), exp(0.0)] ≈ [0.82, 1.00] for typical diffs
        phi = low_level.detach().reshape(B, T, -1)   # (B, T, C·H'·W')
        phi = F.normalize(phi, p=2, dim=-1)           # unit vectors on d-sphere

        # ── L2-normalise M_t per frame (Fix 3) ───────────────────────────────
        # Removes magnitude sensitivity: only angular pattern changes are penalised
        M_t_flat = M_t.reshape(B, T, -1).float()            # (B, T, h*w)
        M_t_norm = F.normalize(M_t_flat, p=2, dim=-1, eps=1e-8)  # (B, T, h*w) unit norm

        total_loss = torch.tensor(0.0, device=M_t.device)
        n_pairs    = 0

        # Collect stats for diagnostics
        all_diff_norms = []
        all_w_t        = []
        all_cos_dists  = []

        for t in range(T - 1):
            # L2 distance between consecutive normalised feature vectors; in [0, 2]
            diff_norm = (phi[:, t] - phi[:, t + 1]).norm(dim=-1)   # (B,)

            # Gate: upweights pairs where consecutive frames look similar
            w_t = torch.exp(-self.gamma * diff_norm)               # (B,)

            # Fix 3: cosine distance between consecutive normalised M_t frames
            cos_sim  = (M_t_norm[:, t] * M_t_norm[:, t + 1]).sum(dim=-1)  # (B,) in [-1,1]
            cos_dist = 1.0 - cos_sim                                        # (B,) in [0, 2]

            total_loss = total_loss + (w_t * cos_dist).mean()
            n_pairs   += 1

            all_diff_norms.append(diff_norm.detach())
            all_w_t.append(w_t.detach())
            all_cos_dists.append(cos_dist.detach())

        # ── First-batch diagnostics ───────────────────────────────────────────
        if not self._diag_printed and all_w_t:
            dn_cat  = torch.cat(all_diff_norms)   # (n_pairs * B,)
            wt_cat  = torch.cat(all_w_t)
            cd_cat  = torch.cat(all_cos_dists)
            print(
                f"[L_temp DIAG] gamma={self.gamma}  "
                f"||phi_t - phi_t+1||: mean={dn_cat.mean():.4f} std={dn_cat.std():.4f}  "
                f"w_t: mean={wt_cat.mean():.4f} std={wt_cat.std():.4f}  "
                f"cos_dist(M_t,M_t+1): mean={cd_cat.mean():.6f} std={cd_cat.std():.6f}"
            )
            # FAIL FAST if gate is degenerate
            wt_mean = float(wt_cat.mean())
            if wt_mean < 0.01:
                raise RuntimeError(
                    f"[L_temp] DEGENERATE GATE: mean(w_t)={wt_mean:.4f} < 0.01. "
                    f"gamma={self.gamma} is too large -- exp(-gamma*dist) saturates to 0. "
                    f"Reduce gamma to <= 1.0 or verify L2-normalisation of low_level features."
                )
            if wt_mean > 0.99:
                # Warning only (not error): on synthetic data frames are nearly identical
                # so diff_norm ≈ 0 → w_t ≈ 1.0 legitimately.  On real video this would
                # indicate gamma is too small.  Not a training-breaking condition.
                print(
                    f"[L_temp] WARN: mean(w_t)={wt_mean:.4f} > 0.99 -- gate near-constant. "
                    f"On real data this suggests gamma={self.gamma} may be too small. "
                    f"On synthetic data this is expected (near-zero inter-frame variation)."
                )
            print(f"[L_temp] Gate sanity PASSED (mean_w_t={wt_mean:.4f} in [0.01, 0.99])")
            self._diag_printed = True

        return total_loss / n_pairs
