"""
metrics/explanation.py — Explanation quality metrics.

Provides the two intrinsic explanation-quality measures reported in Chapter 4:
  - temporal_ssim          : temporal stability of the explanation maps M_t
  - deletion_insertion_auc : causal faithfulness of the saliency ordering
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


class ExplanationMetrics:

    @staticmethod
    def temporal_ssim(M_t_up: torch.Tensor) -> float:
        """
        Mean SSIM between consecutive explanation frames.
        M_t_up: (N, T, H, W) subset.
        """
        values = []
        N, T, H, W = M_t_up.shape
        for b in range(N):
            for t in range(T - 1):
                a = M_t_up[b, t].cpu().numpy().astype(np.float32)
                b_ = M_t_up[b, t + 1].cpu().numpy().astype(np.float32)
                val = ssim(a, b_, data_range=1.0)
                values.append(val)
        return float(np.mean(values)) if values else 1.0

    @staticmethod
    def deletion_insertion_auc(model, frames, saliency,
                               steps: int = 10) -> dict:
        """
        Deletion/Insertion AUC: simplified implementation.
        Steps are coarse for speed; increase for publication-quality numbers.
        """
        device = next(model.parameters()).device
        B, T, C, H, W = frames.shape
        total_pixels  = H * W

        # The pixel-masking path below uses np.argsort / boolean-mask indexing, so
        # saliency must be a numpy array. Both call sites pass numpy already; this
        # coercion keeps the function correct if a caller passes a torch tensor
        # (otherwise sal.mean(1) stays a tensor and np.argsort(...).copy() raises,
        # silently zeroing the metric).
        if isinstance(saliency, torch.Tensor):
            saliency = saliency.detach().cpu().numpy()

        with torch.no_grad():
            baseline_logit = model(frames.to(device)).prob.mean().item()

        del_scores = []
        ins_scores = []

        # Use mean explanation over time
        sal = saliency.mean(1)   # (B, H, W) or just use first frame

        for step in range(steps + 1):
            frac = step / steps
            k    = max(1, int(frac * total_pixels))

            # Deletion: mask out top-k salient pixels
            del_frames = frames.clone()
            ins_frames = torch.zeros_like(frames)

            for b in range(B):
                flat_sal = sal[b].reshape(-1)                         # np.ndarray
                top_k_idx = np.argsort(flat_sal)[-k:].copy()          # top-k, contiguous
                mask     = np.zeros(H * W, dtype=bool)
                mask[top_k_idx] = True
                mask_2d  = mask.reshape(H, W)

                del_frames[b, :, :, mask_2d] = 0.0
                ins_frames[b, :, :, mask_2d] = frames[b, :, :, mask_2d]

            with torch.no_grad():
                del_score = model(del_frames.to(device)).prob.mean().item()
                ins_score = model(ins_frames.to(device)).prob.mean().item()

            del_scores.append(del_score)
            ins_scores.append(ins_score)

        # numpy 2.x renamed np.trapz -> np.trapezoid and REMOVED np.trapz.
        # A getattr(np, "trapezoid", np.trapz) idiom would still raise on numpy 2.x
        # because the default argument np.trapz is evaluated eagerly. Select by
        # name so this works on both numpy 1.x (trapz) and numpy 2.x (trapezoid).
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        del_auc = float(_trapz(del_scores) / steps)
        ins_auc = float(_trapz(ins_scores) / steps)
        return {"deletion_auc": del_auc, "insertion_auc": ins_auc}
