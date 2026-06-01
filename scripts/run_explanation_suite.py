"""
scripts/run_explanation_suite.py — Orchestrator for intrinsic explanation-quality metrics.

Computes the two intrinsic explanation-quality measures reported in Chapter 4
on the trained model and the test loader, then writes a unified JSON to
``output_path``:

  - temporal_ssim          : temporal stability of the explanation maps M_t
  - deletion_insertion_auc : causal faithfulness of the saliency ordering

Output schema (consumed by the reporting/plotting code):

    {
      "active_manipulation": "<manipulation name>",
      "intrinsic": {
        "deletion_auc":  <float>,
        "insertion_auc": <float>,
        "temporal_ssim": <float>
      }
    }
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from metrics.explanation import ExplanationMetrics


def run_explanation_suite(model, test_loader, config, output_path: Path) -> dict:
    """
    Run the intrinsic explanation metrics on the trained model + test loader.
    Save the unified JSON to output_path, print a short summary, and return the
    metrics dict.

    Args:
        model       : trained EAHN model (eval mode will be set internally)
        test_loader : DataLoader for the test set (no shuffle)
        config      : EAHNConfig
        output_path : Path where explanation_metrics.json will be written
    """
    device = torch.device(config.device)
    model.eval()

    print("\n[ExplanationSuite] Collecting M_t across test set...")

    # ── 1. Collect all M_t + frames ─────────────────────────────────────────
    all_M_t_up = []
    all_frames = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Suite pass", leave=False):
            frames = batch["frames"].to(device)
            out    = model(frames)
            all_M_t_up.append(out.M_t_up.cpu())
            all_frames.append(frames.cpu())

    all_M_t_up = torch.cat(all_M_t_up, dim=0)   # (N, T, H, W)
    all_frames = torch.cat(all_frames, dim=0)   # (N, T, C, H, W)
    N = len(all_M_t_up)

    subset_size = min(getattr(config, "heatmap_samples", 20), N)
    rng         = np.random.default_rng(42)
    indices     = rng.choice(N, subset_size, replace=False)

    # ── 2. Temporal SSIM ────────────────────────────────────────────────────
    print("[ExplanationSuite] Computing temporal SSIM...")
    ssim_val = ExplanationMetrics.temporal_ssim(all_M_t_up[indices])

    # ── 3. Deletion / Insertion AUC ─────────────────────────────────────────
    print("[ExplanationSuite] Computing deletion/insertion AUC...")
    del_ins = {"deletion_auc": 0.0, "insertion_auc": 0.0}
    try:
        sample_idx    = int(indices[0])
        frames_sample = all_frames[sample_idx:sample_idx + 1]
        sal_sample    = all_M_t_up[sample_idx:sample_idx + 1].numpy()
        del_ins = ExplanationMetrics.deletion_insertion_auc(
            model, frames_sample, sal_sample, steps=10
        )
    except Exception as e:
        import traceback
        print("\n" + "!" * 70)
        print("[ExplanationSuite] WARNING: deletion/insertion AUC FAILED and was "
              "left at 0.0/0.0.")
        print(f"  Reason: {type(e).__name__}: {e}")
        print("  These zeros are NOT a real result. Fix the cause before trusting "
              "explanation_metrics.json.")
        traceback.print_exc()
        print("!" * 70 + "\n")

    # ── Assemble result ─────────────────────────────────────────────────────
    result = {
        "active_manipulation": getattr(config, "active_manipulation", ""),
        "intrinsic": {
            "deletion_auc":  float(del_ins.get("deletion_auc", 0.0)),
            "insertion_auc": float(del_ins.get("insertion_auc", 0.0)),
            "temporal_ssim": float(ssim_val),
        },
    }

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n[ExplanationSuite] === Summary ===")
    print(f"  Temporal SSIM  : {result['intrinsic']['temporal_ssim']:.3f}")
    print(f"  Deletion AUC   : {result['intrinsic']['deletion_auc']:.3f}")
    print(f"  Insertion AUC  : {result['intrinsic']['insertion_auc']:.3f}")

    # ── Save JSON ───────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ExplanationSuite] metrics saved -> {output_path}")

    return result
