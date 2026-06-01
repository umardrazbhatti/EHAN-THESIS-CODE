"""
scripts/train_synthetic.py — Phase 1: CPU-only smoke test using synthetic data.
Runs 2 epochs to verify that the full pipeline is wired correctly before
committing GPU time on Kaggle.
"""

import os
from config import EAHNConfig
from scripts.train_real import main as train_main


def main():
    config = EAHNConfig(
        dataset_name="synthetic",
        epochs=2,
        batch_size=2,
        num_frames=4,
        frame_size=224,
        backbone="efficientnet_b0",   # B0 for fast CPU smoke test
        transformer_layers=2,
        transformer_heads=2,
        d_model=64,
        mixed_precision=False,
        num_workers=0,
        output_dir="outputs_synthetic/",
        eval_after_train=True,
        heatmap_samples=4,
        device="cpu",
        # Phase 20 Final: exercise all 5 losses from epoch 1 (no warmup for smoke test)
        explanation_warmup_epochs=0,
        explanation_ramp_epochs=1,
        cross_attention_mode="content_adaptive",
        use_faithfulness_loss=True,
        lambda_div=0.5,
        lambda_faith=0.3,
        lambda1=2.0,
        lambda1_weak=0.1,   # weak-supervision weight (used when batch has no masks)
    )
    os.makedirs(config.output_dir, exist_ok=True)
    train_main(config)


if __name__ == "__main__":
    main()
