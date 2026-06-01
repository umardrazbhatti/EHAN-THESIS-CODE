# EHAN — Explanation-Aware Hybrid Attention Network

A hybrid CNN–Transformer for **deepfake detection** that produces, in a single
forward pass, both a classification score and an **intrinsic explanation map**
`M_t` highlighting the facial regions that drove the decision. The model is
trained per-manipulation on the **FaceForensics++ (c23)** dataset as five
specialists: `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, and
`NeuralTextures`.

This repository contains the model, training/evaluation pipeline, and the
explanation tooling used to produce the results and figures in the thesis.

## Architecture

```
frames (B, T, 3, 224, 224)
        │
        ▼
  EfficientNet-B4 spatial stream      ──►  per-frame spatial features
        │
        ▼
  Transformer temporal stream         ──►  temporally-contextualised tokens
        │
        ▼
  content-adaptive cross-attention    ──►  intrinsic explanation maps  M_t
        │                                  + attention-pooled descriptor
        ▼
  classifier head                     ──►  real / fake logit + probability
```

The training objective combines a classification loss with three explanation
regularisers — an explanation (weak-prior) loss, a diversity loss, and a
**gradient-faithfulness loss** — plus a temporal-consistency loss. The
explanation losses are zeroed for a short warmup and then ramped to full
strength, which stabilises the explanation head without hurting detection.

## Repository layout

```
config.py                     single source of truth for all hyperparameters / CLI flags
run_full_pipeline.py          entry point: train (+ optional eval) + dashboard

data/
  datasets.py                 FF++ specialist DeepfakeDataset (stratified, balanced)
  collate.py                  batch collation
  transforms.py               train/val augmentation pipelines
  face_align.py               MTCNN-based face alignment + caching
  synthetic_generator.py      lightweight synthetic data (smoke tests)

models/
  eahn.py                     top-level EAHN model
  spatial_stream.py           EfficientNet-B4 spatial backbone
  temporal_stream.py          Transformer temporal encoder
  cross_attention.py          content-adaptive cross-attention -> M_t

losses/
  classification.py           BCE / focal loss with label smoothing
  explanation.py              explanation, diversity, and faithfulness losses
  temporal.py                 temporal-consistency loss

metrics/
  detection.py                AUC-ROC, AUC-PR, F1, balanced accuracy, thresholds
  explanation.py              temporal SSIM + deletion/insertion AUC

xai/
  gradcam.py                  Grad-CAM (post-hoc baseline)
  attention_rollout.py        attention-rollout (post-hoc baseline)
  overlay.py                  heatmap-on-frame overlay helpers

scripts/
  train_real.py               FF++ training loop (warmup/ramp loss schedule)
  train_synthetic.py          synthetic smoke-test training
  evaluate.py                 detection + explanation eval, plots, heatmaps
  run_explanation_suite.py    intrinsic explanation metrics -> explanation_metrics.json
  save_xai_overlays.py        per-frame intrinsic / Grad-CAM / rollout overlays
  build_heatmap_stripes.py    assembles Chapter-4 stripe + comparison figures
  summary_chart.py            headline summary chart
  dashboard.py                end-of-run console dashboard
  verify_dataset.py           dataset sanity checks

utils/
  checkpointing.py            save/load checkpoints (classifier-surgery aware)
  logging_utils.py            logging helpers
  visualization.py            annotated frame strips, explanation videos
```

## Running on Kaggle

The intended workflow uses the companion notebook
(`EHAN_specialist_training.ipynb`, distributed separately) on a Kaggle Tesla T4.
Fork it five times and set `ACTIVE_MANIPULATION` to a different manipulation in
each fork. Each fork:

1. verifies the FF++ dataset layout and the balanced specialist split,
2. clones this repository,
3. trains for 20 epochs (effective batch size 16 via gradient accumulation),
4. evaluates on the FF++ test split, and
5. emits three result zips (`chapter4`, `essentials`, `everything`).

## Running locally

```bash
pip install -r requirements.txt

python run_full_pipeline.py \
    --data_root /path/to/ffpp_data \
    --dataset_name ff++ \
    --active_manipulation Deepfakes \
    --output_dir outputs \
    --cache_dir .face_cache \
    --epochs 20 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_per_class 1000 \
    --cross_attention_mode content_adaptive \
    --explanation_warmup_epochs 5 \
    --explanation_ramp_epochs 5 \
    --lambda1 2.0 --lambda1_weak 0.1 --lambda2 1.0 \
    --lambda_div 0.5 --lambda_faith 0.3 \
    --use_faithfulness_loss \
    --use_amp --grad_checkpoint \
    --eval_after_train
```

## Outputs

Written under `OUTPUT_DIR/`:

- `best_model.pth` — best checkpoint by validation balanced accuracy.
- `eval/ffpp_test_metrics.json`, `metrics.csv`, `eval/report.txt` — detection metrics.
- `explanation_metrics.json` — intrinsic explanation metrics (temporal SSIM,
  deletion AUC, insertion AUC).
- `plots/ffpp_*.png` — ROC, precision–recall, confusion matrices, score distribution.
- `plots/heatmaps/` — per-frame intrinsic / Grad-CAM / attention-rollout overlays.
- `plots/stripes/` — assembled heatmap stripes and real-vs-fake comparison panels.
- `heatmaps/` — annotated explanation videos and frame strips.

## Dataset

FaceForensics++ (c23) in a custom layout:

```
ffpp_data/
  original_sequences/youtube/c23/videos/*.mp4
  manipulated_sequences/<Manipulation>/c23/videos/*.mp4
```

## Requirements

Python 3.10+ and PyTorch 2.1+. See `requirements.txt`. Developed and tested on a
Kaggle Tesla T4 (CUDA).
