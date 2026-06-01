"""
models/eahn.py — Explanation-Aware Hybrid Network (EAHN).

Assembles SpatialStream → TemporalStream → CrossAttentionFusion → classifier.
Single forward pass produces:
  - logit / prob  : classification output
  - M_t           : intrinsic explanation maps  (B, T, h, w) at feature resolution
  - M_t_up        : upsampled explanation maps  (B, T, H, W) for visualisation
  - S             : spatial tokens              (B, T, N, d_model)
  - low_level     : low-level features          (B, T, C_low, Hl, Wl)  for gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from config import EAHNConfig
from models.spatial_stream import SpatialStream
from models.temporal_stream import TemporalStream
from models.cross_attention import CrossAttentionFusion, ContentAdaptiveCrossAttention


@dataclass
class EAHNOutput:
    logit:     torch.Tensor   # (B,)
    prob:      torch.Tensor   # (B,)
    M_t:       torch.Tensor   # (B, T, h, w)
    M_t_up:    torch.Tensor   # (B, T, H, W)
    S:         torch.Tensor   # (B, T, N, d_model)
    low_level: torch.Tensor   # (B, T, C_low, Hl, Wl)
    attn_pool: torch.Tensor   # (B, d_model) — attention-weighted pooling for grad path


class EAHN(nn.Module):
    def __init__(self, config: EAHNConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # ── Spatial Stream ────────────────────────────────────────────────────
        self.spatial_stream = SpatialStream(
            backbone_name=config.backbone,
            pretrained=config.backbone_pretrained,
            d_model=d,
            freeze_backbone=False,
        )

        # Infer N = h*w from a dummy forward pass
        dummy = torch.zeros(1, 3, config.frame_size, config.frame_size)
        with torch.no_grad():
            dummy_tokens = self.spatial_stream(dummy)
        N = dummy_tokens.shape[1]
        self.N      = N
        self.feat_h = self.spatial_stream.feat_h
        self.feat_w = self.spatial_stream.feat_w

        # ── Temporal Stream ───────────────────────────────────────────────────
        # max_seq_len = T*N + 1 (CLS token)
        max_seq = config.num_frames * N + 1
        self.temporal_stream = TemporalStream(
            d_model=d,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            dropout=config.dropout,
            max_seq_len=max_seq,
        )

        # ── Cross-Attention Fusion ────────────────────────────────────────────
        # Phase 20: dispatch on cross_attention_mode
        #   "content_adaptive" -> ContentAdaptiveCrossAttention  (default)
        #   "cls"              -> CrossAttentionFusion CLS path  (Phase 19)
        #   "legacy"           -> CrossAttentionFusion Q->S path (Phase 8)
        _ca_mode = getattr(config, "cross_attention_mode", "content_adaptive")
        self._ca_mode = _ca_mode  # stash for forward dispatch

        if _ca_mode == "content_adaptive":
            self.cross_attention = ContentAdaptiveCrossAttention(
                d_model=d,
                num_heads=config.transformer_heads,
            )
        else:
            # Legacy paths: CLS or Q->S
            _use_cls = (_ca_mode != "legacy") and getattr(
                config, "cross_attention_use_cls_query", True
            )
            self.cross_attention = CrossAttentionFusion(
                d_model=d,
                num_heads=config.transformer_heads,
                attn_temp_init=getattr(config, "attn_temp_init", 0.0),
                use_cls_query=_use_cls,
                cross_attention_temperature=getattr(
                    config, "cross_attention_temperature", 2.0
                ),
            )

        # ── Classification Head ───────────────────────────────────────────────
        # Surgery: 2*d input so the M_t-weighted spatial pool (S_temporal, second
        # half) can enter the classifier alongside the existing final_feat (first half).
        self.classifier = nn.Linear(2 * d, 1)

        # _init_weights runs Xavier-uniform on the full [1, 2*d] weight matrix.
        # The zero block MUST run after so it is not overwritten by Xavier init.
        self._init_weights()

        # Zero the second half: pre-surgery behavior is preserved at init because
        # combined[:, d:] == S_temporal and weight[:, d:] == 0 → no contribution.
        with torch.no_grad():
            self.classifier.weight[:, d:].zero_()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def enable_gradient_checkpointing(self):
        if hasattr(self.temporal_stream, "enable_gradient_checkpointing"):
            self.temporal_stream.enable_gradient_checkpointing()
        if hasattr(self.spatial_stream, "set_grad_checkpointing"):
            self.spatial_stream.set_grad_checkpointing(True)   # timm backbone support

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, frames: torch.Tensor) -> EAHNOutput:
        """
        Args:
            frames : (B, T, 3, H, W)
        Returns:
            EAHNOutput
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        # Spatial stream — processes all B*T frames in parallel
        spatial_tokens = self.spatial_stream(frames_flat)   # (B*T, N, d)
        low_feat = self.spatial_stream.low_level_features() # (B*T, C_low, Hl, Wl)

        N = spatial_tokens.shape[1]
        d = self.config.d_model
        C_low, Hl, Wl = low_feat.shape[1], low_feat.shape[2], low_feat.shape[3]

        spatial_tokens = spatial_tokens.view(B, T, N, d)
        low_level      = low_feat.view(B, T, C_low, Hl, Wl)

        # Phase 20 Final Fix 5: ensure S is in the gradient graph for FaithfulnessLoss.
        # spatial_tokens is a non-leaf tensor from SpatialStream, but its .requires_grad
        # attribute may be False even though grad_fn is set (PyTorch doesn't always
        # propagate requires_grad=True to non-leaf outputs from module.forward()).
        # Calling requires_grad_(True) here makes the FaithfulnessLoss guard pass and
        # allows autograd.grad(logit, S) to compute a real gradient.
        # Only active during training when faithfulness loss is enabled.
        if getattr(self.config, "use_faithfulness_loss", False) and self.training:
            spatial_tokens = spatial_tokens.requires_grad_(True)

        # Temporal stream — flatten T*N spatial tokens as the sequence
        Q, cls_out = self.temporal_stream(
            spatial_tokens.reshape(B, T * N, d)
        )                                                    # Q: (B, T*N, d)

        Q = Q.reshape(B, T, N, d)

        # Cross-attention fusion → explanation maps + attention-pooled features
        # Phase 20: dispatch on _ca_mode set during __init__
        # For faithfulness loss, S needs to be in the computation graph.
        # Since spatial_tokens is computed from params (requires_grad), autograd
        # can compute grad(logit, spatial_tokens) without explicit requires_grad_(True).
        if self._ca_mode == "content_adaptive":
            M_t, attn_pool = self.cross_attention(spatial_tokens)           # content-adaptive
        elif self._ca_mode == "cls":
            M_t, attn_pool = self.cross_attention(cls_out, spatial_tokens)  # CLS path
        else:
            M_t, attn_pool = self.cross_attention(Q, spatial_tokens)        # legacy Q->S

        # Upsample explanation maps to input resolution for visualisation / loss
        M_t_up = F.interpolate(
            M_t.reshape(B * T, 1, self.feat_h, self.feat_w),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, T, H, W)                               # (B, T, H, W)

        # Stochastic CLS_out dropout: during training, randomly force classification
        # through the attention branch only, ensuring gradient pressure flows to M_t.
        if self.training and torch.rand(1).item() < self.config.cls_dropout_p:
            final_feat = attn_pool
        else:
            final_feat = cls_out + attn_pool                # (B, d)

        # ── M_t-weighted spatial pool of S (classifier-coupling surgery) ────────
        # Adds an explicit path: L_cls gradient flows through M_t attention weights
        # directly, not only through V in attn_pool. Zero-init on weight[:, d:]
        # makes this a no-op at start; the branch activates as M_t learns.
        # M_t : (B, T, h, w)  S : (B, T, L, d)  where L = h*w
        B_f, T_f, h_f, w_f = M_t.shape
        L_f = h_f * w_f
        M_t_flat    = M_t.reshape(B_f, T_f, L_f)
        # Sum-normalise to a proper spatial probability distribution.
        # M_t = attn_weights * L (already non-negative); do NOT softmax again.
        M_t_weights = M_t_flat / (M_t_flat.sum(dim=-1, keepdim=True) + 1e-8)
        M_t_weights = M_t_weights.unsqueeze(-1)              # (B, T, L, 1)
        S_pooled    = (spatial_tokens * M_t_weights).sum(dim=2)  # (B, T, d)
        S_temporal  = S_pooled.mean(dim=1)                   # (B, d)
        # ────────────────────────────────────────────────────────────────────────

        combined = torch.cat([final_feat, S_temporal], dim=-1)   # (B, 2*d)
        logit = self.classifier(combined).squeeze(-1)             # (B,)
        prob  = torch.sigmoid(logit)

        return EAHNOutput(
            logit=logit, prob=prob,
            M_t=M_t, M_t_up=M_t_up,
            S=spatial_tokens, low_level=low_level,
            attn_pool=attn_pool,
        )
