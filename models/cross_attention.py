"""
models/cross_attention.py — Cross-Attention Fusion.

Returns (M_t, attn_pool):
  M_t      : (B, T, h, w)  intrinsic explanation maps
  attn_pool : (B, d_model)  attention-weighted spatial pooling for classifier gradient path

Three computation paths, selected via config.cross_attention_mode:

  "content_adaptive" (Phase 20 default):
    Each frame generates its OWN query by compressing its spatial feature map through
    a learned MLP (Linear -> GELU -> Linear).  This breaks the fixed-prior collapse
    where a single CLS token produced near-identical M_t for all inputs.
    Implemented by ContentAdaptiveCrossAttention (uses nn.MultiheadAttention internally).

  "cls" (Phase 19 legacy):
    A single [CLS]-token query attends to all spatial positions per frame.
    attn = softmax(CLS·K^T / (sqrt(head_dim) * temperature), dim=-1)
    M_t = mean_heads(attn).reshape(B, T, h, w) * (h*w)
    Temperature is a fixed config scalar (cross_attention_temperature, default 2.0).

  "legacy" (Phase 8 original):
    Q->S attention: all T*L temporal-query positions attend to all L spatial positions,
    then averaged over queries.  Temperature is a learned scalar (log_temp).

CHECKPOINT COMPATIBILITY:
  ContentAdaptiveCrossAttention has different parameter names from CrossAttentionFusion.
  Phase 20 training starts fresh; Phase 19 checkpoints can be loaded if mode="cls".
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Phase 20: Content-Adaptive Cross-Attention ────────────────────────────────

class ContentAdaptiveCrossAttention(nn.Module):
    """
    Per-frame content-adaptive cross-attention (Phase 20 Fix 1).

    Each frame derives its own query by mean-pooling its spatial tokens through
    a small two-layer MLP with GELU activation.  This makes the query input-
    dependent, breaking the fixed-prior collapse of the Phase 19 CLS token.

    Architecture:
      query_projector : Linear(d) -> GELU -> Linear(d//2, d)
      mha             : nn.MultiheadAttention  (standard scaled dot-product)
      norm            : nn.LayerNorm on the attended output
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads

        # Per-frame content-adaptive query projector
        # Input: mean-pooled spatial features (B, T, d_model)
        # Output: content-dependent query (B, T, d_model)
        self.query_projector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Standard multi-head cross-attention
        # Temperature = 1.0 (implicit in standard 1/sqrt(head_dim) scaling)
        # batch_first=True: (batch, seq, embed) layout
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        # Layer norm on attended output (stabilises early training)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, S: torch.Tensor):
        """
        Args:
            S : (B, T, L, d_model)  — spatial CNN feature tokens (L = h*w)
        Returns:
            M_t       : (B, T, h, w)  — explanation maps
            attn_pool : (B, d_model)  — attention-pooled features for classifier
        """
        B, T, L, d = S.shape
        h = w = int(math.sqrt(L))   # assumes square spatial grid

        # Step 1: pool spatial dimension -> per-frame content vector (B, T, d)
        frame_ctx = S.mean(dim=2)

        # Step 2: project to content-adaptive per-frame query (B, T, d)
        Q_adaptive = self.query_projector(frame_ctx)

        # Step 3: cross-attend Q_adaptive to spatial positions of S
        # Process all B*T frames in parallel by merging B and T into the batch dim
        BT = B * T
        Q_bt = Q_adaptive.reshape(BT, 1, d)   # (B*T, 1, d) — 1 query per frame
        S_bt = S.reshape(BT, L, d)             # (B*T, L, d) — L spatial keys/values

        # nn.MultiheadAttention(batch_first=True):
        #   query  (B*T, 1, d), key (B*T, L, d), value (B*T, L, d)
        #   attn_output  : (B*T, 1, d)
        #   attn_weights : (B*T, 1, L)  — averaged over heads
        attn_output, attn_weights = self.mha(
            Q_bt, S_bt, S_bt,
            need_weights=True,
        )

        # Step 4: reshape attention weights -> (B, T, h, w)
        # Scale by L so values express deviation from uniform (1/L * L = 1 at uniform)
        M_t = attn_weights.squeeze(1).reshape(B, T, h, w) * L

        # Step 5: apply layer norm to attended output
        attn_out_norm = self.norm(
            attn_output.squeeze(1).reshape(B, T, d)
        )   # (B, T, d)

        # Step 6: pool over T frames -> (B, d)  for classifier gradient path
        attn_pool = attn_out_norm.mean(dim=1)

        return M_t, attn_pool


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8,
                 attn_temp_init: float = 0.0,
                 use_cls_query: bool = True,
                 cross_attention_temperature: float = 2.0):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = math.sqrt(self.head_dim)

        # CLS path: fixed temperature scalar from config
        self.use_cls_query = use_cls_query
        self.temperature   = float(cross_attention_temperature)

        # Legacy path: learnable temperature τ = exp(log_temp)
        # Kept so old checkpoints load cleanly when use_cls_query=False.
        self.log_temp = nn.Parameter(torch.tensor(float(attn_temp_init)))

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def forward(self, Q_or_cls, S):
        """
        Args:
            Q_or_cls : (B, d_model)       if use_cls_query=True  — CLS token embedding
                     : (B, T, L, d_model) if use_cls_query=False — temporal queries
            S        : (B, T, L, d_model) spatial tokens
        Returns:
            M_t      : (B, T, h, w)
            attn_pool: (B, d_model)
        """
        if self.use_cls_query:
            return self._cls_forward(Q_or_cls, S)
        else:
            return self._legacy_forward(Q_or_cls, S)

    # ── CLS path ──────────────────────────────────────────────────────────────

    def _cls_forward(
        self,
        cls_query: torch.Tensor,   # (B, d_model)
        S:         torch.Tensor,   # (B, T, L, d_model)
    ):
        """
        Single CLS token attends to all spatial positions per frame.

        1. Project CLS via q_proj → (B, d_model); broadcast across T.
        2. Project S via k_proj/v_proj.
        3. Multi-head scaled dot-product: Q(1)·K(L)^T / (√head_dim · temperature).
        4. Softmax over L, average across heads → attn (B, T, L).
        5. M_t = attn.reshape(B,T,h,w) * (h*w)  — no min-max normalisation.
        6. attn_pool = weighted sum of V → mean over T → (B, d_model).
        """
        B, T, L, d = S.shape
        h = w = int(math.sqrt(L))   # assumes square spatial grid (7×7 for 224px)
        H  = self.num_heads
        hd = self.head_dim

        # ── Project CLS query ────────────────────────────────────────────────
        # cls_query: (B, d) → q_proj → (B, d) → expand to (B*T, 1, d)
        Qp = self.q_proj(cls_query)                          # (B, d)
        Qp = Qp.unsqueeze(1).expand(B, T, d).reshape(B * T, 1, d)  # (B*T, 1, d)
        # Split into multi-head: (B*T, H, 1, hd)
        Qp_h = Qp.view(B * T, 1, H, hd).transpose(1, 2)    # (B*T, H, 1, hd)

        # ── Project spatial keys / values ────────────────────────────────────
        S_flat = S.reshape(B * T, L, d)
        Kp     = self.k_proj(S_flat)                         # (B*T, L, d)
        Vp     = self.v_proj(S_flat)                         # (B*T, L, d)
        # Split into multi-head: (B*T, H, L, hd)
        Kp_h = Kp.view(B * T, L, H, hd).transpose(1, 2)    # (B*T, H, L, hd)
        Vp_h = Vp.view(B * T, L, H, hd).transpose(1, 2)    # (B*T, H, L, hd)

        # ── Scaled dot-product attention ─────────────────────────────────────
        # scores: (B*T, H, 1, hd) @ (B*T, H, hd, L) → (B*T, H, 1, L)
        scores = torch.matmul(Qp_h, Kp_h.transpose(-2, -1)) / (math.sqrt(hd) * self.temperature)
        attn   = F.softmax(scores, dim=-1)                   # (B*T, H, 1, L)

        # ── Mean across heads → explanation map ─────────────────────────────
        attn_mean = attn.mean(dim=1).squeeze(1)              # (B*T, L)
        # Reshape and scale by h*w so values express deviation from uniform (1/L)
        M_t = attn_mean.reshape(B, T, h, w) * (h * w)       # (B, T, h, w)

        # ── Attention-pooled features for classifier gradient path ───────────
        # attn: (B*T, H, 1, L) → mean across heads: (B*T, 1, L)
        # V: (B*T, H, L, hd) → compute weighted sum per head then cat
        attn_for_pool = attn.mean(dim=1)                     # (B*T, 1, L)
        S_pool        = torch.bmm(attn_for_pool, Vp)         # (B*T, 1, d)
        S_pool        = S_pool.squeeze(1)                    # (B*T, d)
        attn_pool     = S_pool.reshape(B, T, d).mean(dim=1)  # (B, d)

        return M_t, attn_pool

    # ── Legacy path (safety net) ──────────────────────────────────────────────

    def _legacy_forward(
        self,
        Q: torch.Tensor,   # (B, T, L, d_model)  temporal queries
        S: torch.Tensor,   # (B, T, L, d_model)  spatial keys/values
    ):
        """
        Original Phase-8 implementation: all temporal-query positions attend to
        spatial positions; column-mean collapses query dimension.
        Temperature is a learned scalar (log_temp), clamped to [0.5, 10.0].
        """
        B, T, L, d = Q.shape
        h = w = int(math.sqrt(L))

        Q_flat = Q.reshape(B * T, L, d)
        S_flat = S.reshape(B * T, L, d)

        Qp = self.q_proj(Q_flat)    # (B·T, L, d)
        Kp = self.k_proj(S_flat)
        Vp = self.v_proj(S_flat)

        # Temperature-scaled attention
        tau    = torch.exp(self.log_temp).clamp(min=0.5, max=10.0)
        scores = torch.bmm(Qp, Kp.transpose(-2, -1)) / (self.scale * tau)  # (B·T, L, L)
        A      = F.softmax(scores, dim=-1)

        # Column-mean → already a probability distribution over L spatial positions
        M_flat = A.mean(dim=-2)                        # (B·T, L)
        M_t    = M_flat.reshape(B, T, h, w)

        # Attention-pooled features (classifier gradient path)
        W         = M_flat.unsqueeze(-1)               # (B·T, L, 1)
        S_pool    = (W * Vp).sum(dim=1)                # (B·T, d)
        attn_pool = S_pool.reshape(B, T, d).mean(dim=1)  # (B, d)

        return M_t, attn_pool
