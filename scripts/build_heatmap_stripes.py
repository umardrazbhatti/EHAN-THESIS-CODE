"""
scripts/build_heatmap_stripes.py — Assemble Chapter-4 heatmap figures.

Consumes the per-frame XAI overlay PNGs written by scripts/save_xai_overlays.py
(file pattern  ``{video_id}_{label}_conf{prob}_{method}_f{frame}.png``, with
method in {intrinsic, gradcam, rollout} and label in {real, fake}) and assembles
two kinds of multi-panel figures used in Chapter 4:

  1. Per-(method, label) stripes
       {manip}_{method}_{label}_stripe_NN.png
     A horizontal strip of up to STRIP_LEN frames for a single video, so the
     evolution of the explanation across frames is visible at a glance.

  2. Real-vs-fake comparison panels
       {manip}_intrinsic_real_vs_fake_comparison_NN.png
     A two-row panel: top row REAL frames (green banner), bottom row FAKE frames
     (red banner), using the intrinsic M_t overlays.

Only Pillow + numpy are required (no model inference happens here).

The ``manipulation`` argument is lower-cased and used as the output filename
prefix. NeuralTextures is mapped to ``neuraltexture`` to match the existing
figure-naming convention.
"""

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


STRIP_LEN = 12   # frames per stripe panel (padded/truncated to this width)

_OVERLAY_RE = re.compile(
    r"^(?P<vid>.+)_(?P<label>real|fake)_conf(?P<conf>[0-9.]+)_"
    r"(?P<method>intrinsic|gradcam|rollout)_f(?P<frame>\d+)\.png$"
)


def _manip_prefix(manipulation: str) -> str:
    m = (manipulation or "").strip().lower()
    if not m:
        return "model"
    if m.startswith("neuraltexture"):
        return "neuraltexture"
    return m


def _font(size: int = 18):
    """Best-effort TrueType font with a safe bitmap fallback."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _index_overlays(overlay_dir: Path):
    """
    Group overlay PNGs by (method, label, video_id) -> ordered list of
    (frame_index, path). Returns a nested dict.
    """
    index: dict = {}
    for p in sorted(overlay_dir.glob("*.png")):
        m = _OVERLAY_RE.match(p.name)
        if not m:
            continue
        method = m.group("method")
        label  = m.group("label")
        vid    = m.group("vid")
        frame  = int(m.group("frame"))
        index.setdefault(method, {}).setdefault(label, {}).setdefault(vid, []).append((frame, p))
    for method in index:
        for label in index[method]:
            for vid in index[method][label]:
                index[method][label][vid].sort(key=lambda t: t[0])
    return index


def _hstrip(paths, banner_text, banner_rgb):
    """Build one horizontal strip image (with a coloured title banner)."""
    tiles = [Image.open(p).convert("RGB") for _, p in paths[:STRIP_LEN]]
    if not tiles:
        return None
    tw, th = tiles[0].size
    tiles = [t if t.size == (tw, th) else t.resize((tw, th)) for t in tiles]

    pad      = 6
    banner_h = 30
    n        = len(tiles)
    width    = n * tw + (n + 1) * pad
    height   = banner_h + th + 2 * pad

    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw   = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, banner_h], fill=banner_rgb)
    draw.text((pad, 6), banner_text, fill=(255, 255, 255), font=_font(18))

    x = pad
    y = banner_h + pad
    for t in tiles:
        canvas.paste(t, (x, y))
        x += tw + pad
    return canvas


def _stack_rows(rows):
    """Vertically stack equal-width row images, top-aligned, left-padded to max width."""
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    width  = max(r.width for r in rows)
    height = sum(r.height for r in rows) + 6 * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 6
    return canvas


def build_heatmap_stripes(overlay_dir, out_dir, manipulation: str = "") -> int:
    """
    Build all Chapter-4 stripe and comparison panels from the overlays in
    ``overlay_dir`` and write them to ``out_dir``. Returns the number of figures
    written. Safe to call when no overlays are present (returns 0).
    """
    overlay_dir = Path(overlay_dir)
    out_dir     = Path(out_dir)
    if not overlay_dir.exists():
        print(f"[HeatmapStripes] overlay dir not found: {overlay_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = _manip_prefix(manipulation)
    index  = _index_overlays(overlay_dir)
    if not index:
        print(f"[HeatmapStripes] no overlay PNGs matched in {overlay_dir}")
        return 0

    banner = {"real": (34, 139, 34), "fake": (200, 40, 40)}  # green / red
    written = 0

    # 1. Per-(method, label) stripes
    for method in ("intrinsic", "gradcam", "rollout"):
        for label in ("real", "fake"):
            vids = index.get(method, {}).get(label, {})
            for n, (vid, frames) in enumerate(sorted(vids.items()), start=1):
                title = f"{prefix}  |  {method}  |  {label}  |  {vid}"
                strip = _hstrip(frames, title, banner[label])
                if strip is None:
                    continue
                name = f"{prefix}_{method}_{label}_stripe_{n:02d}.png"
                strip.save(out_dir / name)
                written += 1

    # 2. Real-vs-fake comparison panels (intrinsic only)
    real_vids = sorted(index.get("intrinsic", {}).get("real", {}).items())
    fake_vids = sorted(index.get("intrinsic", {}).get("fake", {}).items())
    for n, ((rvid, rframes), (fvid, fframes)) in enumerate(
        zip(real_vids, fake_vids), start=1
    ):
        row_real = _hstrip(rframes, f"REAL  |  {rvid}", banner["real"])
        row_fake = _hstrip(fframes, f"FAKE  |  {fvid}", banner["fake"])
        panel = _stack_rows([row_real, row_fake])
        if panel is None:
            continue
        name = f"{prefix}_intrinsic_real_vs_fake_comparison_{n:02d}.png"
        panel.save(out_dir / name)
        written += 1

    print(f"[HeatmapStripes] wrote {written} figures -> {out_dir}")
    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Assemble Chapter-4 heatmap stripe figures.")
    ap.add_argument("--overlay_dir", required=True,
                    help="Directory containing per-frame XAI overlay PNGs "
                         "(outputs/plots/heatmaps).")
    ap.add_argument("--out_dir", required=True,
                    help="Destination directory for the assembled stripe figures.")
    ap.add_argument("--manipulation", default="",
                    help="Manipulation name used as the output filename prefix.")
    args = ap.parse_args()
    build_heatmap_stripes(args.overlay_dir, args.out_dir, args.manipulation)
