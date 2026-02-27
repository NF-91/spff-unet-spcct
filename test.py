#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  1) Bland–Altman GROUP plots: HA + Iodine (slice-wise; subplots per model)
  2) Per-class heatmaps (mean±std across seeds) from test_details.csv
  3) Qualitative overlays from best checkpoints

Outputs:
  analysis_plots/
    ├─ bland_altman/
    │   ├─ group_HA.png (+pdf)
    │   └─ group_Iodine.png (+pdf)
    ├─ per_class_heatmaps/
    │   ├─ heatmap_dice.png
    │   ├─ heatmap_sensitivity.png
    │   ├─ heatmap_specificity.png
    │   ├─ heatmap_precision.png
    │   └─ heatmap_iou.png
    └─ qual_overlays/  (if DO_QUAL_VIZ=1)
"""

from __future__ import annotations

import os
import re
import sys
import inspect
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

# seaborn optional
try:
    import seaborn as sns
except Exception:
    sns = None

# ─────────────────────────────────────────────────────────────────────────────
# Robust package import if running inside innovative3D/
# ─────────────────────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
if _THIS.parent.name == "innovative3D" and str(_THIS.parent.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Project config
# ─────────────────────────────────────────────────────────────────────────────
from innovative3D.config import (
    VARIANTS,
    SEEDS,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_FRAMES,
    NUM_CLASSES,
    trainval_sets,
    global_label_names,
    label_colors,
)

CKPT_DIR = Path(CHECKPOINT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
HEATMAP_CMAP = mpl.colormaps["YlOrRd"].copy()

DO_QUAL_VIZ = os.getenv("DO_QUAL_VIZ", "1") == "1"
QUAL_VIZ_INDICES = [0, 1, 2, 3, 4, 5]
QUAL_VIZ_MAX_ITEMS_PER_BATCH = 2
QUAL_VIZ_OUTROOT = "analysis_plots/qual_overlays"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
def _logs_path(model: str, seed: int) -> Path:
    return CKPT_DIR / model / f"seed{seed}" / "logs"

def _test_details_csv(model: str, seed: int) -> Optional[Path]:
    p = CKPT_DIR / model / f"seed{seed}" / "test_details.csv"
    return p if p.exists() else None

def _find_best_or_last_ckpt(model: str, seed: int) -> Optional[Path]:
    root = CKPT_DIR / model / f"seed{seed}"
    bests = sorted(root.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if bests:
        return bests[0]
    p = root / "last.ckpt"
    return p if p.exists() else None

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _save(fig: plt.Figure, path: Path, also_pdf: bool = False):
    _ensure_dir(path)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    pdf_path = None
    if also_pdf:
        pdf_path = Path(str(path).rsplit(".", 1)[0] + ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    if pdf_path:
        print(f"[PLOT] saved {path} and {pdf_path}")
    else:
        print(f"[PLOT] saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Labels + HA/Iodine groups (inferred, no hard-coded indices)
# ─────────────────────────────────────────────────────────────────────────────
def _label_names() -> List[str]:
    names = []
    for i in range(NUM_CLASSES):
        if i < len(global_label_names):
            names.append(str(global_label_names[i]))
        else:
            names.append(str(i))
    if names:
        names[0] = "BG"
    return names

def _canon(s: str) -> str:
    s = str(s).strip()
    s = s.replace("CT water", "Water").replace("water", "Water")
    s = s.replace("background", "BG").replace("Background", "BG")
    return s

LABEL_NAMES = [_canon(x) for x in _label_names()]

def infer_group_classes(names: List[str]) -> Tuple[List[int], List[int]]:
    ha, iod = [], []
    for i, n in enumerate(names):
        if i == 0:
            continue
        u = _canon(n).upper()
        if re.match(r"^HA\d+", u):
            ha.append(i)
        elif re.match(r"^I\d+", u):
            iod.append(i)
    return ha, iod

HA_CLASSES, IODINE_CLASSES = infer_group_classes(LABEL_NAMES)

def baseline_model_name(model_names: List[str]) -> str:
    for cand in ("S3UNet", "S3UNet_Backbone"):
        for m in model_names:
            if cand == m or cand in m:
                return m
    return model_names[0]

# ─────────────────────────────────────────────────────────────────────────────
# Load details + sanitize absent GT artifacts
# ─────────────────────────────────────────────────────────────────────────────
def load_test_details(model: str, seed: int) -> Optional[pd.DataFrame]:
    f = _test_details_csv(model, seed)
    if f is None:
        return None
    try:
        df = pd.read_csv(f)
    except Exception:
        return None
    return df if not df.empty else None

def _clean_absent_gt_artifacts(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Some older runs produce Dice=Sens=Spec=1 for absent GT classes.
    Mark those as NaN so they don't inflate means.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in ("dice", "sensitivity", "specificity", "precision", "iou"):
        if col not in df.columns:
            df[col] = np.nan

    d  = pd.to_numeric(df["dice"], errors="coerce")
    se = pd.to_numeric(df["sensitivity"], errors="coerce")
    sp = pd.to_numeric(df["specificity"], errors="coerce")
    mask = (d == 1.0) & (se == 1.0) & (sp == 1.0)

    df.loc[mask, ["dice", "sensitivity", "specificity", "precision", "iou"]] = np.nan
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 1) Per-class heatmaps (single folder)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_per_class_from_details(
    model_names: List[str],
    metric_key: str,  # dice|sensitivity|specificity|precision|iou
) -> Tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for model in model_names:
        per_class_seed_means: List[List[float]] = [[] for _ in range(NUM_CLASSES)]

        for seed in SEEDS:
            df = load_test_details(model, seed)
            if df is None:
                continue
            df = _clean_absent_gt_artifacts(df)
            if df is None or df.empty:
                continue
            if {"class", metric_key} - set(df.columns):
                continue

            d = df.copy()
            d["class"] = pd.to_numeric(d["class"], errors="coerce")
            d[metric_key] = pd.to_numeric(d[metric_key], errors="coerce")
            d = d.dropna(subset=["class", metric_key])
            if d.empty:
                continue
            d["class"] = d["class"].astype(int)

            g = d.groupby("class")[metric_key].mean()
            for c in range(NUM_CLASSES):
                if c in g.index and np.isfinite(g.loc[c]):
                    per_class_seed_means[c].append(float(g.loc[c]))

        m = [float(np.mean(v)) if v else np.nan for v in per_class_seed_means]
        s = [float(np.std(v))  if v else np.nan for v in per_class_seed_means]
        means.append(m); stds.append(s)

    return np.array(means, dtype=float), np.array(stds, dtype=float)

def _reorder_columns_by_name(
    means: np.ndarray,
    stds: np.ndarray,
    col_labels: List[str],
    desired_fg: List[str],
    keep_first_label: str = "BG",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    means = np.atleast_2d(means)
    stds  = np.atleast_2d(stds)
    name2idx = {_canon(n): i for i, n in enumerate(col_labels)}
    perm: List[int] = []

    if keep_first_label in name2idx:
        perm.append(name2idx[keep_first_label])

    for n in desired_fg:
        nn = _canon(n)
        if nn in name2idx and name2idx[nn] not in perm:
            perm.append(name2idx[nn])

    perm += [i for i in range(len(col_labels)) if i not in perm]
    return means[:, perm], stds[:, perm], [col_labels[i] for i in perm]

def plot_heatmap_mean_std(
    means: np.ndarray,
    stds: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    out_path: Path,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    means = np.asarray(means, dtype=float)
    stds  = np.asarray(stds, dtype=float)

    annot = np.empty(means.shape, dtype=object)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            m = means[i, j]
            s = stds[i, j]
            annot[i, j] = "NaN" if not np.isfinite(m) else (f"{m:.2f}\n±{s:.2f}" if np.isfinite(s) else f"{m:.2f}")

    fig_w = max(10.0, 0.55 * len(col_labels) + 2.0)
    fig_h = max(4.0,  0.45 * len(row_labels) + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if sns is not None:
        sns.heatmap(
            means,
            cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax,
            xticklabels=col_labels, yticklabels=row_labels,
            linewidths=0.5, linecolor="white",
            annot=annot, fmt="",
            cbar_kws={"label": "mean (± std)"},
            annot_kws={"fontsize": 9},
            ax=ax,
        )
    else:
        im = ax.imshow(means, vmin=vmin, vmax=vmax, cmap=HEATMAP_CMAP, aspect="auto")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for (i, j), t in np.ndenumerate(annot):
            if t:
                ax.text(j, i, t, ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("mean (± std)")

    ax.set_xlabel("Class")
    ax.set_ylabel("Model")
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, out_path)

def run_per_class_heatmaps(model_names: List[str], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preferred class order (by name) — still keeps ALL columns
    desired_fg = [
        "HA800","HA400","HA200","HA100","HA50",
        "I15","I10","I5",
        "Adipose","Liver","Lung","Water"
    ]

    col_labels = LABEL_NAMES[:]  # 0..C-1, BG at 0

    for key, nice in [
        ("dice", "Dice"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("iou", "IoU"),
    ]:
        print(f"[HEAT] {nice}")
        means, stds = aggregate_per_class_from_details(model_names, key)

        # reorder columns ONLY (no dropping; keeps ALL classes)
        means, stds, cols = _reorder_columns_by_name(
            means, stds, col_labels, desired_fg, keep_first_label="BG"
        )

        plot_heatmap_mean_std(
            means, stds,
            row_labels=model_names,
            col_labels=cols,
            out_path=out_dir / f"heatmap_{key}.png",
            title=f"Per-class {nice} (mean ± std across seeds)",
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2) Bland–Altman GROUP plots (slice-wise, subplots per model)
# ─────────────────────────────────────────────────────────────────────────────
_SLICE_CANDIDATES = ["slice", "z", "frame", "index", "idx"]

def _ensure_slice_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    for cand in _SLICE_CANDIDATES:
        if cand in df.columns:
            return df, cand
    if "case" in df.columns:
        df = df.sort_values(["case"]).copy()
        df["__slice"] = df.groupby("case").cumcount()
        return df, "__slice"
    df["__slice"] = np.arange(len(df))
    return df, "__slice"

def _build_global_slice_index(model_names: List[str]) -> Dict[Tuple, int]:
    pairs = set()
    for model in model_names:
        for seed in SEEDS:
            df = load_test_details(model, seed)
            if df is None or df.empty or "case" not in df.columns:
                continue
            df, sname = _ensure_slice_column(df)
            sub = df[["case", sname]].dropna()
            for _, r in sub.iterrows():
                pairs.add((r["case"], int(r[sname])))
    pairs_sorted = sorted(pairs, key=lambda t: (t[0], t[1]))
    return {pair: i + 1 for i, pair in enumerate(pairs_sorted)}

def _grid_rc(n: int) -> Tuple[int, int]:
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, (n + 1) // 2
    cols = 4
    rows = int(np.ceil(n / cols))
    return rows, cols

def _per_model_slice_errors_for_group(model: str, group_classes: List[int], index_map: Dict[Tuple, int]) -> Tuple[np.ndarray, np.ndarray]:
    agg: Dict[Tuple, List[float]] = {}
    for seed in SEEDS:
        df = load_test_details(model, seed)
        if df is None or df.empty:
            continue
        if {"case", "class", "dice"} - set(df.columns):
            continue

        df = _clean_absent_gt_artifacts(df)
        df = df[df["class"].isin(group_classes)].copy()
        if df.empty:
            continue

        df, sname = _ensure_slice_column(df)
        # mean Dice across group classes for each (case, slice)
        df = df.groupby(["case", sname], as_index=False)["dice"].mean()

        for _, r in df.iterrows():
            key = (r["case"], int(r[sname]))
            if key in index_map and np.isfinite(r["dice"]):
                agg.setdefault(key, []).append(float(r["dice"]))

    if not agg:
        return np.array([]), np.array([])

    items = [(index_map[k], 1.0 - float(np.mean(v))) for k, v in agg.items() if k in index_map]
    items.sort(key=lambda t: t[0])
    x = np.array([t[0] for t in items], dtype=int)
    y = np.array([t[1] for t in items], dtype=float)
    return x, y

def bland_altman_group_per_model(
    model_names: List[str],
    group_classes: List[int],
    group_name: str,
    out_path: Path,
    y_range: Optional[Tuple[float, float]] = (0.0, 1.2),
    baseline_model: Optional[str] = None,
):
    baseline_model = baseline_model or baseline_model_name(model_names)
    index_map = _build_global_slice_index(model_names)
    if not index_map:
        print(f"[BA] No slices discovered for group '{group_name}'. Skipping.")
        return
    x_min, x_max = 1, max(index_map.values())

    xs, ys = [], []
    for m in model_names:
        x, y = _per_model_slice_errors_for_group(m, group_classes, index_map)
        xs.append(x); ys.append(y)

    ymin, ymax = y_range if y_range is not None else (0.0, 1.0)

    rows, cols = _grid_rc(len(model_names))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.array(axes).reshape(rows, cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx >= len(model_names):
                ax.axis("off"); continue

            mname = model_names[idx]
            x = xs[idx]; y = ys[idx]

            if y.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            else:
                ax.scatter(x, y, alpha=0.35, s=8)
                md = float(np.nanmean(y)); sd = float(np.nanstd(y))
                ax.axhline(md, color="red", linewidth=1.0)
                ax.axhline(md + 1.96*sd, color="black", linestyle="--", linewidth=1.0)
                ax.axhline(md - 1.96*sd, color="black", linestyle="--", linewidth=1.0)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(ymin, ymax)
            ax.grid(alpha=0.25)
            ax.set_title(mname)
            if r == rows - 1:
                ax.set_xlabel("Slice # (global order)")
            if c == 0:
                ax.set_ylabel("1 − mean(Dice)")
            idx += 1

    fig.suptitle(f"Slice-wise group error — {group_name}", y=0.995, fontsize=12)
    plt.tight_layout()
    _save(fig, out_path, also_pdf=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Qualitative overlays (best ckpt)
# ─────────────────────────────────────────────────────────────────────────────
def make_label_cmap(label_colors_obj, num_classes: int, bg_white: bool = True) -> ListedColormap:
    cols = []
    def _get_rgb(i):
        if isinstance(label_colors_obj, dict):
            return label_colors_obj.get(i, (0, 0, 0))
        if isinstance(label_colors_obj, (list, tuple)) and i < len(label_colors_obj):
            return label_colors_obj[i]
        return (0, 0, 0)

    for i in range(num_classes):
        if i == 0 and bg_white:
            cols.append((1.0, 1.0, 1.0))
        else:
            rgb = _get_rgb(i)
            cols.append(tuple(np.asarray(rgb, dtype=float) / 255.0))
    return ListedColormap(cols)

def _expects_3d_from_dmclass(dm_cls: type) -> bool:
    name = dm_cls.__name__.lower()
    return ("3d" in name) and ("2d" not in name)

def _adapt_input_for_model(x: torch.Tensor, expects_3d: bool) -> torch.Tensor:
    # 2D expects [B,F,H,W] ; 3D expects [B,1,D,H,W]
    if expects_3d:
        return x.unsqueeze(1) if x.ndim == 4 else x
    else:
        return x[:, 0] if x.ndim == 5 else x

def _extract_logits_from_output(out, prefer_classes: int = NUM_CLASSES) -> torch.Tensor:
    def collect(obj):
        if torch.is_tensor(obj):
            return [obj]
        if isinstance(obj, (list, tuple)):
            r = []
            for it in obj:
                r += collect(it)
            return r
        if isinstance(obj, dict):
            for k in ("logits", "y_hat", "y", "out", "output", "seg", "pred"):
                v = obj.get(k, None)
                if torch.is_tensor(v):
                    return [v]
            r = []
            for v in obj.values():
                r += collect(v)
            return r
        return []

    cands = collect(out)
    if not cands:
        raise TypeError("Model forward returned no tensors.")
    best = None
    for t in cands:
        if t.ndim in (4, 5) and (prefer_classes is None or (t.shape[1] == prefer_classes)):
            best = t if best is None or t.numel() > best.numel() else best
    return best if best is not None else max(cands, key=lambda t: t.numel())

def _align_state_dict_keys(ckpt_sd, model_sd):
    ckpt_keys = list(ckpt_sd.keys())
    model_keys = list(model_sd.keys())

    def frac_starts(keys, p):
        ks = [k for k in keys if "." in k]
        if not ks:
            return 0.0
        return sum(k.startswith(p) for k in ks) / max(1, len(ks))

    def add_prefix(sd, p):
        return {(p + k) if not k.startswith(p) else k: v for k, v in sd.items()}

    def strip_prefix(sd, p):
        return {k[len(p):] if k.startswith(p) else k: v for k, v in sd.items()}

    want_model = frac_starts(model_keys, "model.") > 0.9
    have_model = frac_starts(ckpt_keys,  "model.") > 0.9
    if want_model and not have_model:
        ckpt_sd = add_prefix(ckpt_sd, "model.")
    elif not want_model and have_model:
        ckpt_sd = strip_prefix(ckpt_sd, "model.")

    ckpt_keys = list(ckpt_sd.keys())
    want_module = frac_starts(model_keys, "module.") > 0.9
    have_module = frac_starts(ckpt_keys,  "module.") > 0.9
    if want_module and not have_module:
        ckpt_sd = add_prefix(ckpt_sd, "module.")
    elif not want_module and have_module:
        ckpt_sd = strip_prefix(ckpt_sd, "module.")

    return ckpt_sd

def qualitative_overlays_best_ckpt(
    model_names: List[str],
    indices: List[int],
    out_root: Path,
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    registry = {name: (builder_or_class, dm_factory, base_dir)
                for (name, builder_or_class, dm_factory, base_dir) in VARIANTS}

    seed = SEEDS[0] if SEEDS else 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed, workers=True)

    # Use first model's DM for test order
    ref_name = model_names[0]
    _, DMRefFactory, _ = registry[ref_name]
    dm_ref = DMRefFactory(trainval_sets, batch_size=BATCH_SIZE, num_frames=NUM_FRAMES)
    dm_ref.setup("test")
    test_loader: DataLoader = dm_ref.test_dataloader()

    handles: Dict[str, torch.nn.Module] = {}
    expects3d: Dict[str, bool] = {}

    for mname in model_names:
        builder_or_class, DMFactory, _ = registry[mname]
        ckpt_path = _find_best_or_last_ckpt(mname, seed)
        if ckpt_path is None:
            print(f"[QUAL] No checkpoint for {mname}/seed{seed}")
            continue

        prototype = None
        ModelCls = None
        if callable(builder_or_class) and not inspect.isclass(builder_or_class):
            try:
                prototype = builder_or_class()
                ModelCls = prototype.__class__
            except Exception as e:
                print(f"[QUAL] Could not instantiate prototype for {mname}: {e}")
                continue
        else:
            ModelCls = builder_or_class

        model = None
        # PL restore attempt
        try:
            model = ModelCls.load_from_checkpoint(ckpt_path, map_location=device)
        except Exception:
            model = None

        # Manual restore fallback
        if model is None:
            try:
                if prototype is None:
                    prototype = builder_or_class() if callable(builder_or_class) else ModelCls()
                ckpt = torch.load(ckpt_path, map_location=device)
                sd = ckpt.get("state_dict", ckpt)
                sd = _align_state_dict_keys(sd, prototype.state_dict())
                prototype.load_state_dict(sd, strict=False)
                model = prototype
            except Exception as e:
                print(f"[QUAL] Could not load weights for {mname}: {e}")
                continue

        model = model.to(device).eval()
        handles[mname] = model
        expects3d[mname] = _expects_3d_from_dmclass(DMFactory)

    if not handles:
        print("[QUAL] No models loaded. Skipping.")
        return

    cmap_mask = make_label_cmap(label_colors, NUM_CLASSES, bg_white=False)
    want = set(indices)
    global_idx = 0

    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            x_ref, y_ref = batch
        else:
            x_ref, y_ref = batch["image"], batch["label"]

        preds: Dict[str, torch.Tensor] = {}
        for mname, model in handles.items():
            try:
                x_in = _adapt_input_for_model(x_ref.to(device, non_blocking=True), expects3d[mname])
                with torch.no_grad():
                    raw = model(x_in)
                    logits = _extract_logits_from_output(raw, prefer_classes=NUM_CLASSES)
                    preds[mname] = logits.detach().cpu()
            except Exception as e:
                print(f"[QUAL] {mname}: forward failed ({e})")

        B = x_ref.shape[0]
        for b in range(B):
            if global_idx not in want:
                global_idx += 1
                continue

            x_item = x_ref[b].cpu()
            if x_item.ndim == 4:
                img2d = x_item[0, x_item.shape[1] // 2].numpy()
            elif x_item.ndim == 3:
                img2d = x_item[x_item.shape[0] // 2].numpy()
            else:
                img2d = np.zeros((128, 128), dtype=np.float32)

            y_center = None
            if y_ref is not None:
                y_item = y_ref[b]
                if y_item.ndim == 4 and y_item.shape[0] == 1:
                    y_item = y_item[0]
                if y_item.ndim == 3:
                    y_center = y_item[y_item.shape[0] // 2].cpu().numpy()
                elif y_item.ndim == 2:
                    y_center = y_item.cpu().numpy()

            ncols = 2 + len(preds)
            fig, axes = plt.subplots(1, ncols, figsize=(3*ncols, 5))
            axes[0].imshow(img2d, cmap="gray"); axes[0].set_title("Input"); axes[0].axis("off")

            axes[1].imshow(img2d, cmap="gray"); axes[1].set_title("GT"); axes[1].axis("off")
            if y_center is not None:
                m = np.ma.masked_where(y_center == 0, y_center)
                axes[1].imshow(m, cmap=cmap_mask, alpha=0.65, vmin=0, vmax=cmap_mask.N - 1)

            col = 2
            for mname, logit in preds.items():
                L = logit[b].numpy()  # (C,D,H,W) or (C,H,W)
                if L.ndim == 4:
                    C, D, H, W = L.shape
                    probs = []
                    for z in range(D):
                        zz = L[:, z]
                        zz = zz - zz.max(axis=0, keepdims=True)
                        p = np.exp(zz); p /= (p.sum(axis=0, keepdims=True) + 1e-8)
                        probs.append(p)
                    P = np.stack(probs, axis=0).max(axis=0)  # (C,H,W)
                    pred = P.argmax(axis=0)
                else:
                    pred = np.argmax(L, axis=0)

                axes[col].imshow(img2d, cmap="gray"); axes[col].axis("off")
                mm = np.ma.masked_where(pred == 0, pred)
                axes[col].imshow(mm, cmap=cmap_mask, alpha=0.65, vmin=0, vmax=cmap_mask.N - 1)
                axes[col].set_title(mname)
                col += 1

            legend_elems = []
            for c in range(NUM_CLASSES):
                rgb = label_colors.get(c, (0, 0, 0))
                fc = tuple(np.asarray(rgb, dtype=float) / 255.0)
                legend_elems.append(Patch(facecolor=fc, edgecolor="k", label=str(LABEL_NAMES[c])))

            fig.legend(handles=legend_elems, loc="lower center", ncol=min(10, len(legend_elems)),
                       bbox_to_anchor=(0.5, 0.02), frameon=False)
            plt.tight_layout(rect=[0, 0.06, 1, 1])

            _save(fig, out_root / f"overlay_idx{global_idx}.png")
            global_idx += 1

        if global_idx > max(indices):
            break

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_root = Path("analysis_plots")
    out_root.mkdir(parents=True, exist_ok=True)

    model_names = [name for (name, _, _, _) in VARIANTS]
    if not model_names:
        print("[MAIN] VARIANTS is empty.")
        return

    base_model = baseline_model_name(model_names)
    print(f"[MAIN] Models: {model_names}")
    print(f"[MAIN] Baseline: {base_model}")
    print(f"[MAIN] HA classes: {HA_CLASSES} -> {[LABEL_NAMES[i] for i in HA_CLASSES]}")
    print(f"[MAIN] Iodine classes: {IODINE_CLASSES} -> {[LABEL_NAMES[i] for i in IODINE_CLASSES]}")

    # (A) per-class heatmaps (single folder)
    heat_dir = out_root / "per_class_heatmaps"
    run_per_class_heatmaps(model_names, heat_dir)

    # (B) Bland–Altman group plots
    ba_dir = out_root / "bland_altman"
    ba_dir.mkdir(parents=True, exist_ok=True)

    if HA_CLASSES:
        bland_altman_group_per_model(
            model_names=model_names,
            group_classes=HA_CLASSES,
            group_name="HA",
            out_path=ba_dir / "group_HA.png",
            y_range=(0.0, 1.2),
            baseline_model=base_model,
        )
    else:
        print("[MAIN] No HA classes detected → skipping group_HA.")

    if IODINE_CLASSES:
        bland_altman_group_per_model(
            model_names=model_names,
            group_classes=IODINE_CLASSES,
            group_name="Iodine",
            out_path=ba_dir / "group_Iodine.png",
            y_range=(0.0, 1.2),
            baseline_model=base_model,
        )
    else:
        print("[MAIN] No Iodine classes detected → skipping group_Iodine.")

    # (C) Qualitative overlays (optional)
    if DO_QUAL_VIZ:
        print("[MAIN] Qualitative overlays...")
        qualitative_overlays_best_ckpt(
            model_names=model_names,
            indices=QUAL_VIZ_INDICES,
            out_root=Path(QUAL_VIZ_OUTROOT),
        )
    else:
        print("[MAIN] DO_QUAL_VIZ=0 → skipping qualitative overlays.")

    print(f"[MAIN] Done. Outputs in: {out_root.resolve()}")

if __name__ == "__main__":
    main()