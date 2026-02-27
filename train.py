#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast-capable train.py

• Training is unchanged.
• "Fast" mode only trims/short-circuits TEST-TIME extras:
    - Limits test dataloader batches (default 2)
    - Skips heavy visualizations (optional)
    - Skips per-sample PR/ROC/AUC & IoU/precision (optional)
    - Skips test_details.csv + summary.csv (optional)

How to run fast (any of the below):
    python train.py --fast
    FAST_TEST=1 python train.py
    FAST_TEST=1 FAST_SKIP_VIZ=1 FAST_SKIP_TEST_DETAILS=1 FAST_SIMPLE_METRICS=1 python train.py

Env flags (all optional):
    FAST_TEST=1                 -> enable fast path for test-only steps
    FAST_TEST_LIMIT=<int>       -> cap test batches (default 2)
    FAST_SKIP_VIZ=1             -> don't attach visualization callback
    FAST_SKIP_TEST_DETAILS=1    -> skip test_details.csv and summary.csv
    FAST_SIMPLE_METRICS=1       -> write a light test_metrics.csv (skip PR/ROC/IoU/precision loops)
"""

import os
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")  # optional, helpful logs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# from innovative3D.unified_loss import apply_unified_loss
# from innovative3D.unified_optimizer import apply_unified_optimizer

import argparse
from pathlib import Path
import csv
import pandas as pd
import torch

from inspect import isclass
from pytorch_lightning.callbacks import EarlyStopping
torch.set_float32_matmul_precision('high')  # or 'medium'
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


USE_DDP_FIND_UNUSED = False # flip True only if you KNOW some params won’t contribute to loss

from sklearn.metrics import precision_recall_curve, auc, precision_score, jaccard_score, roc_auc_score

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import Logger

@rank_zero_only
def r0print(*args, **kwargs):
    print(*args, **kwargs)

# ── Project config ─────────────────────────────────────────────────────────────
from innovative3D.config import (
    SEEDS,
    VARIANTS,  # (model_name, BuilderOrClass, DataModuleClass, base_ckpt_folder)
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_FRAMES,
    NUM_CLASSES,
    BEST_LR,
    FINAL_EPOCHS,
    trainval_sets,
    # used by overlays
    global_label_names,
    label_colors,
    IMAGE_HEIGHT, IMAGE_WIDTH,     
)



# Optional background / ignore ids (used for overlays)
try:
    from innovative3D.config import BACKGROUND_CLASS
    BACKGROUND_ID = int(BACKGROUND_CLASS)
except Exception:
    BACKGROUND_ID = 0
try:
    from innovative3D.config import IGNORE_INDEX
    IGNORE_ID = int(IGNORE_INDEX)
except Exception:
    IGNORE_ID = None

# Normalize label colors: ensure int keys and RGB uint8 arrays
LABEL_COLORS = {int(k): np.array(v, dtype=np.uint8)[:3] for k, v in label_colors.items()}

# Optional 2D/3D metrics
try:
    from innovative3D.helpers import per_class_metrics_2d, per_class_metrics_3d
except ModuleNotFoundError:
    from helpers import per_class_metrics_2d, per_class_metrics_3d



# ── FAST MODE FLAGS ───────────────────────────────────────────────────────────
def _env_flag(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default)
    return str(val).lower() in ("1", "true", "yes", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

FAST_TEST = _env_flag("FAST_TEST", "0")
FAST_TEST_LIMIT = _env_int("FAST_TEST_LIMIT", 2)   # limit_test_batches if FAST_TEST
FAST_SKIP_VIZ = _env_flag("FAST_SKIP_VIZ", "0")   # skip VisualizeEveryNEpochsBuffered
FAST_SKIP_TEST_DETAILS = _env_flag("FAST_SKIP_TEST_DETAILS", "0")
FAST_SIMPLE_METRICS = _env_flag("FAST_SIMPLE_METRICS", "0")

# Allow viz on multi-GPU (rank-0 only)
FORCE_VIZ_MULTI_GPU = _env_flag("FORCE_VIZ_MULTI_GPU", "0")
# Put this near the top of train.py, before the scan-label helper functions:
import os
try:
    SCAN_MIN_VOX
except NameError:
    SCAN_MIN_VOX = int(os.getenv("SCAN_MIN_VOX", "800"))
# --- compute imports ---
import copy
try:
    from thop import profile
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False

try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE = True
except Exception:
    _HAS_FVCORE = False

# ── SCAN/REGION LABELS (post-processing) ────────────────────────────
# 
# 
# 
# 
# 
# 
# e VS Code
_IS_VSC = ("VSCODE_PID" in os.environ) or (os.environ.get("TERM_PROGRAM", "").lower() == "vscode")
if _IS_VSC:
    FORCE_VIZ_MULTI_GPU = True
    FAST_SKIP_VIZ = False

SKIP_VIZ = int(os.getenv("SKIP_VIZ", "0"))
# ── helpers: detect test shape & write details ────────────────────────────────
# --- normalize model outputs: pick the main/highest-res logits tensor ---
def _select_main_logits(out):
    """Return a Tensor from model output that may be Tensor | tuple/list | dict."""
    import torch
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        def _spatial_vol(t):
            if not isinstance(t, torch.Tensor): return -1
            if t.ndim >= 5:  # [B,C,D,H,W]
                D, H, W = t.shape[-3:]
                return int(D) * int(H) * int(W)
            if t.ndim == 4:  # [B,C,H,W]
                H, W = t.shape[-2:]
                return int(H) * int(W)
            return t.numel()
        try:
            return max(out, key=_spatial_vol)
        except Exception:
            return out[0]
    if isinstance(out, dict):
        for k in ("logits", "y_main", "main", "out"):
            v = out.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f"Unsupported output type for logits selection: {type(out)}")

def _peek_test_shape(dm):
    dl = dm.test_dataloader()
    batch = next(iter(dl))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch["image"]
    return x.ndim, (x.shape[1] if x.ndim >= 2 else None)

@torch.no_grad()
def write_test_details_planar(model, dm, out_csv_path: Path, num_classes: int, ignore_index: int = IGNORE_ID):
    model.eval()
    rows = []
    case_id = 0
    for batch in dm.test_dataloader():
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x, y = batch["image"], batch["label"]
        # logits = model(x)  # [B,C,H,W]
        out = model(x)
        logits = _select_main_logits(out)  # NEW

        if logits.ndim != 4:
            raise ValueError(f"Planar details expected 4D logits, got {tuple(logits.shape)}")
        if y.ndim == 4 and y.shape[1] == 1:
            y_use = y[:, 0]
        elif y.ndim == 3:
            y_use = y
        else:
            raise ValueError(f"Unexpected planar label shape: {tuple(y.shape)}")
        preds = torch.argmax(logits, dim=1)  # [B,H,W]
        B = preds.shape[0]
        for b in range(B):
            lbl_b = y_use[b]
            pred_b = preds[b]
            mask_b = (lbl_b != ignore_index) if ignore_index is not None else torch.ones_like(lbl_b, dtype=torch.bool)
            valid = mask_b
            n_total = int(valid.sum().item())
            for c in range(num_classes):
                pc = (pred_b == c) & valid
                lc = (lbl_b  == c) & valid
                tp = int((pc & lc).sum().item())
                fp = int((pc & (~lc)).sum().item())
                fn = int(((~pc) & lc).sum().item())
                tn = int(((~pc) & (~lc)).sum().item())
                n_pos = int(lc.sum().item())
                n_neg = int((~lc & valid).sum().item())
                n_pred_pos = int(pc.sum().item())
                if n_pos == 0:
                    dice = float('nan'); sens = float('nan')
                else:
                    denom_d = (2*tp + fp + fn)
                    dice = float((2*tp) / denom_d) if denom_d > 0 else float('nan')
                    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
                spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')
                # >>> add these two lines here <<<
                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float('nan')
                iou  = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float('nan')
                # <<<
                rows.append({
                    "case": int(case_id),
                    "class": int(c),
                    "dice": dice,
                    "sensitivity": sens,
                    "specificity": spec,
                    "precision": prec,          # <-- NEW
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "n_pos": n_pos, "n_neg": n_neg, "n_pred_pos": n_pred_pos,
                    "present_gt": int(n_pos > 0),
                    "n_total_valid": n_total,
                })
            case_id += 1
    if rows:
        pd.DataFrame(rows).to_csv(out_csv_path, index=False)

@torch.no_grad()
def write_test_details_3d(model, dm, out_csv_path: Path, num_classes: int, ignore_index: int = IGNORE_ID):
    model.eval()
    rows = []
    case_id = 0
    for batch in dm.test_dataloader():
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x, y = batch["image"], batch["label"]
        # logits = model(x)  # [B,C,D,H,W]
        out = model(x)
        logits = _select_main_logits(out)  # NEW

        if logits.ndim != 5:
            raise ValueError(f"3D details expected 5D logits, got {tuple(logits.shape)}")
        if y.ndim == 5 and y.shape[1] == 1:
            y_use = y[:, 0]
        elif y.ndim == 4:
            y_use = y
        else:
            raise ValueError(f"Unexpected 3D label shape: {tuple(y.shape)}")
        preds = torch.argmax(logits, dim=1)  # [B,D,H,W]
        B = preds.shape[0]
        for b in range(B):
            lbl_b = y_use[b]
            pred_b = preds[b]
            mask_b = (lbl_b != ignore_index) if ignore_index is not None else torch.ones_like(lbl_b, dtype=torch.bool)
            valid = mask_b
            n_total = int(valid.sum().item())
            for c in range(num_classes):
                pc = (pred_b == c) & valid
                lc = (lbl_b  == c) & valid
                tp = int((pc & lc).sum().item())
                fp = int((pc & (~lc)).sum().item())
                fn = int(((~pc) & lc).sum().item())
                tn = int(((~pc) & (~lc)).sum().item())
                n_pos = int(lc.sum().item())
                n_neg = int((~lc & valid).sum().item())
                n_pred_pos = int(pc.sum().item())
                if n_pos == 0:
                    dice = float('nan'); sens = float('nan')
                else:
                    denom_d = (2*tp + fp + fn)
                    dice = float((2*tp) / denom_d) if denom_d > 0 else float('nan')
                    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
                spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')

                
                # >>> add these two lines here <<<
                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float('nan')
                iou  = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float('nan')
                # <<<

                rows.append({
                    "case": int(case_id),
                    "class": int(c),
                    "dice": dice,
                    "sensitivity": sens,
                    "specificity": spec,
                    "precision": prec,          # <-- NEW
                    "iou": iou,                 # <-- NEW
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "n_pos": n_pos, "n_neg": n_neg, "n_pred_pos": n_pred_pos,
                    "present_gt": int(n_pos > 0),
                    "n_total_valid": n_total,
                })
            case_id += 1
    if rows:
        pd.DataFrame(rows).to_csv(out_csv_path, index=False)

def write_summary_csv(details_csv: Path):
    if not details_csv.exists():
        r0print(f"[WARN] No test_details.csv found at {details_csv}")
        return None
    df = pd.read_csv(details_csv)
    df.columns = [c.lower() for c in df.columns]

    cls_col = "class" if "class" in df.columns else ("label" if "label" in df.columns else None)
    if cls_col is None:
        r0print("[WARN] test_details.csv missing 'class'/'label'; skipping summary.")
        return None

    # pick the metrics that actually exist
    metric_cols = [m for m in ("dice","sensitivity","specificity","precision","iou") if m in df.columns]
    if not metric_cols:
        r0print("[WARN] test_details.csv has no known metric columns; skipping summary.")
        return None

    summary = df.groupby(cls_col)[metric_cols].agg(["mean","std"]).round(4)
    summary.columns = ["_".join(cols) for cols in summary.columns.to_flat_index()]
    out = details_csv.parent / "summary.csv"
    summary.to_csv(out)
    r0print(f"[INFO] Summary saved to {out}")
    return out


# def write_summary_csv(details_csv: Path):
#     if not details_csv.exists():
#         r0print(f"[WARN] No test_details.csv found at {details_csv}")
#         return None
#     df = pd.read_csv(details_csv)
#     df.columns = [c.lower() for c in df.columns]
#     cls_col = "class" if "class" in df.columns else ("label" if "label" in df.columns else None)
#     if cls_col is None:
#         r0print("[WARN] test_details.csv missing 'class'/'label'; skipping summary.")
#         return None
#     need = {"dice", "sensitivity", "specificity"}
#     if not need.issubset(set(df.columns)):
#         r0print("[WARN] test_details.csv missing metric columns; skipping summary.")
#         return None
#     summary = df.groupby(cls_col)[["dice","sensitivity","specificity"]].agg(["mean","std"]).round(4)
#     summary.columns = ["_".join(cols) for cols in summary.columns.to_flat_index()]
#     out = details_csv.parent / "summary.csv"
#     summary.to_csv(out)
#     r0print(f"[INFO] Summary saved to {out}")
#     return out
def _scan_labels_from_mask_np(pred_mask_np: np.ndarray,
                              num_classes: int,
                              min_vox: int = SCAN_MIN_VOX,
                              background: int = BACKGROUND_ID) -> np.ndarray:
    """
    Return a (num_classes,) 0/1 vector: class present if predicted voxels >= min_vox.
    Background is ignored.
    """
    labels = np.zeros(num_classes, dtype=np.int32)
    for k in range(num_classes):
        if k == background:
            continue
        labels[k] = int((pred_mask_np == k).sum() >= int(min_vox))
    return labels

# @torch.no_grad()
# def write_scan_labels_csv(model,
#                           dm,
#                           out_csv_path: Path,
#                           num_classes: int,
#                           min_vox: int = SCAN_MIN_VOX):
#     """
#     Runs the model on the test loader, converts argmax masks → scan-level multi-labels,
#     and writes one row per case: [case, scan_<name_or_class_idx>].
#     Uses the largest-resolution logits via _select_main_logits.
#     """
#     model.eval()
#     rows = []
#     case_id = 0

#     # Try to use label names if they match num_classes
#     try:
#         names = global_label_names if isinstance(global_label_names, (list, tuple)) and len(global_label_names) == num_classes else None
#     except NameError:
#         names = None

#     for batch in dm.test_dataloader():
#         # images only; we don't need y for this postproc
#         if isinstance(batch, (list, tuple)):
#             x = batch[0]
#         else:
#             x = batch["image"]

#         out = model(x)
#         logits = _select_main_logits(out)     # [B,C,...]
#         preds = torch.argmax(logits, dim=1)   # [B,...]
#         preds_np = preds.detach().cpu().numpy()

#         B = preds_np.shape[0]
#         for b in range(B):
#             scan_vec = _scan_labels_from_mask_np(
#                 preds_np[b], num_classes, min_vox=min_vox, background=BACKGROUND_ID
#             )
#             row = {"case": int(case_id)}
#             if names:
#                 for k in range(num_classes):
#                     row[f"scan_{names[k]}"] = int(scan_vec[k])
#             else:
#                 for k in range(num_classes):
#                     row[f"scan_class_{k}"] = int(scan_vec[k])
#             rows.append(row)
#             case_id += 1

#     if rows:
#         out_csv_path.parent.mkdir(parents=True, exist_ok=True)
#         pd.DataFrame(rows).to_csv(out_csv_path, index=False)

@torch.no_grad()
def write_scan_labels_csv(model,
                          dm,
                          out_csv_path: Path,
                          num_classes: int,
                          min_vox: int = SCAN_MIN_VOX):
    model.eval()
    rows = []
    case_id = 0

    # model device
    device = next(model.parameters()).device

    # Try to use label names if they match num_classes
    try:
        names = global_label_names if isinstance(global_label_names, (list, tuple)) and len(global_label_names) == num_classes else None
    except NameError:
        names = None

    for batch in dm.test_dataloader():
        # images only; we don't need y for this postproc
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch["image"]

        # ✅ move to the same device as the model (handle list/tuple too)
        if isinstance(x, (list, tuple)):
            x_dev = [t.to(device, non_blocking=True) for t in x]
            out = model(x_dev)
        else:
            x_dev = x.to(device, non_blocking=True)
            out = model(x_dev)

        logits = _select_main_logits(out)     # [B,C,...]
        preds = torch.argmax(logits, dim=1)   # [B,...]
        preds_np = preds.detach().cpu().numpy()

        B = preds_np.shape[0]
        for b in range(B):
            scan_vec = _scan_labels_from_mask_np(
                preds_np[b], num_classes, min_vox=min_vox, background=BACKGROUND_ID
            )
            row = {"case": int(case_id)}
            if names:
                for k in range(num_classes):
                    row[f"scan_{names[k]}"] = int(scan_vec[k])
            else:
                for k in range(num_classes):
                    row[f"scan_class_{k}"] = int(scan_vec[k])
            rows.append(row)
            case_id += 1

    if rows:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv_path, index=False)

# ── auto-resume helper ────────────────────────────────────────────────────────
def _resolve_resume_ckpt(ckpt_folder: Path):
    last = ckpt_folder / "last.ckpt"
    if last.exists():
        return str(last)
    ckpts = sorted(ckpt_folder.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(ckpts[0]) if ckpts else None

# ── Phase CSV logger (kept for reference; not used by default) ────────────────
class PhaseCSVLogger(Logger):
    def __init__(self, save_dir: str, name: str = "logs"):
        super().__init__()
        self.root = Path(save_dir) / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = {
            "train": self.root / "metrics.csv",
            "val":   self.root / "metrics.csv",
            "test":  self.root / "test_metrics.csv",
        }
        self.fieldnames = {"train": None, "val": None, "test": None}
        self.last_epoch = {"train": -1, "val": -1, "test": -1}
        self._bootstrap()
    @property
    def name(self): return "phase_csv"
    @property
    def version(self): return ""
    @property
    def log_dir(self): return str(self.root)
    def log_hyperparams(self, params): pass
    def finalize(self, status): pass
    def _bootstrap(self):
        for phase in ("train","val","test"):
            fp = self.files[phase]
            if not fp.exists(): continue
            try:
                with fp.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    self.fieldnames[phase] = reader.fieldnames
                    for row in reader:
                        e = row.get("epoch", "")
                        try:
                            e = int(float(e))
                            self.last_epoch[phase] = max(self.last_epoch[phase], e)
                        except Exception:
                            pass
            except Exception:
                pass
    def _phase_of(self, metrics: dict):
        has = {p: any(k.startswith(f"{p}_") for k in metrics.keys()) for p in ("train","val","test")}
        phases = [p for p, ok in has.items() if ok]
        if len(phases) != 1:
            return None
        return phases[0]
    def _filter_keys(self, metrics: dict):
        keep = {"epoch", "step"}
        return {k: v for k, v in metrics.items()
                if k in keep or k.startswith("train_") or k.startswith("val_") or k.startswith("test_")}
    @rank_zero_only
    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        phase = self._phase_of(metrics)
        if phase is None:
            return
        metrics = self._filter_keys(metrics)
        if "epoch" not in metrics or metrics["epoch"] is None or metrics["epoch"] == "":
            if phase == "test":
                metrics["epoch"] = self.last_epoch["test"] + 1
            else:
                return
        try:
            epoch = int(float(metrics["epoch"]))
        except Exception:
            return
        if epoch <= self.last_epoch[phase]:
            return
        existing = self.fieldnames[phase]
        new_keys = ["epoch"]
        if "step" in metrics:
            new_keys.append("step")
        new_keys += sorted(k for k in metrics.keys() if k not in ("epoch", "step"))
        if existing is None:
            self.fieldnames[phase] = new_keys
            write_header = not self.files[phase].exists() or self.files[phase].stat().st_size == 0
            with self.files[phase].open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames[phase])
                if write_header:
                    w.writeheader()
                w.writerow({k: metrics.get(k, "") for k in self.fieldnames[phase]})
        else:
            missing = [k for k in new_keys if k not in existing]
            if missing:
                self.fieldnames[phase] += missing
                old_rows = []
                if self.files[phase].exists() and self.files[phase].stat().st_size > 0:
                    with self.files[phase].open("r", newline="") as f:
                        r = csv.DictReader(f)
                        for row in r:
                            old_rows.append(row)
                with self.files[phase].open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=self.fieldnames[phase])
                    w.writeheader()
                    for row in old_rows:
                        w.writerow({k: row.get(k, "") for k in self.fieldnames[phase]})
            with self.files[phase].open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames[phase])
                w.writerow({k: metrics.get(k, "") for k in self.fieldnames[phase]})
        self.last_epoch[phase] = epoch

# ── tiny viz helpers ─────────────────────────────────────────────────────────
def _softmax_over_classes(logits: np.ndarray) -> np.ndarray:
    L = logits - logits.max(axis=0, keepdims=True)
    P = np.exp(L); P /= P.sum(axis=0, keepdims=True) + 1e-8
    return P

def _center_frame_from_input(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:  # (C,D,H,W)
        _, D, _, _ = x.shape
        return x[0, D // 2].detach().cpu().numpy()
    elif x.ndim == 3:  # (C,H,W)
        C, _, _ = x.shape
        return x[C // 2].detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected input ndim={x.ndim}")

def _mip_from_input_tensor(x_item: torch.Tensor) -> np.ndarray:
    if x_item.ndim == 4:
        mip = x_item[0].max(dim=0).values.detach().cpu().numpy()
        return mip
    elif x_item.ndim == 3:
        C = x_item.shape[0]
        return x_item[C // 2].detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected input ndim={x_item.ndim}")

def _color_mask(mask: np.ndarray, color_map: dict) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in color_map.items():
        out[mask == c] = col
    return out

def _is_foreground(pred_mask: np.ndarray) -> np.ndarray:
    fg = pred_mask != BACKGROUND_ID
    if IGNORE_ID is not None and IGNORE_ID >= 0:
        fg &= pred_mask != IGNORE_ID
    return fg

def _prediction_views_and_mip(logits: np.ndarray):
    def _softmax_over_classes_np(L):
        L = L - L.max(axis=0, keepdims=True)
        P = np.exp(L); P /= P.sum(axis=0, keepdims=True) + 1e-8
        return P
    if logits.ndim == 4:  # (C,D,H,W)
        C, D, H, W = logits.shape
        mid = D // 2
        Pc = _softmax_over_classes_np(logits[:, mid])  # (C,H,W)
        center_pred = Pc.argmax(axis=0)
        alpha_center = Pc.max(axis=0)
        probs_over_d = np.stack([_softmax_over_classes_np(logits[:, z]) for z in range(D)], axis=1)  # (C,D,H,W)
        summary_pred = probs_over_d.max(axis=1).argmax(axis=0)  # (H,W)
        mip_probs = probs_over_d.max(axis=1)  # (C,H,W)
        mip_pred = mip_probs.argmax(axis=0)
        return center_pred, summary_pred, alpha_center, mip_pred
    elif logits.ndim == 3:  # (C,H,W)
        Pc = _softmax_over_classes_np(logits)
        center_pred = Pc.argmax(axis=0)
        alpha_center = Pc.max(axis=0)
        return center_pred, center_pred, alpha_center, center_pred
    else:
        raise ValueError(f"Unexpected logits ndim={logits.ndim}")

# apply_unified_loss()                 # makes everyone use ce_plus_macro_dice_loss
# apply_unified_optimizer(lr=1e-4)  
# ── test_metrics.csv: writer (FAST path supported) ───────────────────────────
@torch.no_grad()
def write_test_metrics_csv_from_pass(
    model,
    dm,
    out_csv_path: Path,
    num_classes: int,
    ignore_index: int = IGNORE_ID,
    per_case: bool = False,
    is_3d: bool | None = None,
    max_batches: int | None = None,
    fast_simple: bool = False,
):
    
    """
    If fast_simple=True:
      - computes only per-class Dice/Sens/Spec (mean over processed batches)
      - computes macro/micro dice/sens/spec
      - skips PR/ROC/IoU/precision loops (columns left empty)
      - respects max_batches cap on test dataloader
    """
    model.eval()
    device = next(model.parameters()).device
    dl = dm.test_dataloader()
    if is_3d is None:
        try:
            batch0 = next(iter(dl))
        except StopIteration:
            r0print("[WARN] Empty test dataloader; skipping write_test_metrics_csv_from_pass")
            return
        x0, y0 = (batch0 if isinstance(batch0, (list, tuple)) else (batch0["image"], batch0["label"]))
        if isinstance(x0, (list, tuple)) and len(x0) > 0:
            x0 = x0[0]
        is_3d = (x0.ndim == 5)
        try:
            dl = dm.test_dataloader()
        except Exception:
            pass

    dice_all, sens_all, spec_all = [], [], []
    micro_dice_list, micro_sens_list, micro_spec_list = [], [], []

    # These are only used in the slow/full path:
    per_class_pr = [[] for _ in range(num_classes)]
    per_class_roc = [[] for _ in range(num_classes)]
    per_class_iou = [[] for _ in range(num_classes)]
    per_class_prec= [[] for _ in range(num_classes)]

    for ib, batch in enumerate(dl):
        if (max_batches is not None) and (ib >= max_batches):
            break
        x, y = (batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"]))
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # logits = model(x)
        out = model(x)
        logits = _select_main_logits(out)  # NEW

        
        if is_3d:
            fn = per_class_metrics_3d
            y_use = y[:, 0] if (y.ndim == 5 and y.shape[1] == 1) else y
        else:
            fn = per_class_metrics_2d
            y_use = y[:, 0] if (y.ndim == 4 and y.shape[1] == 1) else y

        (dice_list, sens_list, spec_list,
         macro_dice, macro_sens, macro_spec,
         micro_dice, micro_sens, micro_spec) = fn(
            logits, y_use, num_classes, ignore_index=ignore_index
        )
        dice_np = np.array([float(v) for v in dice_list])
        sens_np = np.array([float(v) for v in sens_list])
        spec_np = np.array([float(v) for v in spec_list])

        micro_dice_list.append(float(micro_dice))
        micro_sens_list.append(float(micro_sens))
        micro_spec_list.append(float(micro_spec))

        dice_all.append(dice_np)
        sens_all.append(sens_np)
        spec_all.append(spec_np)

        if fast_simple:
            # Skip expensive PR/ROC/IoU calculations entirely
            continue

        # ---- Full path: compute PR/ROC/IoU/precision per class (slow) ----
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        probs_np = probs.detach().cpu().numpy()
        y_np = y_use.detach().cpu().numpy()
        pred_np = preds.detach().cpu().numpy()

        B = probs_np.shape[0]
        for b in range(B):
            for c in range(num_classes):
                y_true = (y_np[b] == c).ravel().astype(np.int32)
                y_score = probs_np[b, c].ravel()
                y_pred  = (pred_np[b] == c).ravel().astype(np.int32)
                if y_true.sum() > 0 and y_true.sum() < y_true.size:
                    p, r, _ = precision_recall_curve(y_true, y_score)
                    pr_val = auc(r, p)
                    try:
                        roc_val = roc_auc_score(y_true, y_score)
                    except Exception:
                        roc_val = np.nan
                else:
                    pr_val = np.nan; roc_val = np.nan
                try:
                    iou_val = jaccard_score(y_true, y_pred) if (y_true.sum() > 0 or y_pred.sum() > 0) else np.nan
                except Exception:
                    iou_val = np.nan
                try:
                    prec_val = precision_score(y_true, y_pred, zero_division=0) if (y_pred.sum() + y_true.sum()) > 0 else np.nan
                except Exception:
                    prec_val = np.nan
                per_class_pr[c].append(pr_val)
                per_class_roc[c].append(roc_val)
                per_class_iou[c].append(iou_val)
                per_class_prec[c].append(prec_val)

    dice_all = np.vstack(dice_all) if len(dice_all) else np.zeros((0, num_classes))
    sens_all = np.vstack(sens_all) if len(sens_all) else np.zeros((0, num_classes))
    spec_all = np.vstack(spec_all) if len(spec_all) else np.zeros((0, num_classes))

    run = {}
    for c in range(num_classes):
        if dice_all.size:
            run[f"test_dice_class_{c}"] = float(np.nanmean(dice_all[:, c]))
            run[f"test_sens_class_{c}"] = float(np.nanmean(sens_all[:, c]))
            run[f"test_spec_class_{c}"] = float(np.nanmean(spec_all[:, c]))
        else:
            run[f"test_dice_class_{c}"] = ""
            run[f"test_sens_class_{c}"] = ""
            run[f"test_spec_class_{c}"] = ""
        if fast_simple:
            # Leave derived columns empty in fast mode
            run[f"test_pr_auc_class_{c}"] = ""
            run[f"test_roc_auc_class_{c}"] = ""
            run[f"test_iou_class_{c}"] = ""
            run[f"test_precision_class_{c}"] = ""
        else:
            pc_pr  = np.array(per_class_pr[c],  dtype=float)
            pc_roc = np.array(per_class_roc[c], dtype=float)
            pc_iou = np.array(per_class_iou[c], dtype=float)
            pc_pre = np.array(per_class_prec[c],dtype=float)
            run[f"test_pr_auc_class_{c}"] = float(np.nanmean(pc_pr))  if pc_pr.size  else ""
            run[f"test_roc_auc_class_{c}"] = float(np.nanmean(pc_roc)) if pc_roc.size else ""
            run[f"test_iou_class_{c}"] = float(np.nanmean(pc_iou))    if pc_iou.size else ""
            run[f"test_precision_class_{c}"] = float(np.nanmean(pc_pre)) if pc_pre.size else ""

    def macro_mean(arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 0: return ""
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr[:, 1:]
            return float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else ""
        if arr.ndim == 1 and arr.size > 1:
            arr = arr[1:]
            return float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else ""
        return float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else ""

    run["test_macro_dice"] = macro_mean(dice_all)
    run["test_macro_sens"] = macro_mean(sens_all)
    run["test_macro_spec"] = macro_mean(spec_all)

    run["test_micro_dice"] = float(np.nanmean(np.array(micro_dice_list))) if micro_dice_list else ""
    run["test_micro_sens"] = float(np.nanmean(np.array(micro_sens_list))) if micro_sens_list else ""
    run["test_micro_spec"] = float(np.nanmean(np.array(micro_spec_list))) if micro_spec_list else ""

    if fast_simple:
        # Keep macro PR/ROC/IoU/precision empty in fast mode
        run["test_pr_auc_macro"] = ""
        run["test_roc_auc_macro"] = ""
        run["test_iou_macro"] = ""
        run["test_precision_macro"] = ""
    else:
        def macro_from_pc(pc_lists):
            vals = []
            for c in range(1, num_classes):
                v = np.array(pc_lists[c], dtype=float)
                if v.size: vals.append(np.nanmean(v))
            return float(np.nanmean(vals)) if vals else ""
        run["test_pr_auc_macro"] = macro_from_pc(per_class_pr)
        run["test_roc_auc_macro"] = macro_from_pc(per_class_roc)
        run["test_iou_macro"] = macro_from_pc(per_class_iou)
        run["test_precision_macro"] = macro_from_pc(per_class_prec)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if out_csv_path.exists():
        old = pd.read_csv(out_csv_path)
        for c in run.keys():
            if c not in old.columns:
                old[c] = ""
        new_row = pd.DataFrame([run])[old.columns]
        df = pd.concat([old, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([run])
    df.to_csv(out_csv_path, index=False)
    r0print(f"[INFO] Wrote test metrics to {out_csv_path}")

# ── Visualization callback (unchanged; can be skipped in FAST mode) ──────────
class VisualizeEveryNEpochsBuffered(Callback):
    def __init__(self, out_root, every_n_epochs=1, max_items=2, dpi=200, save_pdf=False):
        super().__init__()
        self.out_root = Path(out_root)
        self.every = max(1, int(every_n_epochs))
        self.max_items = max(1, int(max_items))
        self.dpi = dpi
        self.save_pdf = save_pdf
        self._last_train = None
        self._last_val = None

    def _is_rank0(self, trainer): return getattr(trainer, "is_global_zero", True)

    def _save_png_atomic(self, fig, path, dpi=200, bbox_inches="tight"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part.png")
        fig.savefig(tmp, dpi=dpi, bbox_inches=bbox_inches, format="png")
        plt.close(fig)
        os.replace(tmp, path)

    def _center_frame_from_input(self, x):
        if x.ndim == 4:
            _, D, _, _ = x.shape
            return x[0, D // 2].detach().cpu().numpy()
        elif x.ndim == 3:
            C, _, _ = x.shape
            return x[C // 2].detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected input ndim={x.ndim}")

    def _mip_from_input_tensor(self, x_item):
        if x_item.ndim == 4:
            return x_item[0].max(dim=0).values.detach().cpu().numpy()
        elif x_item.ndim == 3:
            C = x_item.shape[0]
            return x_item[C // 2].detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected input ndim={x_item.ndim}")

    def _color_mask(self, mask, color_map):
        h, w = mask.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for c, col in color_map.items():
            out[mask == c] = col
        return out

    def _softmax_over_classes_np(self, L):
        L = L - L.max(axis=0, keepdims=True)
        P = np.exp(L); P /= P.sum(axis=0, keepdims=True) + 1e-8
        return P

    def _prediction_views_and_mip(self, logits):
        if logits.ndim == 4:
            C, D, H, W = logits.shape
            mid = D // 2
            Pc = self._softmax_over_classes_np(logits[:, mid])
            center_pred = Pc.argmax(axis=0)
            alpha_center = Pc.max(axis=0)
            probs_over_d = np.stack([self._softmax_over_classes_np(logits[:, z]) for z in range(D)], axis=1)
            summary_pred = probs_over_d.max(axis=1).argmax(axis=0)
            mip_probs = probs_over_d.max(axis=1)
            mip_pred = mip_probs.argmax(axis=0)
            return center_pred, summary_pred, alpha_center, mip_pred
        elif logits.ndim == 3:
            Pc = self._softmax_over_classes_np(logits)
            center_pred = Pc.argmax(axis=0)
            alpha_center = Pc.max(axis=0)
            return center_pred, center_pred, alpha_center, center_pred
        else:
            raise ValueError(f"Unexpected logits ndim={logits.ndim}")

    def _frames_list(self, x_item):
        if x_item.ndim == 4:
            return [x_item[0, d].detach().cpu().numpy() for d in range(x_item.shape[1])]
        elif x_item.ndim == 3:
            C = x_item.shape[0]
            return [x_item[c].detach().cpu().numpy() for c in range(C)]
        else:
            raise ValueError(f"Unexpected input ndim={x_item.ndim}")

    def _save_frames_montage(self, x_item, save_path, cols=5):
        frames = self._frames_list(x_item)
        F = len(frames)
        cols = min(cols, F) if F else 1
        rows = int(np.ceil(max(1, F) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = np.array(axes).reshape(rows, cols)
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.axis("off")
            if i < F:
                ax.imshow(frames[i], cmap="gray")
                ax.set_title(f"f={i}")
        fig.suptitle("Input frames montage (shows grid-shuffle if present)")
        fig.tight_layout()
        self._save_png_atomic(fig, save_path, dpi=200)

    # def _stash(self, batch, pl_module, store_attr: str):
    #     if isinstance(batch, (list, tuple)):
    #         x, y = batch
    #     else:
    #         x, y = batch.get("image"), batch.get("label", None)
    #     device = pl_module.device
    #     with torch.no_grad():
    #         was_training = pl_module.training
    #         pl_module.eval()
    #         if isinstance(x, (list, tuple)):
    #             x_dev = [t.to(device, non_blocking=True) for t in x]
    #             logits = pl_module(x_dev)
    #             x_display = x[0].detach().cpu()
    #         else:
    #             x_dev = x.to(device, non_blocking=True)
    #             logits = pl_module(x_dev)
    #             x_display = x.detach().cpu()
    #         if was_training:
    #             pl_module.train()
    #     y_cpu = (y.detach().cpu() if (y is not None and hasattr(y, "detach")) else None)
    #     setattr(self, store_attr, {"x": x_display, "y": y_cpu, "logits": logits.detach().cpu()})
    def _stash(self, batch, pl_module, store_attr: str):
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x, y = batch.get("image"), batch.get("label", None)

        device = pl_module.device
        with torch.no_grad():
            was_training = pl_module.training
            pl_module.eval()

            # Move to device and run forward
            if isinstance(x, (list, tuple)):
                x_dev = [t.to(device, non_blocking=True) for t in x]
                # out = pl_module(x_dev)
                out = pl_module(x_dev)
                logits = _select_main_logits(out)  # NEW

                x_display = x[0].detach().cpu()
            else:
                x_dev = x.to(device, non_blocking=True)
                # out = pl_module(x_dev)
                out = pl_module(x_dev)
                logits = _select_main_logits(out)  # NEW

                x_display = x.detach().cpu()

            # --- NEW: pick a single tensor to visualize ---
            def _pick_main_logits(o):
                # dict with logits
                if isinstance(o, dict) and "logits" in o and torch.is_tensor(o["logits"]):
                    return o["logits"]
                # tuple/list of tensors (deep supervision)
                if isinstance(o, (tuple, list)):
                    tensors = [t for t in o if torch.is_tensor(t)]
                    if len(tensors) == 0:
                        raise TypeError("Model output is a tuple/list but contains no tensors.")
                    # choose the one with the largest spatial volume
                    def _vol(t):
                        if t.ndim >= 5:
                            return int(t.shape[-3] * t.shape[-2] * t.shape[-1])
                        elif t.ndim == 4:
                            return int(t.shape[-2] * t.shape[-1])
                        return 0
                    return max(tensors, key=_vol)
                # already a tensor
                if torch.is_tensor(o):
                    return o
                raise TypeError(f"Unsupported model output type for viz: {type(o)}")

            logits = _pick_main_logits(out)

            if was_training:
                pl_module.train()

        y_cpu = (y.detach().cpu() if (y is not None and hasattr(y, "detach")) else None)
        setattr(self, store_attr, {"x": x_display, "y": y_cpu, "logits": logits.detach().cpu()})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._is_rank0(trainer):
            return
        if batch_idx == 0:
            self._stash(batch, pl_module, "_last_train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self._is_rank0(trainer):
            return
        if batch_idx == 0:
            self._stash(batch, pl_module, "_last_val")

    def _render_buffer(self, buf: dict, epoch: int, phase: str, LABEL_COLORS=None, global_label_names=None):
        if not buf:
            return
        save_dir = self.out_root / "viz" / f"{phase}_epoch{epoch+1:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        x = buf["x"]
        y = buf["y"]
        L = buf["logits"]
        B = x.shape[0]
        for b in range(min(self.max_items, B)):
            x_item = x[b]
            center_img = self._center_frame_from_input(x_item)
            mip_img = self._mip_from_input_tensor(x_item)
            self._save_frames_montage(x_item, save_dir / f"frames_montage_b{b}.png", cols=5)
            y_center = None
            if y is not None:
                y_item = y[b]
                if y_item.ndim == 4 and y_item.shape[0] == 1:
                    y_item = y_item[0]
                if y_item.ndim == 3:
                    y_center = y_item[y_item.shape[0] // 2].numpy()
                elif y_item.ndim == 2:
                    y_center = y_item.numpy()
            logits_b = L[b].numpy()
            center_pred, summary_pred, alpha_center, mip_pred = self._prediction_views_and_mip(logits_b)
            fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=False)
            axes[0].imshow(center_img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(center_img, cmap="gray")
            if y_center is not None:
                axes[1].imshow(_color_mask(y_center.astype(np.int64), LABEL_COLORS), alpha=0.85, interpolation="none")
            axes[1].set_title("Ground Truth"); axes[1].axis("off")
            axes[2].imshow(center_img, cmap="gray")
            axes[2].imshow(_color_mask(center_pred.astype(np.int64), LABEL_COLORS), alpha=0.85, interpolation="none")
            axes[2].set_title("Prediction (center)"); axes[2].axis("off")
            axes[3].imshow(mip_img, cmap="gray")
            axes[3].imshow(_color_mask(mip_pred.astype(np.int64), LABEL_COLORS), alpha=0.85, interpolation="none")
            axes[3].set_title("Prediction MIP" if x_item.ndim == 4 else "Prediction (2D)"); axes[3].axis("off")
            axes[4].imshow(center_img, cmap="gray")
            alpha_vis = np.clip(alpha_center, 0.35, 1.0).astype(np.float32)
            axes[4].imshow(_color_mask(center_pred.astype(np.int64), LABEL_COLORS), alpha=alpha_vis*0.8, interpolation="none")
            axes[4].set_title("Probability Overlay"); axes[4].axis("off")
            legend_elems = [
                Patch(facecolor=np.array(col, dtype=float)/255.0, edgecolor="k", label=global_label_names[c])
                for c, col in sorted(LABEL_COLORS.items(), key=lambda kv: kv[0])
            ]
            fig.subplots_adjust(right=0.82, bottom=0.10)
            fig.legend(handles=legend_elems, loc="center left", bbox_to_anchor=(0.84, 0.5), ncol=1, frameon=False)
            out_png = save_dir / f"overlay_b{b}.png"
            self._save_png_atomic(fig, out_png, dpi=self.dpi)

    # def on_train_epoch_end(self, trainer, pl_module):
    #     if not self._is_rank0(trainer): return
    #     epoch = int(trainer.current_epoch)
    #     if (epoch + 1) % self.every != 0: return
    #     from innovative3D.config import label_colors as LABEL_COLORS, global_label_names as GLOBAL_NAMES
    #     self._render_buffer(self._last_train, epoch, "train", LABEL_COLORS=LABEL_COLORS, global_label_names=GLOBAL_NAMES)
    #     self._last_train = None

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     if not self._is_rank0(trainer): return
    #     epoch = int(trainer.current_epoch)
    #     if (epoch + 1) % self.every != 0: return
    #     from innovative3D.config import label_colors as LABEL_COLORS, global_label_names as GLOBAL_NAMES
    #     self._render_buffer(self._last_val, epoch, "val", LABEL_COLORS=LABEL_COLORS, global_label_names=GLOBAL_NAMES)
    #     self._last_val = None

    def on_train_epoch_end(self, trainer, pl_module):
        if SKIP_VIZ:
            return
        if not self._is_rank0(trainer):
            return
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.every != 0:
            return
        from innovative3D.config import label_colors as LABEL_COLORS, global_label_names as GLOBAL_NAMES
        self._render_buffer(
            self._last_train, epoch, "train",
            LABEL_COLORS=LABEL_COLORS, global_label_names=GLOBAL_NAMES
        )
        self._last_train = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if SKIP_VIZ:
            return
        if not self._is_rank0(trainer):
            return
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.every != 0:
            return
        from innovative3D.config import label_colors as LABEL_COLORS, global_label_names as GLOBAL_NAMES
        self._render_buffer(
            self._last_val, epoch, "val",
            LABEL_COLORS=LABEL_COLORS, global_label_names=GLOBAL_NAMES
        )
        self._last_val = None

# ── Minimal train/val CSV logger ──────────────────────────────────────────────
class TrainValCSVLogger(Logger):
    def __init__(self, save_dir: str, name: str = "logs", filename: str = "metrics.csv"):
        super().__init__()
        self.root = Path(save_dir) / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / filename
        self.fieldnames = None
        self.last_epoch = {"train": -1, "val": -1}
        self._bootstrap()
    @property
    def name(self): return "trainval_csv"
    @property
    def version(self): return ""
    @property
    def log_dir(self): return str(self.root)
    def log_hyperparams(self, params): pass
    def finalize(self, status): pass
    def _bootstrap(self):
        if not self.file.exists(): return
        try:
            with self.file.open("r", newline="") as f:
                r = csv.DictReader(f)
                self.fieldnames = r.fieldnames
                for row in r:
                    p = row.get("phase", ""); e = row.get("epoch", "")
                    if p in self.last_epoch:
                        try: self.last_epoch[p] = max(self.last_epoch[p], int(float(e)))
                        except Exception: pass
        except Exception: pass
    def _phase_of(self, metrics: dict):
        has_train = any(k.startswith("train_") for k in metrics)
        has_val   = any(k.startswith("val_") for k in metrics)
        if has_train and not has_val: return "train"
        if has_val   and not has_train: return "val"
        return None
    def _filter(self, metrics: dict):
        keep = {"epoch", "step"}
        return {k: v for k, v in metrics.items()
                if k in keep or k.startswith("train_") or k.startswith("val_")}
    @rank_zero_only
    def log_metrics(self, metrics: dict, step=None):
        phase = self._phase_of(metrics)
        if phase is None: return
        m = self._filter(metrics)
        if "epoch" not in m or m["epoch"] in ("", None): return
        try:
            epoch = int(float(m["epoch"]))
        except Exception:
            return
        if epoch <= self.last_epoch[phase]: return
        m["phase"] = phase
        base = ["epoch", "phase"]
        if "step" in m: base.append("step")
        ordered = base + sorted(k for k in m if k not in base)
        if self.fieldnames is None:
            self.fieldnames = ordered
            write_header = not self.file.exists() or self.file.stat().st_size == 0
            with self.file.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                if write_header: w.writeheader()
                w.writerow({k: m.get(k, "") for k in self.fieldnames})
        else:
            missing = [k for k in ordered if k not in self.fieldnames]
            if missing:
                self.fieldnames += missing
                old_rows = []
                if self.file.exists() and self.file.stat().st_size > 0:
                    with self.file.open("r", newline="") as f:
                        r = csv.DictReader(f)
                        for row in r:
                            old_rows.append(row)
                with self.file.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=self.fieldnames)
                    w.writeheader()
                    for row in old_rows:
                        w.writerow({k: row.get(k, "") for k in self.fieldnames})
            with self.file.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                    {k: m.get(k, "") for k in self.fieldnames}
                )
        self.last_epoch[phase] = epoch

# ── builder that handles class vs factory ─────────────────────────────────────
def _build_lit(builder):
    """
    Accepts either:
      - a Lightning class (we try common ctor signatures), or
      - a zero-arg factory/callable returning a LightningModule.
    """
    if callable(builder) and not isclass(builder):
        # factory / lambda → just call it
        return builder()
    # Class: try a few signatures
    for kwargs in (
        {"num_classes": NUM_CLASSES, "lr": BEST_LR},
        {"num_classes": NUM_CLASSES},
        {}
    ):
        try:
            return builder(**kwargs)
        except TypeError:
            continue
    # If we’re here, we couldn’t construct it
    raise TypeError(f"Could not instantiate {builder} with tried signatures.")
def _save_compute_readout(core_module: torch.nn.Module,
                          num_frames: int,
                          H: int,
                          W: int,
                          out_path: Path):
    """
    Robust compute profiler:
      - Auto-detects 2D vs 3D via Conv layer
      - Handles SwinUNETR padding (multiple of 32)
      - Uses THOP if available, else fvcore
      - Never crashes training
    """

    import torch.nn as nn

    # safe CPU copy
    m = copy.deepcopy(core_module).to("cpu").eval()

    # --------------------------------------------------
    # 1️⃣ Detect 2D vs 3D automatically
    # --------------------------------------------------
    is_3d = False
    for module in m.modules():
        if isinstance(module, nn.Conv3d):
            is_3d = True
            break
        if isinstance(module, nn.Conv2d):
            is_3d = False
            break

    if is_3d:
        in_shape = (1, 1, num_frames, H, W)
    else:
        in_shape = (1, num_frames, H, W)

    cls_name = m.__class__.__name__
    use_padded_shape = ("SwinUNETR" in cls_name)

    # --------------------------------------------------
    # 2️⃣ SwinUNETR padding logic (only if 3D)
    # --------------------------------------------------
    compute_shape = list(in_shape)
    scale_factor = 1.0

    if use_padded_shape and is_3d:
        spatial_orig = list(in_shape[2:])  # D,H,W

        spatial_padded = []
        for d in spatial_orig:
            if d % 32 == 0:
                spatial_padded.append(d)
            else:
                spatial_padded.append(d + (32 - d % 32))

        compute_shape = list(in_shape[:2]) + spatial_padded

        # compute scaling
        orig_voxels = 1
        for d in spatial_orig:
            orig_voxels *= d

        padded_voxels = 1
        for d in spatial_padded:
            padded_voxels *= d

        if padded_voxels > 0:
            scale_factor = float(orig_voxels) / float(padded_voxels)

    # --------------------------------------------------
    # 3️⃣ Dummy input
    # --------------------------------------------------
    x = torch.randn(*compute_shape)

    params = sum(p.numel() for p in m.parameters())
    line = None

    try:
        # --------------------------------------------------
        # THOP (preferred)
        # --------------------------------------------------
        if _HAS_THOP:
            macs, _ = profile(m, inputs=(x,), verbose=False)
            macs = macs * scale_factor
            flops = macs * 2
            line = (
                f"[COMPUTE] Params: {params/1e6:.2f}M | "
                f"MACs: {macs/1e9:.2f}G  (≈ FLOPs {flops/1e9:.2f}G)"
            )

        # --------------------------------------------------
        # FVCORE fallback
        # --------------------------------------------------
        elif _HAS_FVCORE:
            from fvcore.nn import FlopCountAnalysis

            flops_obj = FlopCountAnalysis(m, x)
            flops_obj = flops_obj.unsupported_ops_warnings(False)
            flops_obj = flops_obj.uncalled_modules_warnings(False)

            flops = flops_obj.total()
            flops = flops * scale_factor

            line = (
                f"[COMPUTE] Params: {params/1e6:.2f}M | "
                f"FLOPs: {flops/1e9:.2f}G"
            )

        # --------------------------------------------------
        # No profiler installed
        # --------------------------------------------------
        else:
            line = (
                f"[COMPUTE] Params: {params/1e6:.2f}M "
                f"(install `thop` or `fvcore` for FLOPs/MACs)"
            )

    except Exception as e:
        line = f"[COMPUTE] skipped: {e}"

    r0print(line)

    try:
        out_path.write_text(line + "\n")
    except Exception:
        pass
# ── main train/test loop ──────────────────────────────────────────────────────
def train_and_log(model_name, BuilderOrClass, DataMod, _base_unused: str, seed: int):
    ckpt_folder = Path(CHECKPOINT_DIR) / model_name / f"seed{seed}"
    ckpt_folder.mkdir(parents=True, exist_ok=True)

    for p in ckpt_folder.glob("last-v*.ckpt"):
        try: p.unlink()
        except Exception: pass

    r0print(f"\n===== {model_name} | seed={seed} =====")
    r0print(f"[FAST] FAST_TEST={int(FAST_TEST)} | limit={FAST_TEST_LIMIT} | skip_viz={int(FAST_SKIP_VIZ)} | skip_details={int(FAST_SKIP_TEST_DETAILS)} | simple_metrics={int(FAST_SIMPLE_METRICS)}")
    seed_everything(seed, workers=True)

    dm = DataMod(trainval_sets, batch_size=BATCH_SIZE, num_frames=NUM_FRAMES)

    # NEW: build model from class OR factory
    model = _build_lit(BuilderOrClass)
    # --- compute Params/MACs/FLOPs (once) and save next to the seed folder ---
    try:
        nn_core = getattr(model, "model", model)

        _save_compute_readout(
            nn_core,
            NUM_FRAMES,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            Path(CHECKPOINT_DIR) / model_name / f"seed{seed}" / "model_compute.txt"
        )

    except Exception as e:
        r0print(f"[COMPUTE] skipped: {e}")
    logger = TrainValCSVLogger(save_dir=str(ckpt_folder), name="logs")  # writes logs/metrics.csv

    ckpt_last = ModelCheckpoint(
        dirpath=str(ckpt_folder),
        save_last=True,
        save_top_k=0,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
    )
    ckpt_best = ModelCheckpoint(
        dirpath=str(ckpt_folder),
        monitor="val_macro_dice",
        mode="max",
        # monitor="val_loss", mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_macro_dice:.4f}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
        every_n_epochs=1,
    )


    early_cb = EarlyStopping(
    monitor="val_macro_dice",   # or keep this if you log it (see note below)
    mode="max",
    patience=12,                # tweak to taste
    min_delta=1e-3,             # require a real improvement
    check_on_train_epoch_end=False,
    verbose=True,
    )



    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    callbacks = [ckpt_last, ckpt_best, early_cb]


    # Attach visualization only on single-GPU to avoid DDP issues
    # Attach visualization (rank-0 safe) even on multi-GPU if forced
    one_gpu = (devices == 1) or (isinstance(devices, (list, tuple)) and len(devices) == 1)
    attach_viz = (not FAST_SKIP_VIZ) and (one_gpu or FORCE_VIZ_MULTI_GPU)

    VIZ_EVERY = int(os.getenv("VIZ_EVERY", "20"))  # how often to render

    if attach_viz:
        viz_cb = VisualizeEveryNEpochsBuffered(
            out_root=ckpt_folder,
            every_n_epochs=VIZ_EVERY,
            max_items=2
        )
        callbacks.append(viz_cb)
    else:
        reason = "disabled (FAST_SKIP_VIZ=1)" if FAST_SKIP_VIZ else ("multi-GPU and FORCE_VIZ_MULTI_GPU=0" if not one_gpu else "unknown")
        r0print(f"[FAST] Skipping visualization callback: {reason}")

    trainer = Trainer(
    max_epochs=FINAL_EPOCHS,
    accelerator=accelerator,
    devices=1,
    logger=logger,
    callbacks=callbacks,
    deterministic="warn",
    log_every_n_steps=1,
    num_sanity_val_steps=0,
    enable_progress_bar=True,
    # strategy=("ddp_find_unused_parameters_true" if USE_DDP_FIND_UNUSED and devices > 1
    # #           else ("ddp_find_unused_parameters_false" if devices > 1 else None)),
    # strategy=("ddp_find_unused_parameters_true" if devices > 1 else None),
    limit_test_batches=(FAST_TEST_LIMIT if FAST_TEST else 1.0),

    # ✅ add this line:
    reload_dataloaders_every_n_epochs=1,
)


    try: trainer.fit_loop.max_epochs = FINAL_EPOCHS
    except Exception: pass

    ckpt_path = _resolve_resume_ckpt(ckpt_folder)
    if ckpt_path:
        r0print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
    else:
        r0print("[INFO] No checkpoint found, starting fresh.")

    dm.setup("fit")
    trainer.fit(model, dm, ckpt_path=ckpt_path)

    dm.setup("test")

    # SKIP Lightning's test loop (DDP can crash on rank>0). We'll run our own eval below.
    # _ = trainer.test(model, dm, ckpt_path=None)

    # --- sync all ranks, then only rank 0 does custom eval & file I/O ---
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass

    if getattr(trainer, "is_global_zero", True):
        test_csv = Path(ckpt_folder) / "logs" / "test_metrics.csv"
        write_test_metrics_csv_from_pass(
            model,
            dm,
            out_csv_path=test_csv,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_ID,
            per_case=False,
            is_3d=None,
            max_batches=(FAST_TEST_LIMIT if FAST_TEST else None),
            fast_simple=FAST_SIMPLE_METRICS,
        )

        details_csv = ckpt_folder / "test_details.csv"
        if FAST_SKIP_TEST_DETAILS:
            r0print("[FAST] Skipping test_details.csv and summary.csv")
        else:
            try:
                ndim, _ = _peek_test_shape(dm)
                if ndim == 4:
                    write_test_details_planar(model, dm, details_csv, num_classes=NUM_CLASSES, ignore_index=IGNORE_ID)
                elif ndim == 5:
                    write_test_details_3d(model, dm, details_csv, num_classes=NUM_CLASSES, ignore_index=IGNORE_ID)
            except Exception as e:
                r0print(f"[WARN] details export skipped/failed: {e}")
            write_summary_csv(details_csv)
        # --- NEW: write scan/region-level labels derived from voxel predictions ---
        # try:
        #     scan_csv = Path(ckpt_folder) / "scan_labels.csv"
        #     write_scan_labels_csv(
        #         model,
        #         dm,
        #         out_csv_path=scan_csv,
        #         num_classes=NUM_CLASSES,
        #         min_vox=SCAN_MIN_VOX,
        #     )
        #     r0print(f"[TEST] Wrote scan-level labels to {scan_csv}")
        # except Exception as e:
        #     r0print(f"[WARN] scan label export skipped/failed: {e}")

        # Console summary
        try:
            df = pd.read_csv(test_csv)
            if "test_macro_dice" in df.columns and not pd.isna(df.iloc[-1]["test_macro_dice"]):
                score = float(df.iloc[-1]["test_macro_dice"])
                r0print(f"[TEST] {model_name} (seed={seed}) → test_macro_dice={score:.4f}")
                return f"{score:.4f}"
        except Exception:
            pass

        r0print(f"[TEST] {model_name} (seed={seed}) → DONE (see {test_csv})")
    return "DONE"

def main():
    # CLI convenience: --fast toggles FAST_TEST=1 for this run
    global FAST_TEST, FAST_TEST_LIMIT, FAST_SKIP_VIZ, FAST_SKIP_TEST_DETAILS, FAST_SIMPLE_METRICS
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Enable fast test path (equivalent to FAST_TEST=1)")
    parser.add_argument("--fast-test-limit", type=int, default=None, help="Limit test batches in fast mode (default 2)")
    parser.add_argument("--fast-skip-viz", action="store_true", help="Skip visualization callback in fast mode")
    parser.add_argument("--fast-skip-details", action="store_true", help="Skip test_details.csv and summary.csv in fast mode")
    parser.add_argument("--fast-simple-metrics", action="store_true", help="Lightweight test metrics (no PR/ROC/IoU)")
    args = parser.parse_args()

    if args.fast:
        FAST_TEST = True
        if args.fast_test_limit is not None:
            FAST_TEST_LIMIT = int(args.fast_test_limit)
        if args.fast_skip_viz:
            FAST_SKIP_VIZ = True
        if args.fast_skip_details:
            FAST_SKIP_TEST_DETAILS = True
        if args.fast_simple_metrics:
            FAST_SIMPLE_METRICS = True
            
    # ---- Run profiler and exit (env toggle) ----
    if os.getenv("PROFILE_ONLY", "0").lower() in ("1", "true", "yes", "on"):
        from innovative3D.profiling import profile_all
        profile_all()
        return

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, BuilderOrClass, DataMod, base in VARIANTS:
        for sd in SEEDS:
            out = train_and_log(model_name, BuilderOrClass, DataMod, base, sd)
            rows.append({"model": model_name, "seed": sd, "test_macro_dice": out})
    master_csv = Path(CHECKPOINT_DIR) / "all_results.csv"
    pd.DataFrame(rows).to_csv(master_csv, index=False)
    r0print(f"\nSaved results to {master_csv}")

if __name__ == "__main__":
    main()
