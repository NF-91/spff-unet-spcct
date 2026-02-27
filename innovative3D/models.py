# models.py
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import math
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
from torchmetrics import MeanMetric
from typing import List, Tuple, Optional, Iterable

# Optional profiling (THOP)
try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, jaccard_score, precision_score
)

# Project config & helpers
from innovative3D.config import (
    NUM_CLASSES, BEST_LR, IGNORE_INDEX, LOSS_NAME, FOCAL_ALPHA, FOCAL_GAMMA,
    GRAD_WEIGHT, NUM_FRAMES
)
from innovative3D.helpers import (
    per_class_metrics_3d, per_class_metrics_2d,
    ce_plus_macro_dice_loss, focal_plus_gradient_loss,
    LOSS_REGISTRY
)

# -----------------------------------------------------------------------------
# Global / env flags
# -----------------------------------------------------------------------------
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
LOG_PER_CLASS = os.getenv("LOG_PER_CLASS", "1") == "1"

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _freeze_module(m: Optional[nn.Module]):
    if m is None: return
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

def _maybe(m, cond: bool):
    return m if cond else None

def _pick_first_if_seq(x):
    return x[0] if isinstance(x, (list, tuple)) else x

def _canonicalize_targets_2d(lbls):
    lbls = _pick_first_if_seq(lbls)
    if not torch.is_tensor(lbls):
        lbls = torch.as_tensor(lbls)
    if lbls.ndim == 4:     # (B,F,H,W) -> merge frames (OR take max-label over frames)
        lbls = lbls.max(dim=1).values
    elif lbls.ndim == 2:   # (H,W) -> (1,H,W)
        lbls = lbls.unsqueeze(0)
    return lbls.long()

def _canonicalize_targets_3d(lbls):
    """
    Normalize labels to (B,F,H,W) long.
    Accepts (B,1,F,H,W)->(B,F,H,W), (B,F,H,W,1)->(B,F,H,W), (F,H,W)->(1,F,H,W).
    """
    lbls = _pick_first_if_seq(lbls)
    if not torch.is_tensor(lbls):
        lbls = torch.as_tensor(lbls)
    if lbls.ndim == 5 and lbls.size(1) == 1:
        lbls = lbls[:, 0]
    if lbls.ndim == 5 and lbls.size(-1) == 1:
        lbls = lbls[..., 0]
    if lbls.ndim == 3:
        lbls = lbls.unsqueeze(0)
    assert lbls.ndim == 4, f"Need (B,F,H,W) labels, got {tuple(lbls.shape)}"
    return lbls.long()

def _validate_or_fix_labels(labels: torch.Tensor, num_classes: int,
                            ignore_index: int, fix: bool = False):
    """
    Ensure labels ∈ {0..num_classes-1} ∪ {ignore_index}.
    If fix=True, invalid values are set to ignore_index (on a clone).
    Returns (labels, bad_values).
    """
    lab = labels
    bad_mask = (lab != ignore_index) & ((lab < 0) | (lab >= num_classes))
    if not bad_mask.any():
        return lab, torch.tensor([], device=lab.device, dtype=lab.dtype)
    bad_vals = torch.unique(lab[bad_mask])
    if fix:
        lab = lab.clone()
        lab[bad_mask] = ignore_index
    return lab, bad_vals

# -----------------------------------------------------------------------------
# Pad / crop helpers (3D)
# -----------------------------------------------------------------------------
def _next_mult(n: int, m: int = 16) -> int:
    return ((n + m - 1) // m) * m

def _pad_to_mult_3d(x: torch.Tensor, m: int = 16):
    """Pad [B,C,D,H,W] so D/H/W are multiples of m. Return (x_pad, (D,H,W)) or (x, None)."""
    if x.ndim != 5: raise ValueError(f"expect [B,C,D,H,W], got {tuple(x.shape)}")
    B, C, D, H, W = x.shape
    Dn, Hn, Wn = _next_mult(D, m), _next_mult(H, m), _next_mult(W, m)
    pd, ph, pw = Dn - D, Hn - H, Wn - W
    if not (pd or ph or pw): return x, None
    pd_l, pd_r = pd // 2, pd - pd // 2
    ph_l, ph_r = ph // 2, ph - ph // 2
    pw_l, pw_r = pw // 2, pw - pw // 2
    x = F.pad(x, (pw_l, pw_r, ph_l, ph_r, pd_l, pd_r), mode="replicate")
    return x, (D, H, W)

def _center_crop_to_3d(x: torch.Tensor, orig_dhw: Optional[tuple]):
    if orig_dhw is None: return x
    D, H, W = orig_dhw
    _, _, Dn, Hn, Wn = x.shape
    sd, sh, sw = (Dn - D) // 2, (Hn - H) // 2, (Wn - W) // 2
    return x[:, :, sd:sd+D, sh:sh+H, sw:sw+W]

def _pad_to_mult16(x: torch.Tensor, multiple: int = 16):  # alias
    return _pad_to_mult_3d(x, m=int(multiple))

def _center_crop(x: torch.Tensor, orig_dhw: Optional[tuple]):  # alias
    return _center_crop_to_3d(x, orig_dhw)

_pad_to_mult16_3d = _pad_to_mult16
_center_crop_3d   = _center_crop

# HW-only pad/crop while preserving depth (F)
def _pad_to_mult16_hw(x: torch.Tensor, multiple: int = 16):
    assert x.ndim == 5, f"expected [B,C,F,H,W], got {tuple(x.shape)}"
    B, C, Fd, H, W = x.shape
    ph = (multiple - (H % multiple)) % multiple
    pw = (multiple - (W % multiple)) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph, 0, 0), mode="replicate")
    return x, (Fd, H, W)

def _crop_to_hw(x: torch.Tensor, orig_fhw: tuple[int,int,int]):
    F0, H0, W0 = orig_fhw
    return x[..., :F0, :H0, :W0]

# Depth-only resize helpers (for depth adapters)
def _resize_depth_like(x: torch.Tensor, target_depth: int):
    # x: [B,C,D,H,W] -> resize D to target_depth (keep H/W)
    B, C, D, H, W = x.shape
    if D == target_depth: return x
    return F.interpolate(x, size=(target_depth, H, W), mode="trilinear", align_corners=False)

def _resize_logits_depth_like(x: torch.Tensor, D0: int):
    # x: [B,C,D*,H,W] -> resize D* -> D0
    B, C, D, H, W = x.shape
    if D == D0: return x
    return F.interpolate(x, size=(D0, H, W), mode="trilinear", align_corners=False)

# -----------------------------------------------------------------------------
# Conv / Norm / Act convenience
# -----------------------------------------------------------------------------
def _norm3d(c: int, kind: str = "instance") -> nn.Module:
    k = (kind or "instance").lower()
    if k.startswith("inst"): return nn.InstanceNorm3d(c, affine=True, eps=1e-5)
    if k.startswith("batch"): return nn.BatchNorm3d(c)
    if k.startswith("group"): return nn.GroupNorm(num_groups=max(1, c // 8), num_channels=c)
    return nn.Identity()

def _act(kind: str = "lrelu") -> nn.Module:
    k = (kind or "lrelu").lower()
    return (
        nn.LeakyReLU(1e-2, inplace=True) if k.startswith("lrel") else
        nn.ReLU(inplace=True)            if k.startswith("relu")  else
        nn.GELU()
    )

class ConvBNAct3d(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, norm="instance", act="lrelu", bias=False):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=bias)
        self.norm = _norm3d(cout, norm)
        self.act  = _act(act)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# -----------------------------------------------------------------------------
# Lightweight profilers
# -----------------------------------------------------------------------------
def _profile_macs_params(model, shape=(1, 1, 5, 512, 512)):
    try:
        from thop import profile
    except Exception:
        return None
    device = next(model.parameters()).device
    dummy = torch.zeros(shape, device=device)
    was_train = model.training
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy,), verbose=False)
    if was_train: model.train()
    return macs, params

def _benchmark_latency(model, shape=(1, 1, 5, 512, 512), warmup=5, iters=20):
    import time
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    was_train = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(warmup): _ = model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): _ = model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    if was_train: model.train()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms

# -----------------------------------------------------------------------------
# Losses / metrics helpers
# -----------------------------------------------------------------------------
def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = F.one_hot(torch.clamp(labels, 0, num_classes - 1), num_classes=num_classes)
    return oh.permute(0, 4, 1, 2, 3).float()

def dice_per_class_from_logits(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int,
    ignore_index: Optional[int] = 255, include_bg: bool = False, eps: float = 1e-6
) -> torch.Tensor:
    # logits: [B,C,D,H,W], target: [B,D,H,W]
    probs = F.softmax(logits, dim=1)
    if ignore_index is not None:
        mask = (target != ignore_index).unsqueeze(1)  # [B,1,D,H,W]
        probs = probs * mask
        tgt = torch.where(target == ignore_index, torch.zeros_like(target), target)
    else:
        tgt = target
    y = _one_hot(tgt, num_classes=num_classes)
    if ignore_index is not None:
        y = y * mask
    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * y, dims)
    denom = torch.sum(probs, dims) + torch.sum(y, dims)
    dice = (2 * inter + eps) / (denom + eps)  # [C]
    if not include_bg and logits.shape[1] > 1:
        dice = dice[1:]
    return dice

def soft_dice_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    include_background: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert logits.ndim == 5 and labels.ndim == 4
    valid_mask = (labels != ignore_index).unsqueeze(1).float()  # (B,1,F,H,W)
    probs = torch.softmax(logits, dim=1) * valid_mask
    labels_safe = labels.clone()
    labels_safe[labels_safe == ignore_index] = 0
    target = F.one_hot(labels_safe, num_classes=num_classes).permute(0,4,1,2,3).float() * valid_mask
    if not include_background and num_classes > 1:
        probs  = probs[:, 1:]
        target = target[:, 1:]
    dims = (0, 2, 3, 4)
    inter = (probs * target).sum(dims)
    den   = (probs * probs).sum(dims) + (target * target).sum(dims)
    dice  = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()

def dice_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    include_background: bool = False,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels.long(), ignore_index=ignore_index)
    dice = soft_dice_loss_from_logits(
        logits, labels, num_classes, ignore_index=ignore_index, include_background=include_background
    )
    return ce_weight * ce + dice_weight * dice

def dice_ce_loss_with_metrics(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int,
    ignore_index: Optional[int] = 255, include_bg_in_dice: bool = False,
    ce_weight: float = 1.0, dice_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dice_vec = dice_per_class_from_logits(logits, target, num_classes, ignore_index, include_bg_in_dice)
    dice_loss = 1.0 - dice_vec.mean()
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index) if ignore_index is not None \
         else F.cross_entropy(logits, target)
    loss = dice_weight * dice_loss + ce_weight * ce
    return loss, dice_vec.mean(), ce

# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------
class ResidualConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.skip  = nn.Conv3d(in_channels, out_channels, 1, bias=False) \
            if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        return self.relu2(out)

class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1,2,4,8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.proj = nn.Sequential(
            nn.Conv3d(len(dilations)*out_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        feats = [conv(x) for conv in self.branches]
        return self.proj(torch.cat(feats, dim=1))

class SEBlock3D(nn.Module):
    def __init__(self, channels, r: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, max(1, channels//r), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, channels//r), channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class SpectralGate(nn.Module):
    """ Gating along spectral (F) via (3,1,1) convs. """
    def __init__(self, channels, hidden=16):
        super().__init__()
        h = max(4, min(hidden, channels))
        self.conv1 = nn.Conv3d(channels, h, kernel_size=(3,1,1), padding=(1,0,0), bias=True)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(h, channels, kernel_size=(3,1,1), padding=(1,0,0), bias=True)
        self.sig   = nn.Sigmoid()
    def forward(self, x):
        s = x.mean(dim=(-1, -2), keepdim=True)          # (B,C,F,1,1)
        g = self.sig(self.conv2(self.act(self.conv1(s))))
        return x * g

class SPConvBlock(nn.Module):
    """Two spatial convs (1,3,3) + optional spectral (3,1,1) + SpectralGate."""
    def __init__(self, cin, cout, norm="instance", act="lrelu",
                 mix_spectral=True, use_gate=True):
        super().__init__()
        self.conv1 = ConvBNAct3d(cin,  cout, k=(1,3,3), s=1, p=(0,1,1), norm=norm, act=act)
        self.conv2 = ConvBNAct3d(cout, cout, k=(1,3,3), s=1, p=(0,1,1), norm=norm, act=act)
        self.mix_spectral = bool(mix_spectral)
        if self.mix_spectral:
            self.mix = ConvBNAct3d(cout, cout, k=(3,1,1), s=1, p=(1,0,0), norm=norm, act=act)
        self.use_gate = bool(use_gate)
        if self.use_gate:
            self.gate = SpectralGate(cout)
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        if self.mix_spectral: x = self.mix(x)
        if self.use_gate:     x = self.gate(x)
        return x

class SpectralTemporalMixer2D(nn.Module):
    """
    Input:  (B, F, H, W) where F = frames/spectral bins
    Output: (B, Cmix, H, W)
    """
    def __init__(self, in_frames: int, out_channels: int = 64, reduction: int = 4):
        super().__init__()
        self.mix = nn.Conv2d(in_frames, out_channels, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, out_channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv1d(out_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mix(x)                         # (B,Cmix,H,W)
        w = self.gap(z).flatten(2)              # (B,Cmix,1)
        w = self.fc(w).unsqueeze(-1)            # (B,Cmix,1,1)
        return z * w

class SE3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class SpatialAttention3D(nn.Module):
    """CBAM-style spatial attention for 3D tensors."""
    def __init__(self, kernel_size=(3, 7, 7)):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, maxv], dim=1)
        attn = self.sigmoid(self.conv(a))
        return x * attn

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=(1,2,2), stride=(1,2,2))
        self.conv = DoubleConv3D(in_ch, out_ch)
        self.attn = SE3D(out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        diff = [skip.shape[i] - x.shape[i] for i in range(2,5)]
        if any(d != 0 for d in diff):
            x = F.pad(x, (0, max(0,diff[2]), 0, max(0,diff[1]), 0, max(0,diff[0])))
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return self.attn(x)

# -----------------------------------------------------------------------------
# Base Lightning with unified logging
# -----------------------------------------------------------------------------
class BaseLitModel(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, lr=BEST_LR, is_3d=True, **kwargs):
        super().__init__()
        self.is_3d = bool(is_3d)
        self.save_hyperparameters({"num_classes": num_classes, "lr": float(lr), "is_3d": bool(is_3d), **kwargs})
        self.lambda_esc    = float(kwargs.get("lambda_esc", 0.0))
        self.lambda_smooth = float(kwargs.get("lambda_smooth", 0.0))

    def _normalize_input(self, x): return _pick_first_if_seq(x)
    def forward(self, x): return self.model(self._normalize_input(x))
    def compute_loss(self, logits, labels):
        return ce_plus_macro_dice_loss(logits, labels, self.hparams.num_classes, ignore_index=IGNORE_INDEX)

    def _shared_step(self, batch, prefix):
        imgs, lbls = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        imgs = _pick_first_if_seq(imgs); lbls = _pick_first_if_seq(lbls)
        if not self.is_3d: lbls = _canonicalize_targets_2d(lbls)

        logits = self(imgs); lbls = lbls.to(logits.device).long()
        loss   = self.compute_loss(logits, lbls)
        fn = per_class_metrics_3d if self.is_3d else per_class_metrics_2d
        (dice_list, sens_list, spec_list,
         macro_dice, macro_sens, macro_spec,
         micro_dice, micro_sens, micro_spec) = fn(
            logits, lbls, self.hparams.num_classes, ignore_index=IGNORE_INDEX
        )

        self.log(f'{prefix}_loss',       loss,       on_step=False, on_epoch=True,
                 prog_bar=(prefix=='train'), sync_dist=True)
        self.log(f'{prefix}_macro_dice', macro_dice, on_step=False, on_epoch=True,
                 prog_bar=(prefix!='test'),  sync_dist=True)
        self.log(f'{prefix}_micro_dice', micro_dice, on_step=False, on_epoch=True,
                 prog_bar=True,              sync_dist=True)
        self.log(f'{prefix}_macro_sens', macro_sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_macro_spec', macro_spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_micro_sens', micro_sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_micro_spec', micro_spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for i, (d, s, sp) in enumerate(zip(dice_list, sens_list, spec_list)):
            self.log(f'{prefix}_dice_class_{i}', d,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'{prefix}_sens_class_{i}', s,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'{prefix}_spec_class_{i}', sp, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # Extra test-only metrics (PR/ROC/IoU/Precision) retained from your original code.
        if prefix == 'test':
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs_np = probs.detach().cpu().numpy()
                lbls_np  = lbls.detach().cpu().numpy()
                pred_np  = preds.detach().cpu().numpy()

                pr_aucs, ious, precisions, roc_aucs = [], [], [], []
                y_true_onehot = []; y_score_flat = []; y_pred_flat = []

                for c in range(self.hparams.num_classes):
                    y_true  = (lbls_np == c).ravel().astype(int)
                    y_score = probs_np[:, c].ravel()
                    y_pred  = (pred_np == c).ravel().astype(int)

                    y_true_onehot.append(y_true)
                    y_score_flat.append(y_score)
                    y_pred_flat.append(y_pred)

                    if y_true.sum() > 0 and y_true.sum() < len(y_true):
                        p, r, _ = precision_recall_curve(y_true, y_score)
                        pr_aucs.append(auc(r, p))
                        roc_aucs.append(roc_auc_score(y_true, y_score))
                    else:
                        pr_aucs.append(float('nan')); roc_aucs.append(float('nan'))

                    try:
                        ious.append(jaccard_score(y_true, y_pred) if (y_true.sum() > 0 or y_pred.sum() > 0) else float('nan'))
                    except Exception:
                        ious.append(float('nan'))
                    try:
                        precisions.append(precision_score(y_true, y_pred, zero_division=0) if (y_pred.sum() + y_true.sum()) > 0 else float('nan'))
                    except Exception:
                        precisions.append(float('nan'))

                def _macro_ignore_nan(x): return float(np.nanmean(x[1:])) if len(x) > 1 else float(np.nanmean(x))
                macro_pr_auc    = _macro_ignore_nan(pr_aucs)
                macro_roc_auc   = _macro_ignore_nan(roc_aucs)
                macro_iou       = _macro_ignore_nan(ious)
                macro_precision = _macro_ignore_nan(precisions)

                if len(y_true_onehot) > 1:
                    y_true_all  = np.concatenate(y_true_onehot[1:], axis=0)
                    y_score_all = np.concatenate(y_score_flat[1:], axis=0)
                    y_pred_all  = np.concatenate(y_pred_flat[1:], axis=0)
                else:
                    y_true_all = y_score_all = y_pred_all = np.array([])

                try:
                    if y_true_all.size and 0 < y_true_all.sum() < len(y_true_all):
                        p_fg, r_fg, _ = precision_recall_curve(y_true_all, y_score_all)
                        pr_auc_micro = auc(r_fg, p_fg)
                        roc_auc_micro = roc_auc_score(y_true_all, y_score_all)
                    else:
                        pr_auc_micro = float('nan'); roc_auc_micro = float('nan')
                    iou_micro = jaccard_score(y_true_all, y_pred_all) if y_true_all.size else float('nan')
                    precision_micro = precision_score(y_true_all, y_pred_all, zero_division=0) if y_true_all.size else float('nan')
                except Exception:
                    pr_auc_micro = roc_auc_micro = iou_micro = precision_micro = float('nan')

                for i, (pr, io, prec, roc) in enumerate(zip(pr_aucs, ious, precisions, roc_aucs)):
                    self.log(f'{prefix}_pr_auc_class_{i}', pr,   on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                    self.log(f'{prefix}_iou_class_{i}',    io,   on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                    self.log(f'{prefix}_precision_class_{i}', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                    self.log(f'{prefix}_roc_auc_class_{i}', roc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

                self.log(f'{prefix}_pr_auc_macro',    macro_pr_auc,    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f'{prefix}_roc_auc_macro',   macro_roc_auc,   on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f'{prefix}_iou_macro',       macro_iou,       on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f'{prefix}_precision_macro', macro_precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f'{prefix}_pr_auc_micro',    pr_auc_micro,    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f'{prefix}_roc_auc_micro',   roc_auc_micro,   on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f'{prefix}_iou_micro',       iou_micro,       on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f'{prefix}_precision_micro', precision_micro, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0): return self._shared_step(batch, 'train')
    def validation_step(self, batch, batch_idx, dataloader_idx=0): return {'val_loss': self._shared_step(batch, 'val')}
    def test_step(self, batch, batch_idx): return self._shared_step(batch, 'test')
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'monitor': 'val_macro_dice'}}


# -----------------------------------------------------------------------------
# UNet3D Spectral Core + variants (SPCT)
# -----------------------------------------------------------------------------
class _SEChannelLite(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        h = max(4, c // r)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(c, h, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv3d(h, c, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(self.pool(x))

class _SpectralSE(nn.Module):
    def forward(self, x):
        w = x.mean(dim=(1,3,4), keepdim=True)  # [B,1,D,1,1]
        return x * torch.sigmoid(w)

def _conv3x3xk(cin, cout, ksd=1, bias=False):
    pad_d = ksd // 2
    return nn.Conv3d(cin, cout, kernel_size=(ksd,3,3), padding=(pad_d,1,1), bias=bias)

class _DoubleConvSpectral(nn.Module):
    def __init__(self, cin, cout, ksd=1, norm="instance", act="lrelu"):
        super().__init__()
        self.b1 = nn.Sequential(_conv3x3xk(cin,  cout, ksd, bias=False), _norm3d(cout, norm), _act(act))
        self.b2 = nn.Sequential(_conv3x3xk(cout, cout, ksd, bias=False), _norm3d(cout, norm), _act(act))
    def forward(self, x): return self.b2(self.b1(x))

class AttentionGate(nn.Module):
    """3D attention gate for skip connections."""
    def __init__(self, F_skip, F_g, F_int=None):
        super().__init__()
        if F_int is None: F_int = min(F_skip, F_g)
        self.W_x = nn.Conv3d(F_skip, F_int, 1, 1, 0, bias=True)
        self.W_g = nn.Conv3d(F_g,   F_int, 1, 1, 0, bias=True)
        self.psi = nn.Conv3d(F_int, 1, 1, 1, 0, bias=True)
        nn.init.constant_(self.psi.bias, 0.0)
        self.relu = nn.ReLU(inplace=True); self.sigmoid = nn.Sigmoid()
    def forward(self, x_skip, g):
        att = self.relu(self.W_x(x_skip) + self.W_g(g))
        att = self.sigmoid(self.psi(att))
        return x_skip * att

class AttentionGate3d(AttentionGate):
    def __init__(self, g_c, x_c, inter_c=None):
        # AttentionGate expects (F_skip, F_g, F_int)
        super().__init__(F_skip=x_c, F_g=g_c, F_int=inter_c)
        
class UNet3D_SpectralCore(nn.Module):
    """
    Depth-preserving UNet:
      - pooling/upsample only in (H,W) with (1,2,2)
      - spectral mixing via (ksd,3,3) kernels (ksd ∈ {1,3})
      - optional Channel-SE, Spectral-SE, SpatialAttention3D, and gated skips
    """
    def __init__(self, in_channels=1, num_classes=2, base=32, ksd=3,
                 use_se=False, use_specse=False, use_spatial=False, use_skip_gate=False,
                 norm="instance", act="lrelu"):
        super().__init__()
        f = int(base); P = (1,2,2)
        # enc
        self.enc1 = _DoubleConvSpectral(in_channels, f,     ksd, norm, act)
        self.pool1= nn.MaxPool3d(P)
        self.enc2 = _DoubleConvSpectral(f,         2*f,     ksd, norm, act)
        self.pool2= nn.MaxPool3d(P)
        self.enc3 = _DoubleConvSpectral(2*f,       4*f,     ksd, norm, act)
        self.pool3= nn.MaxPool3d(P)
        self.bott = _DoubleConvSpectral(4*f,       8*f,     ksd, norm, act)
        # dec
        self.up3  = nn.ConvTranspose3d(8*f, 4*f, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec3 = _DoubleConvSpectral(8*f, 4*f, ksd, norm, act)
        self.up2  = nn.ConvTranspose3d(4*f, 2*f, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = _DoubleConvSpectral(4*f, 2*f, ksd, norm, act)
        self.up1  = nn.ConvTranspose3d(2*f,   f, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec1 = _DoubleConvSpectral(2*f,   f, ksd, norm, act)
        self.out  = nn.Conv3d(f, num_classes, 1)
        # attentions
        self.se = nn.ModuleList([_SEChannelLite(c) if use_se else nn.Identity() for c in (f, 2*f, 4*f, 8*f)])
        self.sp = nn.ModuleList([_SpectralSE() if use_specse else nn.Identity() for _ in (f, 2*f, 4*f, 8*f)])
        self.sa = nn.ModuleList([SpatialAttention3D() if use_spatial else nn.Identity() for _ in (f, 2*f, 4*f, 8*f)])
        # gated skips
        self.g3 = AttentionGate3d(4*f, 4*f, 2*f) if use_skip_gate else None
        self.g2 = AttentionGate3d(2*f, 2*f,   f) if use_skip_gate else None
        self.g1 = AttentionGate3d(  f,   f, f//2) if use_skip_gate else None

    def _post(self, x, stage):
        return self.sa[stage](self.se[stage](self.sp[stage](x)))

    @staticmethod
    def _cat(up, skip):
        if up.shape[-3:] != skip.shape[-3:]:
            up = F.interpolate(up, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x):  # (B,C,D,H,W)
        e1 = self._post(self.enc1(x), 0)
        e2 = self._post(self.enc2(self.pool1(e1)), 1)
        e3 = self._post(self.enc3(self.pool2(e2)), 2)
        b  = self._post(self.bott(self.pool3(e3)), 3)
        d3 = self.up3(b);  s3 = self.g3(d3, e3) if self.g3 is not None else e3; d3 = self.dec3(self._cat(d3, s3))
        d2 = self.up2(d3); s2 = self.g2(d2, e2) if self.g2 is not None else e2; d2 = self.dec2(self._cat(d2, s2))
        d1 = self.up1(d2); s1 = self.g1(d1, e1) if self.g1 is not None else e1; d1 = self.dec1(self._cat(d1, s1))
        return self.out(d1)

class _LitSPCT_Base(BaseLitModel):
    def __init__(self, num_classes=NUM_CLASSES, lr=BEST_LR, pad_multiple: int = 16):
        super().__init__(num_classes=num_classes, lr=lr, is_3d=True)
        self._pad_multiple = int(pad_multiple)
    def forward(self, x):
        x = _pick_first_if_seq(x)
        if x.ndim == 4: x = x.unsqueeze(1)
        x_pad, orig = _pad_to_mult16_3d(x, multiple=self._pad_multiple)
        y_pad = self.model(x_pad)
        return _center_crop_3d(y_pad, orig)


# -----------------------------------------------------------------------------
# Published baselines: Cicek 3D U-Net + depth adapter Lightning
# -----------------------------------------------------------------------------
class Cicek3DUNet(nn.Module):
    def __init__(self, num_classes: int, base: int = 32, use_bn: bool = True):
        super().__init__()
        Norm = (lambda c: nn.BatchNorm3d(c)) if use_bn else (lambda c: nn.Identity())
        def block(ci, co):
            return nn.Sequential(
                nn.Conv3d(ci, co, 3, padding=1, bias=(not use_bn)), Norm(co), nn.ReLU(inplace=True),
                nn.Conv3d(co, co, 3, padding=1, bias=(not use_bn)), Norm(co), nn.ReLU(inplace=True),
            )
        self.enc1 = block(1, base);      self.pool1 = nn.MaxPool3d(2)
        self.enc2 = block(base, base*2); self.pool2 = nn.MaxPool3d(2)
        self.enc3 = block(base*2, base*4); self.pool3 = nn.MaxPool3d(2)
        self.enc4 = block(base*4, base*8); self.pool4 = nn.MaxPool3d(2)
        self.bott = block(base*8, base*16)
        self.up4 = nn.ConvTranspose3d(base*16, base*8, 2, stride=2)
        self.dec4 = block(base*8 + base*8, base*8)
        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, stride=2)
        self.dec3 = block(base*4 + base*4, base*4)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = block(base*2 + base*2, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base,   2, stride=2)
        self.dec1 = block(base + base, base)
        self.out = nn.Conv3d(base, num_classes, 1)
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        b  = self.bott(p4)
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

class LitCicek3DUNet_DepthAdapter_Published(pl.LightningModule):
    def __init__(self, num_classes: int, target_depth: int = 16,
                 lr: float = 1e-2, momentum: float = 0.99, nesterov: bool = False, weight_decay: float = 0.0,
                 ignore_index: int | None = 255, class_weights: list[float] | None = None, voxel_weight_key: str | None = None,
                 ce_weight: float = 1.0, dice_weight: float = 0.0, use_bn: bool = True, include_bg_in_dice: bool = False, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = Cicek3DUNet(num_classes=num_classes, base=32, use_bn=use_bn)
        self.target_depth = int(target_depth)
        if class_weights is not None:
            cw = np.asarray(class_weights, dtype="float32")
            assert cw.shape[0] == int(num_classes)
            self.register_buffer("class_weights", torch.from_numpy(cw), persistent=True)
        else:
            self.class_weights = None
        self.voxel_weight_key = voxel_weight_key
        self.ignore_index = ignore_index
        self.include_bg_in_dice = include_bg_in_dice
        self.ce_weight = float(ce_weight); self.dice_weight = float(dice_weight)

    def forward(self, x):
        D0 = x.shape[2]
        x_up = _resize_depth_like(x, self.target_depth)
        logits_up = self.backbone(x_up)
        return _resize_logits_depth_like(logits_up, D0)

    def _weighted_softmax_ce(self, logits, target, voxel_weights: Optional[torch.Tensor]):
        if target.ndim == 5 and target.shape[1] == 1: target = target[:, 0]
        weight = getattr(self, "class_weights", None)
        if weight is not None: weight = weight.to(logits.device).float()
        ce_per_voxel = F.cross_entropy(
            logits, target, weight=weight,
            ignore_index=(self.ignore_index if self.ignore_index is not None else -1000),
            reduction="none",
        )
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            ce_per_voxel = ce_per_voxel * valid_mask
        else:
            valid_mask = torch.ones_like(ce_per_voxel)
        if voxel_weights is not None:
            voxel_weights = voxel_weights.to(ce_per_voxel.device).float()
            ce_per_voxel = ce_per_voxel * voxel_weights
            denom = (valid_mask * voxel_weights).sum().clamp_min(1.0)
        else:
            denom = valid_mask.sum().clamp_min(1.0)
        return ce_per_voxel.sum() / denom

    def _dice_loss(self, logits, y, eps=1e-6):
        prob = torch.softmax(logits, dim=1)
        if y.ndim == 5 and y.shape[1] == 1: y = y[:, 0]
        if self.ignore_index is not None:
            mask = (y != self.ignore_index)
            t = torch.where(mask, y.clamp_min(0), torch.zeros_like(y))
        else:
            mask = None; t = y
        onehot = F.one_hot(t.clamp_min(0), num_classes=prob.shape[1]).permute(0,4,1,2,3).float()
        if mask is not None:
            m = mask.unsqueeze(1).float()
            onehot = onehot * m; prob = prob * m
        inter = (prob * onehot).sum(dim=(2,3,4))
        den   = prob.sum(dim=(2,3,4)) + onehot.sum(dim=(2,3,4)) + eps
        dice_pc = 2.0 * inter / den
        if not self.include_bg_in_dice and dice_pc.shape[1] > 1: dice_pc = dice_pc[:, 1:]
        return 1.0 - dice_pc.mean()

    def _loss_and_log(self, logits, y, stage: str, voxel_w: Optional[torch.Tensor], log_metrics: bool = True):
        ce = self._weighted_softmax_ce(logits, y, voxel_w) * self.ce_weight
        loss = ce
        if self.dice_weight > 0.0:
            loss = loss + self._dice_loss(logits, y) * self.dice_weight
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), on_step=False, on_epoch=True, sync_dist=True)
        if log_metrics:
            tgt = _canonicalize_targets_3d(y).to(logits.device)
            (_d,_s,_p, macro_dice, *_rest) = per_class_metrics_3d(logits, tgt, self.hparams.num_classes, ignore_index=self.ignore_index)
            self.log(f"{stage}_macro_dice", macro_dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _unpack(self, batch):
        if isinstance(batch, (list, tuple)):
            x, y = batch; voxel_w = None
        else:
            x, y = batch["image"], batch["label"]
            voxel_w = batch.get(self.voxel_weight_key) if self.voxel_weight_key is not None else None
        return x, y, voxel_w

    def training_step(self, batch, _):
        x, y, vw = self._unpack(batch); logits = self(x)
        return self._loss_and_log(logits, y, "train", voxel_w=vw)

    def validation_step(self, batch, _):
        x, y, vw = self._unpack(batch); logits = self(x)
        return self._loss_and_log(logits, y, "val", voxel_w=vw, log_metrics=True)

    def test_step(self, batch, _):
        x, y, vw = self._unpack(batch); logits = self(x)
        return self._loss_and_log(logits, y, "test", voxel_w=vw)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum,
                               nesterov=bool(self.hparams.nesterov), weight_decay=self.hparams.weight_decay)

# -----------------------------------------------------------------------------
# SwinUNETR (MONAI) + Lightning
# -----------------------------------------------------------------------------
class SwinUNETR_Published(nn.Module):
    def __init__(self, num_classes, img_size=(96, 96, 96), in_channels=1, feature_size=48,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), mlp_ratio=4.0, drop_rate=0.0,
                 attn_drop_rate=0.0, dropout_path_rate=0.0, use_checkpoint=False, norm_name="instance", **kwargs):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        _swin_kwargs = dict(
            img_size=img_size, in_channels=in_channels, out_channels=num_classes, feature_size=feature_size,
            depths=depths, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=dropout_path_rate, norm_name=norm_name,
            use_checkpoint=use_checkpoint, res_block=True,
        )
        sig = inspect.signature(SwinUNETR.__init__); allowed = set(sig.parameters) - {"self"}
        if "img_size" not in allowed:
            if "input_size" in allowed: _swin_kwargs["input_size"] = _swin_kwargs.pop("img_size")
            elif "roi_size" in allowed: _swin_kwargs["roi_size"] = _swin_kwargs.pop("img_size")
            else: _swin_kwargs.pop("img_size", None)
        if "dropout_path_rate" in allowed and "drop_path_rate" in _swin_kwargs:
            _swin_kwargs["dropout_path_rate"] = _swin_kwargs.pop("drop_path_rate")
        if "spatial_dims" in allowed: _swin_kwargs["spatial_dims"] = 3
        self.model = SwinUNETR(**{k: v for k, v in _swin_kwargs.items() if k in allowed})
    def forward(self, x): return self.model(x)

class LitSwinUNETR_Published(pl.LightningModule):
    def __init__(self, num_classes: int, img_size=(96, 96, 96), in_channels: int = 1,
                 feature_size: int = 48, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0, dropout_path_rate: float = 0.0,
                 use_checkpoint: bool = False, norm_name: str = "instance",
                 lr: float = 1e-4, weight_decay: float = 1e-2, warmup_epochs: int = 5,
                 use_ce_alongside_dice: bool = True, ce_weight: float = 0.5,
                 ignore_index: int | None = IGNORE_INDEX, include_bg_in_dice: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.model = SwinUNETR_Published(
            num_classes, img_size=img_size, in_channels=in_channels, feature_size=feature_size, depths=depths,
            num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate, use_checkpoint=use_checkpoint, norm_name=norm_name,
        )
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._total_train_iters = None; self._iters_done = 0

    def forward(self, x):
        x = _pick_first_if_seq(x)
        if x.ndim == 4: x = x.unsqueeze(1)
        x_pad, orig = _pad_to_mult16_3d(x, multiple=32)
        y_pad = self.model(x_pad)
        return _center_crop_3d(y_pad, orig)

    def _dice_loss(self, logits, labels, eps=1e-6):
        if labels.ndim == 5 and labels.shape[1] == 1: labels = labels[:, 0]
        C = logits.shape[1]; probs = torch.softmax(logits, dim=1)
        if self.hparams.ignore_index is not None:
            mask = (labels != self.hparams.ignore_index).unsqueeze(1).float()
            labels = torch.where(labels == self.hparams.ignore_index, torch.zeros_like(labels), labels)
            probs = probs * mask
        onehot = F.one_hot(labels.clamp_min(0), num_classes=C).permute(0,4,1,2,3).float()
        start_c = 0 if self.hparams.include_bg_in_dice else 1
        if start_c >= C: return logits.new_tensor(0.0)
        p, g = probs[:, start_c:], onehot[:, start_c:]
        inter = (p * g).sum(dim=(2,3,4))
        den = p.sum(dim=(2,3,4)) + g.sum(dim=(2,3,4)) + eps
        dice = (2*inter / den).mean()
        return 1.0 - dice

    def _loss(self, logits, labels):
        if labels.ndim == 5 and labels.shape[1] == 1: labels = labels[:, 0]
        dice = self._dice_loss(logits, labels)
        if self.hparams.use_ce_alongside_dice:
            ce = self.ce(logits, labels); w = float(self.hparams.ce_weight)
            return (1.0 - w) * dice + w * ce
        return dice

    def setup(self, stage=None):
        if stage in (None, "fit"):
            est = getattr(self.trainer, "estimated_stepping_batches", None)
            if est and est > 0: self._total_train_iters = int(est)
            else:
                steps = int(self.trainer.num_training_batches or 100)
                epochs = int(self.trainer.max_epochs or 100)
                self._total_train_iters = steps * epochs

    def on_train_batch_start(self, batch, batch_idx):
        warmup_iters = int(self.hparams.warmup_epochs * (self.trainer.num_training_batches or 1))
        t = self._iters_done; T = max(1, self._total_train_iters)
        if t < warmup_iters: lr = self.hparams.lr * float(t + 1) / max(1, warmup_iters)
        else:
            prog = (t - warmup_iters) / max(1, T - warmup_iters)
            lr = 0.5 * self.hparams.lr * (1.0 + math.cos(math.pi * prog))
        opt = self.optimizers()
        if opt:
            for pg in opt.param_groups: pg["lr"] = lr

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._iters_done += 1
        opt = self.optimizers()
        if opt: self.log("lr", opt.param_groups[0]["lr"], on_step=True, prog_bar=False, sync_dist=True)

    def training_step(self, batch, _):
        imgs, lbls = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        logits = self(imgs)
        tgt = _canonicalize_targets_3d(lbls).to(self.device)
        loss = self._loss(logits, tgt)
        (_d,_s,_p, macro_dice, _ms,_mp, micro_dice, _mis,_mip) = per_class_metrics_3d(
            logits, tgt, self.hparams.num_classes, ignore_index=self.hparams.ignore_index
        )
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_macro_dice", macro_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_micro_dice", micro_dice, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        imgs, lbls = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        logits = self(imgs)
        tgt = _canonicalize_targets_3d(lbls).to(self.device)
        val_loss = self._loss(logits, tgt)
        (_d,_s,_p, macro_dice, _ms,_mp, micro_dice, _mis,_mip) = per_class_metrics_3d(
            logits, tgt, self.hparams.num_classes, ignore_index=self.hparams.ignore_index
        )
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_macro_dice", macro_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_micro_dice", micro_dice, on_epoch=True, prog_bar=False, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.999))

# -----------------------------------------------------------------------------
# UNETR (MONAI) + Lightning
# -----------------------------------------------------------------------------
class UNETR_Published(nn.Module):
    def __init__(self, num_classes, img_size=(96, 96, 96), in_channels=1, feature_size=16,
                 hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron",
                 norm_name="instance", res_block=True, dropout_rate=0.0):
        super().__init__()
        try:
            from monai.networks.nets import UNETR
        except ImportError as e:
            raise ImportError("UNETR_Published requires MONAI. `pip install monai`") from e
        sig = inspect.signature(UNETR.__init__); allowed = {k for k in sig.parameters if k != "self"}
        candidate = dict(
            img_size=img_size, in_channels=in_channels, out_channels=num_classes,
            feature_size=feature_size, hidden_size=hidden_size, mlp_dim=mlp_dim,
            num_heads=num_heads, pos_embed=pos_embed, norm_name=norm_name,
            res_block=res_block, dropout_rate=dropout_rate, spatial_dims=3,
        )
        self.net = UNETR(**{k: v for k, v in candidate.items() if k in allowed})
    def forward(self, x): return self.net(x)

class LitUNETR_Published(pl.LightningModule):
    def __init__(self, num_classes: int, img_size=(96, 96, 96), in_channels: int = 1, feature_size: int = 16,
                 hidden_size: int = 768, mlp_dim: int = 3072, num_heads: int = 12, pos_embed: str = "perceptron",
                 norm_name: str = "instance", res_block: bool = True, dropout_rate: float = 0.0,
                 lr: float = 1e-4, weight_decay: float = 1e-2, warmup_epochs: int = 5,
                 use_ce_alongside_dice: bool = True, ce_weight: float = 0.5,
                 ignore_index: int | None = None, include_bg_in_dice: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNETR_Published(num_classes=num_classes, img_size=img_size, in_channels=in_channels,
                                   feature_size=feature_size, hidden_size=hidden_size, mlp_dim=mlp_dim,
                                   num_heads=num_heads, pos_embed=pos_embed, norm_name=norm_name,
                                   res_block=res_block, dropout_rate=dropout_rate)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._total_train_iters = None; self._iters_done = 0

    def forward(self, x):
        x = _pick_first_if_seq(x)
        if x.ndim == 4: x = x.unsqueeze(1)
        x_pad, orig_dhw = _pad_to_mult16_3d(x)
        target = tuple(self.hparams.img_size)
        if x_pad.shape[2:] != target:
            x_res = F.interpolate(x_pad, size=target, mode="trilinear", align_corners=False)
        else:
            x_res = x_pad
        y_res = self.net(x_res)
        if y_res.shape[2:] != x_pad.shape[2:]:
            y_pad = F.interpolate(y_res, size=x_pad.shape[2:], mode="trilinear", align_corners=False)
        else:
            y_pad = y_res
        return _center_crop_3d(y_pad, orig_dhw)

    def _dice_loss(self, logits, labels, eps=1e-6):
        if labels.ndim == 5 and labels.shape[1] == 1: labels = labels[:, 0]
        C = logits.shape[1]; probs = torch.softmax(logits, dim=1)
        mask = None
        if self.hparams.ignore_index is not None:
            mask = (labels != self.hparams.ignore_index).unsqueeze(1).float()
            labels = torch.where(labels == self.hparams.ignore_index, torch.zeros_like(labels), labels)
            probs  = probs * mask
        onehot = F.one_hot(labels.clamp_min(0), num_classes=C).permute(0, 4, 1, 2, 3).float()
        start_c = 0 if self.hparams.include_bg_in_dice else 1
        if start_c >= C: return logits.new_tensor(0.0)
        p = probs[:, start_c:]; g = onehot[:, start_c:]
        inter = (p * g).sum(dim=(2, 3, 4))
        den = p.sum(dim=(2, 3, 4)) + g.sum(dim=(2, 3, 4)) + eps
        dice = (2.0 * inter / den).mean()
        return 1.0 - dice

    def _loss(self, logits, labels):
        if labels.ndim == 5 and labels.shape[1] == 1: labels = labels[:, 0]
        dice = self._dice_loss(logits, labels)
        if self.hparams.use_ce_alongside_dice:
            ce = self.ce(logits, labels); w = float(self.hparams.ce_weight)
            return (1.0 - w) * dice + w * ce
        return dice

    def setup(self, stage=None):
        if stage in (None, "fit"):
            est = getattr(self.trainer, "estimated_stepping_batches", None)
            if est and est > 0: self._total_train_iters = int(est)
            else:
                steps = int(self.trainer.num_training_batches or 100)
                epochs = int(self.trainer.max_epochs or 100)
                self._total_train_iters = steps * epochs

    def on_train_batch_start(self, batch, batch_idx):
        warmup_iters = int(self.hparams.warmup_epochs * (self.trainer.num_training_batches or 1))
        t = self._iters_done; T = max(1, int(self._total_train_iters or 1))
        if t < warmup_iters: lr = self.hparams.lr * float(t + 1) / max(1, warmup_iters)
        else:
            prog = (t - warmup_iters) / max(1, T - warmup_iters)
            lr = 0.5 * self.hparams.lr * (1.0 + math.cos(math.pi * prog))
        opt = self.optimizers()
        if opt:
            for pg in opt.param_groups: pg["lr"] = lr

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._iters_done += 1
        opt = self.optimizers()
        if opt: self.log("lr", opt.param_groups[0]["lr"], on_step=True, prog_bar=False, sync_dist=True)

    def training_step(self, batch, _):
        imgs, lbls = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        logits = self(imgs)
        tgt = _canonicalize_targets_3d(lbls).to(self.device)
        loss = self._loss(logits, tgt)
        (_d,_s,_p, macro_dice, _ms,_mp, micro_dice, _mis,_mip) = per_class_metrics_3d(
            logits, tgt, self.hparams.num_classes, ignore_index=self.hparams.ignore_index
        )
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_macro_dice", macro_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_micro_dice", micro_dice, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        imgs, lbls = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        logits = self(imgs)
        tgt = _canonicalize_targets_3d(lbls).to(self.device)
        val_loss = self._loss(logits, tgt)
        (_d,_s,_p, macro_dice, _ms,_mp, micro_dice, _mis,_mip) = per_class_metrics_3d(
            logits, tgt, self.hparams.num_classes, ignore_index=self.hparams.ignore_index
        )
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_macro_dice", macro_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_micro_dice", micro_dice, on_epoch=True, prog_bar=False, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.999))

# -------------------------
# R2U-Net 3D (Residual Recurrent U-Net)
# -------------------------

class _RecurrentUnit3D(nn.Module):
    """Simple recurrent conv unit (t steps) with shared weights."""
    def __init__(self, channels, t=2):
        super().__init__()
        self.t = int(t)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.inn  = nn.InstanceNorm3d(channels, affine=True)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        h = torch.zeros_like(x)
        out = x
        for _ in range(self.t):
            out = self.act(self.inn(self.conv(out + h)))
            h = out
        return out

class _RRCNNBlock3D(nn.Module):
    """Residual Recurrent Block: 1x1 lift -> recurrent unit -> 1x1 project + residual."""
    def __init__(self, cin, cout, t=2):
        super().__init__()
        self.inp = nn.Conv3d(cin, cout, 1, bias=False)
        self.ru  = _RecurrentUnit3D(cout, t=t)
        self.out = nn.Conv3d(cout, cout, 1, bias=False)
        self.bn  = nn.InstanceNorm3d(cout, affine=True)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.inp(x)
        y  = self.out(self.ru(x1))
        return self.act(self.bn(x1 + y))

class R2UNet3D_backbone(nn.Module):
    def __init__(self, in_channels=1, base=16, t=2):
        super().__init__()
        c = [base, base*2, base*4, base*8, base*16]
        self.e1 = _RRCNNBlock3D(in_channels, c[0], t=t); self.p1 = nn.MaxPool3d(2)
        self.e2 = _RRCNNBlock3D(c[0], c[1], t=t);        self.p2 = nn.MaxPool3d(2)
        self.e3 = _RRCNNBlock3D(c[1], c[2], t=t);        self.p3 = nn.MaxPool3d(2)
        self.e4 = _RRCNNBlock3D(c[2], c[3], t=t);        self.p4 = nn.MaxPool3d(2)
        self.b  = _RRCNNBlock3D(c[3], c[4], t=t)

        self.up4 = nn.ConvTranspose3d(c[4], c[3], 2, 2); self.d4 = _RRCNNBlock3D(c[3]+c[3], c[3], t=t)
        self.up3 = nn.ConvTranspose3d(c[3], c[2], 2, 2); self.d3 = _RRCNNBlock3D(c[2]+c[2], c[2], t=t)
        self.up2 = nn.ConvTranspose3d(c[2], c[1], 2, 2); self.d2 = _RRCNNBlock3D(c[1]+c[1], c[1], t=t)
        self.up1 = nn.ConvTranspose3d(c[1], c[0], 2, 2); self.d1 = _RRCNNBlock3D(c[0]+c[0], c[0], t=t)

        self.out_ch = c[0]

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2)); e4 = self.e4(self.p3(e3))
        b  = self.b(self.p4(e4))
        d4 = self.d4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return d1

class LitR2UNet3D_Published(pl.LightningModule):
    """
    Paper-style R2U-Net 3D (Kadia/Alom):
      - Optim: Adam(lr=1e-3), weight_decay=0.0 (paper doesn't specify WD)
      - Loss: Dice-only (Sørensen–Dice), foreground-only, ignore empty-FG patches
    """
    def __init__(self, num_classes: int,
                 in_channels: int = 1,
                 base_features: int = 16,
                 t: int = 2,
                 # keep paper-faithful defaults below:
                 lr: float = 1e-3, weight_decay: float = 0.0,
                 pad_multiple: int = 16,
                 ignore_index: int | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = R2UNet3D_backbone(in_channels=in_channels, base=base_features, t=t)
        # For binary: prefer single-channel head (paper-style) with sigmoid Dice
        self.binary = (num_classes == 1)
        self.head = nn.Conv3d(self.backbone.out_ch, (1 if self.binary else num_classes), 1)

    def forward(self, x):
        x = _pick_first_if_seq(x)
        if x.ndim == 4: x = x.unsqueeze(1)
        x_pad, orig = _pad_to_mult16_3d(x, multiple=self.hparams.pad_multiple)
        y = self.head(self.backbone(x_pad))
        return _center_crop_3d(y, orig)

    @staticmethod
    def _dice_only_loss_with_logits(logits, y, ignore_index=None, eps=1e-6):
        """
        - Binary case: logits[B,1,D,H,W] -> sigmoid probs; y[B,D,H,W] in {0,1}
        - Multi-class case: logits[B,C,...] -> softmax probs; y in {0..C-1}
        - Foreground-only Dice, ignore empty-FG samples in the batch
        """
        if y.ndim == 5 and y.size(1) == 1:
            y = y[:, 0]
        B = logits.size(0)

        if logits.size(1) == 1:
            # Binary Dice (sigmoid)
            probs = torch.sigmoid(logits)
            y_true = (y > 0).float().unsqueeze(1)  # [B,1,D,H,W]
            # mask out ignore_index if present
            if ignore_index is not None:
                mask = (y != ignore_index).float().unsqueeze(1)
                probs = probs * mask
                y_true = y_true * mask
            # per-sample Dice over foreground
            inter = (probs * y_true).sum(dim=(2,3,4))
            denom = (probs + y_true).sum(dim=(2,3,4))
            # ignore empty-FG samples
            has_fg = (y_true.sum(dim=(2,3,4)) > 0)
            inter = inter[has_fg]; denom = denom[has_fg]
            if inter.numel() == 0:
                # if the whole batch is empty-FG, return zero loss (no gradient)
                return logits.new_tensor(0.0), logits.new_tensor(0.0)
            dice = (2*inter + eps) / (denom + eps)
            loss = 1.0 - dice.mean()
            return loss, dice.mean()
        else:
            # Multi-class Dice (softmax), foreground classes only (1..C-1)
            C = logits.size(1)
            probs = torch.softmax(logits, dim=1)  # [B,C,D,H,W]
            # one-hot y (ignore index masked out later)
            if ignore_index is not None:
                valid = (y != ignore_index)
                y = y.clone()
                y[~valid] = 0  # temporary, will zero with mask
            y_oh = torch.nn.functional.one_hot(y.long(), num_classes=C)  # [B,D,H,W,C]
            y_oh = y_oh.permute(0,4,1,2,3).float()  # [B,C,D,H,W]
            if ignore_index is not None:
                mask = valid.unsqueeze(1).float()
                probs = probs * mask
                y_oh = y_oh * mask
            # foreground classes only
            if C <= 1:
                return logits.new_tensor(0.0), logits.new_tensor(0.0)
            probs_fg = probs[:, 1:, ...]
            y_fg = y_oh[:, 1:, ...]
            # drop samples with empty foreground across all FG classes
            has_fg = (y_fg.sum(dim=(1,2,3,4)) > 0)
            if has_fg.any():
                probs_fg = probs_fg[has_fg]
                y_fg = y_fg[has_fg]
                inter = (probs_fg * y_fg).sum(dim=(2,3,4))  # [B_fg, C-1]
                denom = (probs_fg + y_fg).sum(dim=(2,3,4))
                dice_per_class = (2*inter + eps) / (denom + eps)
                dice = dice_per_class.mean()  # mean over classes and batch
                loss = 1.0 - dice
                return loss, dice
            else:
                return logits.new_tensor(0.0), logits.new_tensor(0.0)

    def training_step(self, batch, _):
        x, y = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        logits = self(x)
        loss, dice = self._dice_only_loss_with_logits(logits, y, ignore_index=self.hparams.ignore_index)
        # log Dice even if it's zero; avoid NaN by handling empty-FG above
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_macro_dice", dice, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        logits = self(x)
        val_loss, dice = self._dice_only_loss_with_logits(logits, y, ignore_index=self.hparams.ignore_index)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_macro_dice", dice, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

# -------------------------
# ResUNet++ 3D (ASPP + SE + Attn gates)
# -------------------------
class ResidualUnit3D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.n1 = nn.InstanceNorm3d(cout, affine=True)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = nn.InstanceNorm3d(cout, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
    def forward(self, x):
        s = self.skip(x)
        x = self.act(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        return self.act(x + s)
class ResUNetPP3D_backbone(nn.Module):
    """
    Encoder: Residual units
    Bottleneck: ASPP
    Skips: SE + Attention gate at each concat
    """
    def __init__(self, in_channels=1, base=16):
        super().__init__()
        c = [base, base*2, base*4, base*8, base*16]

        # Encoder
        self.e1 = ResidualUnit3D(in_channels, c[0]); self.p1 = nn.MaxPool3d(2)
        self.e2 = ResidualUnit3D(c[0], c[1]);        self.p2 = nn.MaxPool3d(2)
        self.e3 = ResidualUnit3D(c[1], c[2]);        self.p3 = nn.MaxPool3d(2)
        self.e4 = ResidualUnit3D(c[2], c[3]);        self.p4 = nn.MaxPool3d(2)

        # Bottleneck (ASPP)
        self.b_aspp_in = ResidualUnit3D(c[3], c[4])
        self.b_aspp    = ASPP3D(c[4], c[4])
        self.b_aspp_out= ResidualUnit3D(c[4], c[4])

        # SE on encoder outputs (skips)
        self.se1 = SE3D(c[0]); self.se2 = SE3D(c[1]); self.se3 = SE3D(c[2]); self.se4 = SE3D(c[3])

        # Decoder with attention gates
        self.up4 = nn.ConvTranspose3d(c[4], c[3], 2, 2)
        self.ag4 = AttentionGate3d(g_c=c[3], x_c=c[3], inter_c=c[3]//2)
        self.d4  = ResidualUnit3D(c[3]+c[3], c[3])

        self.up3 = nn.ConvTranspose3d(c[3], c[2], 2, 2)
        self.ag3 = AttentionGate3d(g_c=c[2], x_c=c[2], inter_c=c[2]//2)
        self.d3  = ResidualUnit3D(c[2]+c[2], c[2])

        self.up2 = nn.ConvTranspose3d(c[2], c[1], 2, 2)
        self.ag2 = AttentionGate3d(g_c=c[1], x_c=c[1], inter_c=c[1]//2)
        self.d2  = ResidualUnit3D(c[1]+c[1], c[1])

        self.up1 = nn.ConvTranspose3d(c[1], c[0], 2, 2)
        self.d1  = ResidualUnit3D(c[0]+c[0], c[0])

        self.out_ch = c[0]

    def forward(self, x):
        e1 = self.e1(x);   e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2)); e4 = self.e4(self.p3(e3))

        b = self.b_aspp_out(self.b_aspp(self.b_aspp_in(self.p4(e4))))

        u4 = self.up4(b);   s4 = self.ag4(u4, self.se4(e4)); d4 = self.d4(torch.cat([u4, s4], 1))
        u3 = self.up3(d4);  s3 = self.ag3(u3, self.se3(e3)); d3 = self.d3(torch.cat([u3, s3], 1))
        u2 = self.up2(d3);  s2 = self.ag2(u2, self.se2(e2)); d2 = self.d2(torch.cat([u2, s2], 1))
        u1 = self.up1(d2);  s1 = self.se1(e1);               d1 = self.d1(torch.cat([u1, s1], 1))
        return d1


class LitResUNetPP3D_Published(pl.LightningModule):
    """
    Paper-style recipe (ResUNet++ family):
      - Optim: Adam (default lr=1e-3)  [paper LR not strictly fixed; 1e-3 is standard]
      - Loss: Dice + Cross-Entropy (weights configurable)
    """
    def __init__(self, num_classes: int,
                 in_channels: int = 1, base_features: int = 16,
                 include_bg_in_dice: bool = False, ignore_index: Optional[int] = None,
                 ce_weight: float = 0.5, dice_weight: float = 0.5,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 pad_multiple: int = 16):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = ResUNetPP3D_backbone(in_channels=in_channels, base=base_features)
        self.head = nn.Conv3d(self.backbone.out_ch, num_classes, 1)

    def forward(self, x):
        x = _pick_first_if_seq(x)
        if x.ndim == 4: x = x.unsqueeze(1)  # allow (B,D,H,W)
        x_pad, orig = _pad_to_mult16_3d(x, multiple=self.hparams.pad_multiple)
        y = self.head(self.backbone(x_pad))
        return _center_crop_3d(y, orig)

    def _loss(self, logits, y):
        return dice_ce_loss_with_metrics(
            logits, y, self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            include_bg_in_dice=self.hparams.include_bg_in_dice,
            ce_weight=self.hparams.ce_weight,
            dice_weight=self.hparams.dice_weight,
        )

    def training_step(self, batch, _):
        x, y = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        logits = self(x); loss, macro_dice, _ = self._loss(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_macro_dice", macro_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        logits = self(x); val_loss, macro_dice, _ = self._loss(logits, y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_macro_dice", macro_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


# ----- Lightning wrappers -----
def upgrade_spct_with_novel_blocks(m: nn.Module,
                                   use_efilm: bool = True,
                                   use_fouriergate: bool = True,
                                   use_moe: bool = False,
                                   moe_K: int = 3):
    """
    Recursively walk a module and replace any `_DoubleConvSpectral` with the novel block,
    preserving in/out channels and ksd. Works for enc/dec/bott across your UNet3D_SpectralCore.
    """
    for name, child in list(m.named_children()):
        # replace target blocks
        if isinstance(child, _DoubleConvSpectral):
            # infer channels/ksd from conv weights
            conv1: nn.Conv3d = child.b1[0]
            conv2: nn.Conv3d = child.b2[0]
            cin  = int(conv1.in_channels)
            cout = int(conv2.out_channels)
            ksd  = int(conv1.kernel_size[0])

            new_block = _DoubleConvSpectral_Novel(
                cin, cout, ksd=ksd,
                use_efilm=use_efilm,
                use_fouriergate=use_fouriergate,
                use_moe=use_moe,
                moe_K=moe_K,
            )
            setattr(m, name, new_block)
        else:
            # recurse
            upgrade_spct_with_novel_blocks(child, use_efilm, use_fouriergate, use_moe, moe_K)
    return m

class _DoubleConvSpectral_Novel(nn.Module):
    """
    Two (ksd,3,3) convs + optional Energy-FiLM + optional FourierGate + optional Spectral-MoE.
    Keeps the same in/out channel contract as your original _DoubleConvSpectral.
    """
    def __init__(self, cin, cout, ksd=1, norm="instance", act="lrelu",
                 use_efilm: bool = False,
                 use_fouriergate: bool = False,
                 use_moe: bool = False,
                 moe_K: int = 3):
        super().__init__()
        self.pre = nn.Sequential(
            _conv3x3xk(cin,  cout, ksd, bias=False), _norm3d(cout, norm), _act(act)
        )
        # second path: either classic conv or MoE
        if use_moe:
            self.body = SpectralMoE3D(cout, K=int(moe_K), ksds=(1,ksd,ksd), dilations=(1,1,2),
                                      norm=norm, act=act)
        else:
            self.body = nn.Sequential(
                _conv3x3xk(cout, cout, ksd, bias=False), _norm3d(cout, norm), _act(act)
            )
        self.efilm = EnergyFiLM3D(cout) if use_efilm else nn.Identity()
        self.fgate = FourierGate3D()     if use_fouriergate else nn.Identity()

    def forward(self, x):
        x = self.pre(x)
        x = self.body(x)
        x = self.efilm(x)
        x = self.fgate(x)
        return x
class EnergyFiLM3D(nn.Module):
    """
    Per-energy FiLM: learn (gamma, beta) for each energy index via a tiny MLP
    on a sinusoidal code. Broadcast over H,W and modulate channels.
    """
    def __init__(self, channels: int, hidden: int = 32, pe_dims: int = 16):
        super().__init__()
        self.channels = int(channels)
        self.pe_dims  = int(pe_dims)
        self.mlp = nn.Sequential(
            nn.Conv1d(self.pe_dims, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 2 * self.channels, 1, bias=True),  # -> gamma,beta
        )

    @staticmethod
    def _sinusoidal_pe(F: int, d: int, device):
        # [1, d, F] with classic transformer-style frequencies
        pos  = torch.arange(F, dtype=torch.float32, device=device)[None, None, :]
        i    = torch.arange(max(1, d // 2), dtype=torch.float32, device=device)[None, :, None]
        denom= torch.exp(i * (-math.log(10000.0) / max(1,(d//2))))
        pe   = torch.cat([torch.sin(pos*denom), torch.cos(pos*denom)], dim=1)
        if pe.shape[1] < d:
            pe = torch.cat([pe, torch.zeros(1,1,pe.shape[-1], device=device)], dim=1)
        return pe  # [1, d, F]

    def forward(self, x):  # x: [B,C,F,H,W]
        B, C, F, H, W = x.shape
        pe = self._sinusoidal_pe(F, self.pe_dims, x.device).expand(B, -1, -1)  # [B,pe,F]
        gb = self.mlp(pe)                                     # [B, 2C, F]
        gamma, beta = gb[:, :C], gb[:, C:]                    # [B,C,F]
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1) # [B,C,F,1,1]
        beta  = beta.unsqueeze(-1).unsqueeze(-1)              # [B,C,F,1,1]
        return x * (1 + gamma) + beta


class FourierGate3D(nn.Module):
    """
    Frequency-domain gate over the spectral axis:
    avg pool over C,H,W -> rFFT along F -> learnable magnitude mask -> iFFT -> sigmoid gate.
    NOTE: Assumes F is fixed during training (common in SPCCT).
    """
    def __init__(self, learn_phase: bool = False):
        super().__init__()
        self.learn_phase = bool(learn_phase)
        self.mag_scale   = nn.Parameter(torch.ones(1))
        self._mask = None  # lazily registered nn.Parameter with shape [1,1,L,1,1]

    def forward(self, x):  # [B,C,F,H,W]
        B, C, F, H, W = x.shape
        s  = x.mean(dim=(1,3,4), keepdim=True)          # [B,1,F,1,1]
        Sf = torch.fft.rfft(s, dim=2)                   # [B,1,L,1,1] complex
        L  = Sf.shape[2]
        if (self._mask is None) or (self._mask.shape[2] != L):
            # register a fresh learnable mask when F changes
            self._mask = nn.Parameter(torch.ones(1,1,L,1,1, device=x.device, dtype= Sf.real.dtype))
            self.register_parameter("freq_mask", self._mask)

        M  = self.freq_mask * self.mag_scale            # real-valued magnitude mask
        if self.learn_phase:
            Sf = Sf * (M + 1j*0.01)                    # tiny imaginary term to nudge phase
        else:
            Sf = Sf * M
        w = torch.fft.irfft(Sf, n=F, dim=2)            # [B,1,F,1,1]
        w = torch.sigmoid(w)
        return x * w


def build_spct_energyfilm_fourier(num_classes=NUM_CLASSES, base=32, ksd=3,
                                  use_se=True, use_specse=True, use_spatial=False, use_skip_gate=False,
                                  **kw):
    core = UNet3D_SpectralCore(
        in_channels=1, num_classes=num_classes, base=base, ksd=ksd,
        use_se=use_se, use_specse=use_specse, use_spatial=use_spatial, use_skip_gate=use_skip_gate,
        **kw
    )
    return upgrade_spct_with_novel_blocks(core, use_efilm=True, use_fouriergate=True, use_moe=False)


class LitSPCT_EFiLM_FourierGate(BaseLitModel):
    def __init__(self, num_classes=NUM_CLASSES, lr=BEST_LR, base=32, ksd=3,
                 use_se=True, use_specse=True, use_spatial=False, use_skip_gate=False, **kw):
        super().__init__(num_classes=num_classes, lr=lr, is_3d=True)
        self.model = build_spct_energyfilm_fourier(num_classes=num_classes, base=base, ksd=ksd,
                                                   use_se=use_se, use_specse=use_specse,
                                                   use_spatial=use_spatial, use_skip_gate=use_skip_gate, **kw)
class LitSPCT_EnergyFiLM(BaseLitModel):
    def __init__(self, num_classes=NUM_CLASSES, lr=BEST_LR, base=32, ksd=3,
                 use_se=True, use_specse=True, use_spatial=False, use_skip_gate=False, **kw):
        super().__init__(num_classes=num_classes, lr=lr, is_3d=True, **kw)
        core = UNet3D_SpectralCore(
            in_channels=1, num_classes=num_classes, base=base, ksd=ksd,
            use_se=use_se, use_specse=use_specse, use_spatial=use_spatial, use_skip_gate=use_skip_gate
        )
        self.model = upgrade_spct_with_novel_blocks(core, use_efilm=True, use_fouriergate=False, use_moe=False)

class LitSPCT_FourierGate(BaseLitModel):
    def __init__(self, num_classes=NUM_CLASSES, lr=BEST_LR, base=32, ksd=3,
                 use_se=True, use_specse=True, use_spatial=False, use_skip_gate=False, **kw):
        super().__init__(num_classes=num_classes, lr=lr, is_3d=True, **kw)
        core = UNet3D_SpectralCore(
            in_channels=1, num_classes=num_classes, base=base, ksd=ksd,
            use_se=use_se, use_specse=use_specse, use_spatial=use_spatial, use_skip_gate=use_skip_gate
        )
        self.model = upgrade_spct_with_novel_blocks(core, use_efilm=False, use_fouriergate=True, use_moe=False)

class LitSPCT_SEspec(_LitSPCT_Base):
    """Channel-SE + Spectral-SE at all stages."""
    def __init__(self, num_classes=NUM_CLASSES, lr=BEST_LR, base=32, pad_multiple=16):
        super().__init__(num_classes=num_classes, lr=lr, pad_multiple=pad_multiple)
        self.model = UNet3D_SpectralCore(
            in_channels=1, num_classes=num_classes, base=base, ksd=3,
            use_se=True, use_specse=True, use_spatial=False, use_skip_gate=False
        )

class LitSPCT_ControlUNet(BaseLitModel):
    def __init__(self, num_classes=NUM_CLASSES, lr=BEST_LR, base=32, ksd=3,
                 use_se=False, use_specse=False, use_spatial=False, use_skip_gate=False, **kw):
        super().__init__(num_classes=num_classes, lr=lr, is_3d=True, **kw)
        self.model = UNet3D_SpectralCore(
            in_channels=1,
            num_classes=num_classes,
            base=base,
            ksd=ksd,
            use_se=use_se,
            use_specse=use_specse,
            use_spatial=use_spatial,
            use_skip_gate=use_skip_gate,
        )