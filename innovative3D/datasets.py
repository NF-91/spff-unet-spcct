# ─────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────
import os, random
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl

from innovative3D.config import (
    IMAGE_HEIGHT, IMAGE_WIDTH,
    NUM_CLASSES, IGNORE_INDEX,
    BATCH_SIZE, NUM_WORKERS, num_workers,
    grid_size,USE_VMI, VMI_WEIGHTS, VMI_CLIP, VMI_RETURN_DEPTH
)
from innovative3D.helpers import (
    create_image_and_labels_for_dataset,
    generate_cumulative_grid_sizes,apply_vmi_preprocess
)
from innovative3D.config import trainval_sets, test_set  # keep identical behavior to your old setup


# ─────────────────────────────────────────────────────────────
# 2. SMALL UTILS
# ─────────────────────────────────────────────────────────────
def _sanitize_labels(lbl: torch.Tensor, num_classes: int, ignore_index: int | None):
    if not torch.is_tensor(lbl):
        lbl = torch.as_tensor(lbl)
    if lbl.dtype != torch.long:
        lbl = lbl.long()
    if ignore_index is None:
        lbl = torch.where((lbl < 0) | (lbl >= num_classes), torch.zeros_like(lbl), lbl)
    else:
        lbl = torch.where((lbl < 0) | (lbl >= num_classes), torch.full_like(lbl, ignore_index), lbl)
    return lbl


# ─────────────────────────────────────────────────────────────
# 2.1 RAGGED-SAFE GRID SHUFFLE (SEPARABLE STRIPES)
# ─────────────────────────────────────────────────────────────
def _grid_boundaries(n: int, g: int):
    # e.g., n=512, g=5 → boundaries at [0,102,204,307,409,512] (ragged edges allowed)
    return [(i * n) // g for i in range(g)] + [n]

def _shuffle_stripes(x: torch.Tensor, y: torch.Tensor | None, g_rows: int, g_cols: int):
    """
    Shuffle row-stripes and column-stripes independently.
    Works for:
      x: [C,H,W] or [1,F,H,W]
      y: [H,W]   or [F,H,W]
    """
    if g_rows <= 1 and g_cols <= 1:
        return x, y

    H, W = x.shape[-2], x.shape[-1]
    hs = _grid_boundaries(H, max(1, int(g_rows)))
    ws = _grid_boundaries(W, max(1, int(g_cols)))

    from collections import defaultdict
    row_groups = defaultdict(list)  # key: height → list of (h0,h1)
    col_groups = defaultdict(list)  # key: width  → list of (w0,w1)

    for i in range(len(hs) - 1):
        h0, h1 = hs[i], hs[i + 1]
        row_groups[h1 - h0].append((h0, h1))
    for j in range(len(ws) - 1):
        w0, w1 = ws[j], ws[j + 1]
        col_groups[w1 - w0].append((w0, w1))

    # rows first
    row_map = {}
    for h, lst in row_groups.items():
        perm = lst[:]
        random.shuffle(perm)
        for (t_h0, t_h1), (s_h0, s_h1) in zip(lst, perm):
            row_map[(t_h0, t_h1)] = (s_h0, s_h1)

    xr = x.clone()
    yr = y.clone() if y is not None else None
    for (t_h0, t_h1), (s_h0, s_h1) in row_map.items():
        xr[..., t_h0:t_h1, :] = x[..., s_h0:s_h1, :]
        if yr is not None:
            yr[..., t_h0:t_h1, :] = y[..., s_h0:s_h1, :]

    # then columns
    col_map = {}
    for w, lst in col_groups.items():
        perm = lst[:]
        random.shuffle(perm)
        for (t_w0, t_w1), (s_w0, s_w1) in zip(lst, perm):
            col_map[(t_w0, t_w1)] = (s_w0, s_w1)

    xc = xr.clone()
    yc = yr.clone() if yr is not None else None
    for (t_w0, t_w1), (s_w0, s_w1) in col_map.items():
        xc[..., :, t_w0:t_w1] = xr[..., :, s_w0:s_w1]
        if yc is not None:
            yc[..., :, t_w0:t_w1] = yr[..., :, s_w0:s_w1]

    return xc, yc

def _grid_shuffle_xy(x: torch.Tensor, y: torch.Tensor | None, gh: int, gw: int):
    """
    Wrapper used by the augmenter. Uses separable stripe shuffle.
    Extremely defensive: if anything goes wrong, return identity.
    """
    try:
        return _shuffle_stripes(x, y, gh, gw)
    except Exception:
        return x, y


# ─────────────────────────────────────────────────────────────
# 2.2 TRAIN AUGMENT (per-sample G + visible stamp)
# ─────────────────────────────────────────────────────────────
class TrainGridAug:
    """
    Train-time aug that **uses per-sample grid size `gs`** when provided.
      - random H/V flips
      - random 90° rotations
      - light intensity jitter + gaussian noise (image only)
      - grid-shuffle with probability p_grid and grid size = `gs` (if gs>1)
      - optional bright stamp on top-left to make grid-shuffle visible in saved images
    """
    def __init__(self,
                 gs_choices=(2,3,4,5),
                 p_grid=1.0,
                 flip_p=0.5,
                 rot90_p=0.5,
                 jitter_p=0.3,
                 noise_p=0.3,
                 noise_std=0.01,
                 stamp_top_left=True):
        self.gs_choices = tuple(int(g) for g in gs_choices)
        self.p_grid = float(p_grid)
        self.flip_p = float(flip_p)
        self.rot90_p = float(rot90_p)
        self.jitter_p = float(jitter_p)
        self.noise_p = float(noise_p)
        self.noise_std = float(noise_std)
        self.stamp = bool(stamp_top_left)

    def __call__(self, x: torch.Tensor, y: torch.Tensor | None, gs: int | None):
        # x: 2D→(C,H,W) or 3D→(1,F,H,W). y: (H,W) or (F,H,W).
        assert x.ndim in (3,4), f"Unexpected x.ndim={x.ndim}"

        # flips
        if random.random() < self.flip_p:
            x = torch.flip(x, dims=(-1,))
            if y is not None: y = torch.flip(y, dims=(-1,))
        if random.random() < self.flip_p:
            x = torch.flip(x, dims=(-2,))
            if y is not None: y = torch.flip(y, dims=(-2,))

        # rot90
        if random.random() < self.rot90_p:
            k = random.randint(1,3)
            x = torch.rot90(x, k, dims=(-2, -1))
            if y is not None:
                y = torch.rot90(y, k, dims=(-2, -1))

        # jitter
        if random.random() < self.jitter_p:
            scale = 1.0 + 0.1*(2*random.random()-1)  # ±10%
            shift = 0.05*(2*random.random()-1)      # ±0.05
            x = x*scale + shift

        # noise
        if random.random() < self.noise_p:
            v = x.detach().std().item()
            if v > 0:
                std = min(self.noise_std, 0.25*v)
                x = x + torch.randn_like(x)*std

        # grid-shuffle
        run_grid = (random.random() < self.p_grid)
        use_gs = int(gs) if (gs is not None) else None
        if use_gs is None or use_gs < 1:
            use_gs = random.choice(self.gs_choices) if self.gs_choices else 1

        if run_grid and use_gs > 1:
            x, y = _grid_shuffle_xy(x, y, use_gs, use_gs)

            # stamp for visibility in saved montages/overlays
            if self.stamp:
                if x.ndim == 4:  # (1,F,H,W) → stamp first frame
                    x[0, 0, :32, :32] = x[0, 0, :32, :32].max() + x.abs().max().clamp(min=1.0)*0.25
                else:            # (C,H,W) → stamp first channel
                    x[0, :32, :32] = x[0, :32, :32].max() + x.abs().max().clamp(min=1.0)*0.25

        return x, y


# ─────────────────────────────────────────────────────────────
# 3. DATASETS
# ─────────────────────────────────────────────────────────────
class DicomDataset3D(Dataset):
    """
    3D:
      - images: np.array [N, F, H, W]
      - labels: np.array [N, F, H, W]
      - transform expects (img: [1,F,H,W], lbl: [F,H,W], grid_size)
    """
    def __init__(self, images, labels, grid_sizes, transform=None):
        self.images = images
        self.labels = labels
        self.grid_sizes = grid_sizes
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)  # [F,H,W]
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)     # [F,H,W]
        gs  = self.grid_sizes[idx]

        lbl = torch.where(lbl >= NUM_CLASSES, IGNORE_INDEX, lbl)
        img = img.unsqueeze(0)  # [1,F,H,W]

        if self.transform:
            img, lbl = self.transform(img, lbl, gs)

        return img, lbl


class DicomDataset2D(Dataset):
    """
    2D:
      - images: [C=NUM_FRAMES, H, W]
      - labels: [F,H,W] → collapsed to [H,W] via OR across frames
      - transform expects (img: [C,H,W], lbl: [H,W], grid_size)
    """
    def __init__(self, images, labels, grid_sizes, transforms=None):
        self.images = images
        self.labels = labels
        self.grid_sizes = grid_sizes
        self.transforms = transforms

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img3d = torch.tensor(self.images[idx], dtype=torch.float32)  # [C,H,W]
        lbl3d = self.labels[idx]                                     # [F,H,W]

        combined_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int64)
        for c in range(1, NUM_CLASSES):
            mask_c = np.any(lbl3d == c, axis=0)
            combined_mask[mask_c] = c

        img2d = img3d
        lbl2d = torch.tensor(combined_mask, dtype=torch.long)
        lbl2d = _sanitize_labels(lbl2d, NUM_CLASSES, IGNORE_INDEX)

        if self.transforms:
            gs = self.grid_sizes[idx]
            img2d, lbl2d = self.transforms(img2d, lbl2d, gs)
            lbl2d = _sanitize_labels(lbl2d, NUM_CLASSES, IGNORE_INDEX)

        return img2d, lbl2d


# ─────────────────────────────────────────────────────────────
# 4. DATAMODULES
# ─────────────────────────────────────────────────────────────
class MultiDicomDataModule3D(pl.LightningDataModule):
    def __init__(self, configs, batch_size=BATCH_SIZE, num_frames=5):
        super().__init__()
        self.configs = configs
        self.batch_size = batch_size
        self.num_frames = num_frames

    def prepare_data(self): pass

    def setup(self, stage=None):
        # build train/val from provided configs (same as your old flow)
        all_imgs, all_lbls = [], []
        for cfg in self.configs:
            imgs, lbls = create_image_and_labels_for_dataset(cfg, self.num_frames)
            all_imgs.append(imgs); all_lbls.append(lbls)

        X = np.concatenate(all_imgs, axis=0)
        Y = np.concatenate(all_lbls, axis=0)
        G = generate_cumulative_grid_sizes(len(X), 10, 0.3)  # per-sample G

        tr_x, tr_y, tr_g, val_x, val_y, val_g, _, _, _ = self.ensure_all_classes_in_training(
            X, Y, G, NUM_CLASSES, test_size=0.2, val_size=1.0, random_state=42
        )

        # per-sample-G augmenter (stamp ON for train, OFF for val)
        aug_train = TrainGridAug(gs_choices=(2,3,4,5), p_grid=1.0, stamp_top_left=True)
        aug_val   = TrainGridAug(gs_choices=(2,3,4,5), p_grid=0.0,
                                 flip_p=0.0, rot90_p=0.0, jitter_p=0.0, noise_p=0.0,
                                 stamp_top_left=False)

        self.train_set = DicomDataset3D(tr_x, tr_y, tr_g, transform=aug_train)
        self.val_set   = DicomDataset3D(val_x, val_y, val_g, transform=aug_val)

        # test set (no aug)
        test_imgs, test_lbls = create_image_and_labels_for_dataset(test_set, self.num_frames)
        test_g = generate_cumulative_grid_sizes(len(test_imgs), grid_size, 0.3)
        self.test_set = DicomDataset3D(test_imgs, test_lbls, test_g, transform=None)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0), drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0), drop_last=False,
        )

    def test_dataloader(self):
        if getattr(self, "test_set", None) is None:
            raise AttributeError("Test dataset not set. Did setup('test') run?")
        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers,
        )

    @staticmethod
    def ensure_all_classes_in_training(X, Y, G, num_classes, test_size=0.2, val_size=1.0, random_state=42):
        total_indices = np.arange(len(X))
        class_to_indices = {cls: set(np.where([np.any(y == cls) for y in Y])[0]) for cls in range(num_classes)}
        required_indices = set()
        for cls, inds in class_to_indices.items():
            if inds:
                required_indices.add(next(iter(inds)))

        remaining = list(set(total_indices) - required_indices)
        np.random.seed(random_state); np.random.shuffle(remaining)

        n_train = int(len(X) * (1 - test_size))
        extra_train = n_train - len(required_indices)
        train_inds = list(required_indices) + remaining[:extra_train]
        testval = remaining[extra_train:]
        n_val = int(len(testval) * val_size)
        val_inds = testval[:n_val]
        test_inds = testval[n_val:]

        return (
            X[train_inds], Y[train_inds], np.array(G)[train_inds],
            X[val_inds],   Y[val_inds],   np.array(G)[val_inds],
            X[test_inds],  Y[test_inds],  np.array(G)[test_inds],
        )


class MultiDicomDataModule2D(pl.LightningDataModule):
    def __init__(self, configs, batch_size=BATCH_SIZE, num_frames=5):
        super().__init__()
        self.configs = configs
        self.batch_size = batch_size
        self.num_frames = num_frames

    def prepare_data(self): pass

    def setup(self, stage=None):
        all_imgs, all_lbls = [], []
        for cfg in self.configs:
            imgs, lbls = create_image_and_labels_for_dataset(cfg, self.num_frames)
            all_imgs.append(imgs); all_lbls.append(lbls)

        X = np.concatenate(all_imgs, axis=0)
        Y = np.concatenate(all_lbls, axis=0)
        G = generate_cumulative_grid_sizes(len(X), 10, 0.3)  # per-sample G

        tr_x, tr_y, tr_g, val_x, val_y, val_g, _, _, _ = MultiDicomDataModule3D.ensure_all_classes_in_training(
            X, Y, G, NUM_CLASSES, test_size=0.2, val_size=1.0, random_state=42
        )

        aug_train = TrainGridAug(gs_choices=(2,3,4,5), p_grid=1.0, stamp_top_left=True)
        aug_val   = TrainGridAug(gs_choices=(2,3,4,5), p_grid=0.0,
                                 flip_p=0.0, rot90_p=0.0, jitter_p=0.0, noise_p=0.0,
                                 stamp_top_left=False)

        self.train_set = DicomDataset2D(tr_x, tr_y, tr_g, transforms=aug_train)
        self.val_set   = DicomDataset2D(val_x, val_y, val_g, transforms=aug_val)

        test_imgs, test_lbls = create_image_and_labels_for_dataset(test_set, self.num_frames)
        test_g = generate_cumulative_grid_sizes(len(test_imgs), grid_size, 0.3)
        self.test_set = DicomDataset2D(test_imgs, test_lbls, test_g, transforms=None)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0), drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0), drop_last=False,
        )

    def test_dataloader(self):
        if getattr(self, "test_set", None) is None:
            raise AttributeError("Test dataset not set. Did setup('test') run?")
        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers,
        )
