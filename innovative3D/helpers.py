# ─────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────
import os, random, multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Avoids issues with non-interactive backends
import matplotlib.pyplot as plt
import seaborn as sns
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


import pydicom
import shutil
import warnings
from sklearn.metrics import precision_recall_curve, auc, precision_score, jaccard_score
from sklearn.metrics import roc_auc_score
from matplotlib.patches import Patch

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger

from innovative3D.config import (
    IMAGE_WIDTH, IMAGE_HEIGHT,
    NUM_CLASSES, IGNORE_INDEX,
    BATCH_SIZE, NUM_WORKERS, num_workers,
    trainval_sets, test_set, grid_size,   global_label_names
)



class AugmentImageAndLabels:
    def __init__(self, p_flip=0.5, p_rotate=0.5, brightness_range=(0.9,1.1), noise_std=0.01):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.brightness_range = brightness_range
        self.noise_std = noise_std
        self._debug_saved = False  # only save the PNGs once per process

    def __call__(self, image, label, grid_size=1):
        # If label is 2D, add channel dim
        if label.ndim == 2:
            label = label.unsqueeze(0)

        # Pull out the volume as a NumPy array, then take its center slice
        vol = image[0].detach().cpu().numpy()  # shape: (F, H, W)
        center_idx = vol.shape[0] // 2
        debug_before = vol[center_idx]         # now (H, W), valid for imsave

        # 1) Flip
        if random.random() < self.p_flip:
            if random.random() < 0.5:
                image, label = TF.hflip(image), TF.hflip(label)
            else:
                image, label = TF.vflip(image), TF.vflip(label)

        # 2) Rotate
        if random.random() < self.p_rotate:
            k = random.choice([1,2,3])
            image, label = TF.rotate(image, 90*k), TF.rotate(label, 90*k)
            # image = TF.rotate(image, 90 * k, interpolation=InterpolationMode.BILINEAR)
            # label = TF.rotate(label, 90 * k, interpolation=InterpolationMode.NEAREST)

        # 3) Brightness scaling
        factor = random.uniform(*self.brightness_range)
        image = image * factor

        # 4) Gaussian noise
        img_std = image.std().item()
        image = image + torch.randn_like(image) * (img_std * self.noise_std)

        # 5) Grid‐shuffle (if 3D volume)
        if image.dim() == 4:
            x = image.squeeze(0)
            Fdim, H, W = x.shape
            if grid_size > 1:
                ph, pw = H // grid_size, W // grid_size
                if ph and pw:
                    coords, pimgs, plbls = [], [], []
                    for f in range(Fdim):
                        for i in range(grid_size):
                            for j in range(grid_size):
                                y0, x0 = i*ph, j*pw
                                y1, x1 = min(H, y0+ph), min(W, x0+pw)
                                if y1>y0 and x1>x0:
                                    patch = x[f, y0:y1, x0:x1]
                                    pl    = label[f, y0:y1, x0:x1]
                                    if patch.numel() and pl.numel():
                                        coords.append((f, y0, x0))
                                        pimgs.append(patch.clone())
                                        plbls.append(pl.clone())
                    if coords:
                        idxs = list(range(len(coords)))
                        random.shuffle(idxs)
                        out_img = torch.zeros_like(x)
                        out_lbl = torch.zeros_like(label)
                        for new, old in enumerate(idxs):
                            dst_f, dst_y, dst_x = coords[new]
                            src_patch = pimgs[old]
                            src_lbl   = plbls[old]
                            h, w = src_patch.shape[-2:]
                            out_img[dst_f, dst_y:dst_y+h, dst_x:dst_x+w] = src_patch
                            out_lbl[dst_f, dst_y:dst_y+h, dst_x:dst_x+w] = src_lbl
                        # center-slice debug after
                        debug_after = out_img[center_idx].detach().cpu().numpy()
                        if not self._debug_saved:
                            # plt.imsave("augment_before.png", debug_before, cmap='gray')
                            # plt.imsave("augment_after.png", debug_after, cmap='gray')
                            self._debug_saved = True
                        return out_img.unsqueeze(0), out_lbl

        # If not 3D-shuffled, or it's 2D: just return
        return image, label.squeeze(0)

def is_pixel_in_ellipse(x, y, roi):
    cx, cy = roi[0] + roi[2]/2, roi[1] + roi[3]/2
    a, b = roi[2]/2, roi[3]/2
    return ((x-cx)**2)/(a*a) + ((y-cy)**2)/(b*b) <= 1


def create_image_and_labels_for_dataset(cfg, num_frames):
    """
    cfg: single dataset config dict OR list/tuple of such dicts.
    num_frames: number of frames to use per volume.
    """

    # 1) If cfg is a list/tuple of configs (e.g. test_set = [cfg2, cfg4])
    if isinstance(cfg, (list, tuple)):
        imgs_list = []
        lbls_list = []

        for single_cfg in cfg:
            imgs, lbls = create_image_and_labels_for_dataset(single_cfg, num_frames)
            imgs_list.append(imgs)
            lbls_list.append(lbls)

        first = imgs_list[0]
        if isinstance(first, np.ndarray):
            imgs_all = np.concatenate(imgs_list, axis=0)
            lbls_all = np.concatenate(lbls_list, axis=0)
            return imgs_all, lbls_all
        elif torch.is_tensor(first):
            imgs_all = torch.cat(imgs_list, dim=0)
            lbls_all = torch.cat(lbls_list, dim=0)
            return imgs_all, lbls_all
        else:
            # Fallback: just return Python lists
            return imgs_list, lbls_list

    # 2) Otherwise, cfg is a *single* config dict → original logic
    raw_dir = Path(cfg["dir"])

    set_folder = raw_dir.resolve()
    set_folder = os.path.expanduser(os.path.expandvars(str(set_folder)))
    if not os.path.isdir(set_folder):
        raise FileNotFoundError(f"Images folder not found or not a directory: {set_folder}")

    dicom_exts = ('.dcm', '.dicom')
    paths = []
    for root, _, files in os.walk(set_folder):
        paths += [os.path.join(root, f) for f in files if f.lower().endswith(dicom_exts)]

    if not paths:
        raise FileNotFoundError(f"No DICOM files (.dcm/.dicom) found under: {set_folder}")

    scale_x, scale_y = IMAGE_WIDTH / 1300.0, IMAGE_HEIGHT / 1300.0
    ox, oy = cfg['offset']
    rois = []
    for (x, y, w, h, lab_str) in cfg['original_rois']:
        rx = int((x + ox) * scale_x)
        ry = int((y + oy) * scale_y)
        rw = int(w * scale_x)
        rh = int(h * scale_y)
        lab_idx = next((i for i, n in global_label_names.items() if n == lab_str), 0)
        rois.append((rx, ry, rw, rh, lab_idx))

    imgs, lbls = [], []
    for fn in paths:
        ds = pydicom.dcmread(os.path.join(set_folder, fn))
        frames = ds.pixel_array
        n = min(frames.shape[0], num_frames)

        im_arr = np.zeros((n, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        lb_arr = np.zeros((n, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int64)

        for f in range(n):
            t = torch.tensor(frames[f].astype(np.float32)).unsqueeze(0)
            resized = TF.resize(t, (IMAGE_HEIGHT, IMAGE_WIDTH)).squeeze(0).numpy()
            im_arr[f] = resized

            for (x0, y0, w0, h0, lab) in rois:
                for px in range(x0, x0 + w0):
                    for py in range(y0, y0 + h0):
                        if is_pixel_in_ellipse(px, py, (x0, y0, w0, h0)):
                            lb_arr[f, py, px] = lab

        imgs.append(im_arr)
        lbls.append(lb_arr)

    return np.stack(imgs), np.stack(lbls)

# def is_pixel_in_ellipse(x, y, roi):
#     cx, cy = roi[0] + roi[2]/2, roi[1] + roi[3]/2
#     a, b = roi[2]/2, roi[3]/2
#     return ((x-cx)**2)/(a*a) + ((y-cy)**2)/(b*b) <= 1

    # # 1) If cfg is a list/tuple of configs (e.g. test_set = [cfg2, cfg4])
    # if isinstance(cfg, (list, tuple)):
    #     imgs_list = []
    #     lbls_list = []

    #     for single_cfg in cfg:
    #         imgs, lbls = create_image_and_labels_for_dataset(single_cfg, num_frames)
    #         imgs_list.append(imgs)
    #         lbls_list.append(lbls)

    #     # concatenate along batch dimension (axis 0)
    #     imgs_all = np.concatenate(imgs_list, axis=0)
    #     lbls_all = np.concatenate(lbls_list, axis=0)
    #     return imgs_all, lbls_all

    # # 2) Otherwise, cfg is a *single* config dict → original logic
    # raw_dir = Path(cfg["dir"])

    # set_folder = raw_dir.resolve()
    # # paths = [f for f in os.listdir(set_folder) if f.lower().endswith(('.dcm', '.dicom'))]
    # set_folder = os.path.expanduser(os.path.expandvars(str(set_folder)))
    # if not os.path.isdir(set_folder):
    #     raise FileNotFoundError(f"Images folder not found or not a directory: {set_folder}")

    # dicom_exts = ('.dcm', '.dicom')
    # paths = []
    # for root, _, files in os.walk(set_folder):
    #     paths += [os.path.join(root, f) for f in files if f.lower().endswith(dicom_exts)]

    # if not paths:
    #     raise FileNotFoundError(f"No DICOM files (.dcm/.dicom) found under: {set_folder}")

    # scale_x, scale_y = IMAGE_WIDTH / 1300.0, IMAGE_HEIGHT / 1300.0
    # ox, oy = cfg['offset']
    # rois = []
    # for (x, y, w, h, lab_str) in cfg['original_rois']:
    #     rx = int((x + ox) * scale_x)
    #     ry = int((y + oy) * scale_y)
    #     rw = int(w * scale_x)
    #     rh = int(h * scale_y)
    #     lab_idx = next((i for i, n in global_label_names.items() if n == lab_str), 0)
    #     rois.append((rx, ry, rw, rh, lab_idx))
    # imgs, lbls = [], []
    # for fn in paths:
    #     ds = pydicom.dcmread(os.path.join(set_folder, fn))
    #     frames = ds.pixel_array
    #     n = min(frames.shape[0], num_frames)
    #     im_arr = np.zeros((n, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    #     lb_arr = np.zeros((n, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int64)
    #     for f in range(n):
    #         t = torch.tensor(frames[f].astype(np.float32)).unsqueeze(0)
    #         resized = TF.resize(t, (IMAGE_HEIGHT, IMAGE_WIDTH)).squeeze(0).numpy()
    #         im_arr[f] = resized
    #         for (x0, y0, w0, h0, lab) in rois:
    #             for px in range(x0, x0 + w0):
    #                 for py in range(y0, y0 + h0):
    #                     if is_pixel_in_ellipse(px, py, (x0, y0, w0, h0)):
    #                         lb_arr[f, py, px] = lab
    #     imgs.append(im_arr)
    #     lbls.append(lb_arr)
    # return np.stack(imgs), np.stack(lbls)

def generate_cumulative_grid_sizes(num_images, num_grid_sizes=grid_size, cumulative_percentage=0.2):
    images_per_size = int(num_images * cumulative_percentage)
    grid_sizes = []
    for gs in range(1, num_grid_sizes+1):
        grid_sizes.extend([gs]*images_per_size)
    remaining = num_images - len(grid_sizes)
    if remaining>0:
        grid_sizes.extend(random.choices(range(1, num_grid_sizes+1), k=remaining))
    random.shuffle(grid_sizes)
    return grid_sizes

class BufferOverlayVisualizer(Callback):
    def __init__(self, color_map, label_names, save_dir, every_n_epochs=10, num_images=3, 
                 plot_softmax_per_class=True, plot_combined_softmax=True):
        super().__init__()
        self.color_map = color_map
        self.label_names = label_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.every_n_epochs = every_n_epochs
        self.num_images = num_images
        self.plot_softmax_per_class = plot_softmax_per_class
        self.plot_combined_softmax = plot_combined_softmax
        self.last_train_batch = None
        self.last_val_batch = None
        self.last_train_logits = None
        self.last_val_logits = None

    @staticmethod
    def get_center_slice(arr):
        arr = np.asarray(arr)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3:  # (F, H, W)
            center = arr.shape[0] // 2
            return arr[center]
        elif arr.ndim == 2:
            return arr
        else:
            raise RuntimeError(f"Unexpected input shape for get_center_slice: {arr.shape}")

    def _ensure_3d(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        while img.ndim > 3:
            img = img[0]
        if img.ndim == 2:
            img = img[None, ...]
        elif img.ndim == 1:
            img = img.reshape((1, img.shape[0], 1))
        return img

    def _color_mask(self, mask):
        mask = np.asarray(mask)
        color_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls, color in self.color_map.items():
            color_img[mask == cls] = color
        return color_img

    def _plot_overlay(self, image, mask, pred, epoch, split, idx):
        image = self._ensure_3d(image)
        mask = self._ensure_3d(mask)
        pred = self._ensure_3d(pred)
        num_frames = image.shape[0]
        if mask.shape[0] == 1 and num_frames > 1:
            mask = np.repeat(mask, num_frames, axis=0)
        if pred.shape[0] == 1 and num_frames > 1:
            pred = np.repeat(pred, num_frames, axis=0)
        fig, axs = plt.subplots(2, num_frames, figsize=(3 * num_frames, 6))
        for f in range(num_frames):
            axs[0, f].imshow(image[f], cmap='gray')
            axs[0, f].imshow(self._color_mask(mask[f]), alpha=0.5)
            axs[0, f].set_title(f'GT Frame {f}')
            axs[0, f].axis('off')
            axs[1, f].imshow(image[f], cmap='gray')
            axs[1, f].imshow(self._color_mask(pred[f]), alpha=0.5)
            axs[1, f].set_title(f'Pred Frame {f}')
            axs[1, f].axis('off')
        legend_elements = [
            Patch(facecolor=np.array(color) / 255.0, edgecolor='k', label=self.label_names.get(cls, str(cls)))
            for cls, color in self.color_map.items()
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        fn = self.save_dir / f"{split}_epoch{epoch:03d}_idx{idx}_overlay.png"
        plt.savefig(fn, bbox_inches='tight')
        plt.close(fig)

    def _plot_softmax_per_class(self, image, logits, epoch, split, idx, max_classes=None):

        image = self._ensure_3d(image)
        if isinstance(logits, torch.Tensor): 
            logits = logits.detach().cpu().numpy()
        # SAFETY PATCH:
        if logits.ndim == 3:  # (C, H, W) for 2D models
            logits = logits[:, None, :, :]
        num_classes, num_frames, H, W = logits.shape
        n_img_frames = image.shape[0]
        plot_frames = min(num_frames, n_img_frames)
        max_classes = min(num_classes, max_classes) if max_classes is not None else num_classes
        logits = logits - np.max(logits, axis=0, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        for f in range(plot_frames):
            fig, axs = plt.subplots(1, max_classes + 1, figsize=(3 * (max_classes + 1), 4))
            axs[0].imshow(image[f], cmap='gray')
            axs[0].set_title(f'Frame {f}\nInput')
            axs[0].axis('off')
            for c in range(max_classes):
                axs[c + 1].imshow(image[f], cmap='gray')
                # Overlay class color, alpha = softmax probability for that class
                color = np.array(self.color_map.get(c, (255, 255, 255))) / 255.0  # fallback: white
                color_img = np.ones((*probs[c, f].shape, 3)) * color  # shape (H, W, 3)
                axs[c + 1].imshow(color_img, alpha=probs[c, f], vmin=0, vmax=1)
                label = self.label_names[c] if c in self.label_names else f"Class {c}"
                axs[c + 1].set_title(f"{label}\nProb")
                axs[c + 1].axis('off')
            plt.suptitle(f"{split} epoch {epoch:03d} idx{idx} frame{f}: Per-Class Softmax Heatmaps")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fn = self.save_dir / f"{split}_epoch{epoch:03d}_idx{idx}_frame{f}_softmax.png"
            plt.savefig(fn, bbox_inches='tight')
            plt.close(fig)
        # Only warn if the model output is truly multi-frame (i.e., 5D)
        if logits.ndim == 5 and num_frames != n_img_frames:
            print(f"[WARN] logits num_frames ({num_frames}) != image frames ({n_img_frames}) in sample {idx} (using min={plot_frames})")
        # If 2D/2.5D model, skip warning: only central frame will be plotted
        elif logits.ndim == 4:
            pass


        # if num_frames != n_img_frames:
        #     print(f"[WARN] logits num_frames ({num_frames}) != image frames ({n_img_frames}) in sample {idx} (using min={plot_frames})")

    def _plot_combined_softmax_overlay(self, image, logits, epoch, split, idx):
        if isinstance(logits, torch.Tensor): logits = logits.detach().cpu().numpy()
        image = self._ensure_3d(image)
        # SAFETY PATCH:
        if logits.ndim == 3:  # (C, H, W) for 2D models
            logits = logits[:, None, :, :]
        num_classes, num_frames, H, W = logits.shape
        img = self.get_center_slice(image)
        logits_center = logits[:, num_frames // 2] if num_frames > 1 else logits[:, 0]
        exp_logits = np.exp(logits_center - np.max(logits_center, axis=0, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        pred_cls = np.argmax(probs, axis=0)
        pred_prob = np.max(probs, axis=0)
        overlay = np.zeros((*img.shape, 3), dtype=np.uint8)
        for cls, color in self.color_map.items():
            overlay[pred_cls == cls] = color
        alpha = pred_prob
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.imshow(overlay, alpha=0.5 * alpha)
        plt.title(f"{split} epoch {epoch:03d} idx{idx} Combined Softmax Overlay")
        legend_elements = [
            Patch(facecolor=np.array(color) / 255.0, edgecolor='k', label=self.label_names.get(cls, str(cls)))
            for cls, color in self.color_map.items()
        ]
        plt.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        fn = self.save_dir / f"{split}_epoch{epoch:03d}_idx{idx}_combined_softmax.png"
        plt.savefig(fn, bbox_inches='tight')
        plt.close()

    def _plot_2d_on_center_frame(self, image, mask, pred, epoch, split, idx):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        # (A) Original DICOM center frame
        center_img = self.get_center_slice(image)
        axs[0].imshow(center_img, cmap='gray')
        axs[0].set_title('Original DICOM Center')
        axs[0].axis('off')
        # (B) GT Center
        axs[1].imshow(center_img, cmap='gray')
        axs[1].imshow(self._color_mask(self.get_center_slice(mask)), alpha=0.5)
        axs[1].set_title('GT Center')
        axs[1].axis('off')
        # (C) Pred Center
        axs[2].imshow(center_img, cmap='gray')
        axs[2].imshow(self._color_mask(self.get_center_slice(pred)), alpha=0.5)
        axs[2].set_title('Pred Center')
        axs[2].axis('off')
        # Add legend
        legend_elements = [
            Patch(facecolor=np.array(color) / 255.0, edgecolor='k', label=self.label_names.get(cls, str(cls)))
            for cls, color in self.color_map.items()
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        fn = self.save_dir / f"{split}_epoch{epoch:03d}_idx{idx}_center_2d_overlay.png"
        plt.savefig(fn, bbox_inches='tight')
        plt.close(fig)


    def max_prob_projection_mask(self, logits):
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        # SAFETY PATCH:
        if logits.ndim == 3:  # (C, H, W) for 2D models
            logits = logits[:, None, :, :]
        probs = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs = probs / np.sum(probs, axis=0, keepdims=True)
        max_probs = np.max(probs, axis=1)
        summary_mask = np.argmax(max_probs, axis=0)
        return summary_mask

    def _plot_3d_summary_overlay_maxprob(self, image, logits, mask, epoch, split, idx):
        image = self._ensure_3d(image)
        mask = self._ensure_3d(mask)
        # SAFETY PATCH:
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if logits.ndim == 3:  # (C, H, W)
            logits = logits[:, None, :, :]
        summary_pred = self.max_prob_projection_mask(logits)
        from scipy.stats import mode
        mask_summary = mode(mask, axis=0).mode.squeeze()
        center_img = self.get_center_slice(image)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(center_img, cmap='gray')
        axs[0].imshow(self._color_mask(mask_summary), alpha=0.5)
        axs[0].set_title('GT Majority Summary')
        axs[0].axis('off')
        axs[1].imshow(center_img, cmap='gray')
        axs[1].imshow(self._color_mask(summary_pred), alpha=0.5)
        axs[1].set_title('Pred Max-Prob Summary')
        axs[1].axis('off')
        legend_elements = [
            Patch(facecolor=np.array(color) / 255.0, edgecolor='k', label=self.label_names.get(cls, str(cls)))
            for cls, color in self.color_map.items()
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1))
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        fn = self.save_dir / f"{split}_epoch{epoch:03d}_idx{idx}_summary_maxprob_overlay.png"
        plt.savefig(fn, bbox_inches='tight')
        plt.close(fig)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            imgs, lbls = batch
            # print(f"[DEBUG Visualizer] Train batch[0] shape: {imgs.shape}, batch_idx={batch_idx}")
            # Optionally, visualize imgs[0] here for debugging
            self.last_train_batch = batch
            with torch.no_grad():
                self.last_train_logits = pl_module(imgs.to(pl_module.device))


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self.last_val_batch = batch
            with torch.no_grad():
                self.last_val_logits = pl_module(batch[0].to(pl_module.device))

    def _visualize_from_buffer(self, batch, logits, epoch, split):
        if batch is None or logits is None:
            return
        imgs, lbls = batch
        N = min(self.num_images, imgs.shape[0])
        imgs = imgs[:N]
        lbls = lbls[:N]
        logits = logits[:N]
        preds = torch.argmax(logits, dim=1)
        for idx in range(N):
            image = imgs[idx].cpu().numpy()
            mask = lbls[idx].cpu().numpy()
            pred = preds[idx].cpu().numpy()
            logit = logits[idx].cpu()
            self._plot_overlay(image, mask, pred, epoch, split, idx)
            if pred.ndim == 2 or (pred.ndim == 3 and pred.shape[0] == 1):
                self._plot_2d_on_center_frame(image, mask, pred, epoch, split, idx)
            elif pred.ndim == 3 and pred.shape[0] > 1:
                self._plot_3d_summary_overlay_maxprob(image, logit, mask, epoch, split, idx)
            if self.plot_softmax_per_class:
                self._plot_softmax_per_class(image, logit, epoch, split, idx)
            if self.plot_combined_softmax:
                self._plot_combined_softmax_overlay(image, logit, epoch, split, idx)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # only every N epochs
        if epoch % self.every_n_epochs != 0:
            return

        dm = trainer.datamodule

        # ——— 1) get the “no‐grid” batch ———
        nogrid_loader = dm.train_nogrid_dataloader()
        imgs_ng, lbls_ng = next(iter(nogrid_loader))
        imgs_ng, lbls_ng = imgs_ng.to(pl_module.device), lbls_ng.to(pl_module.device)
        with torch.no_grad():
            logits_ng = pl_module(imgs_ng)
        preds_ng = torch.argmax(logits_ng, dim=1).cpu().numpy()

        # ——— 2) get the *actual* grid‐shuffled batch you already captured ———
        imgs_g, lbls_g = self.last_train_batch
        logits_g      = self.last_train_logits
        preds_g       = torch.argmax(logits_g, dim=1).cpu().numpy()
        imgs_g = imgs_g.cpu().numpy()
        lbls_g = lbls_g.cpu().numpy()

        # ——— now plot side‐by‐side ———
        N = min(self.num_images, imgs_ng.shape[0])
        for idx in range(N):
            # grab center‐slice for each
            img0 = self.get_center_slice(imgs_ng[idx])
            gt0  = self.get_center_slice(lbls_ng[idx].cpu().numpy())
            pr0  = self.get_center_slice(preds_ng[idx])
            
            img1 = self.get_center_slice(imgs_g[idx])
            gt1  = self.get_center_slice(lbls_g[idx])
            pr1  = self.get_center_slice(preds_g[idx])

            fig, axs = plt.subplots(2, 3, figsize=(12, 8))

            # row 0: no‐grid
            axs[0,0].imshow(img0, cmap='gray');           axs[0,0].set_title('Input (no grid)')
            axs[0,1].imshow(img0, cmap='gray'); axs[0,1].imshow(self._color_mask(gt0), alpha=0.5);  axs[0,1].set_title('GT (no grid)')
            axs[0,2].imshow(img0, cmap='gray'); axs[0,2].imshow(self._color_mask(pr0), alpha=0.5);  axs[0,2].set_title('Pred (no grid)')

            # row 1: actual grid‐shuffle
            axs[1,0].imshow(img1, cmap='gray');           axs[1,0].set_title('Input (grid shuffle)')
            axs[1,1].imshow(img1, cmap='gray'); axs[1,1].imshow(self._color_mask(gt1), alpha=0.5);  axs[1,1].set_title('GT (grid shuffle)')
            axs[1,2].imshow(img1, cmap='gray'); axs[1,2].imshow(self._color_mask(pr1), alpha=0.5);  axs[1,2].set_title('Pred (grid shuffle)')

            for ax in axs.ravel():
                ax.axis('off')

            plt.tight_layout()
            fn = self.save_dir / f"train_epoch{epoch:03d}_idx{idx}_compare.png"
            plt.savefig(fn, bbox_inches='tight')
            plt.close(fig)

        # then your full overlay logic
        self._visualize_from_buffer(self.last_train_batch, self.last_train_logits, epoch, 'train')

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        dm = trainer.datamodule

        # 1) No-grid validation batch
        nogrid_loader = dm.train_nogrid_dataloader()  # reuse train_nogrid for val debugging
        imgs_ng, lbls_ng = next(iter(nogrid_loader))
        imgs_ng, lbls_ng = imgs_ng.to(pl_module.device), lbls_ng.to(pl_module.device)
        with torch.no_grad():
            logits_ng = pl_module(imgs_ng)
        preds_ng = torch.argmax(logits_ng, dim=1).cpu().numpy()

        # 2) Actual grid-shuffled validation batch
        imgs_g, lbls_g = self.last_val_batch
        logits_g      = self.last_val_logits
        preds_g       = torch.argmax(logits_g, dim=1).cpu().numpy()
        imgs_g = imgs_g.cpu().numpy()
        lbls_g = lbls_g.cpu().numpy()

        # 3) Plot side-by-side
        N = min(self.num_images, imgs_ng.shape[0])
        for idx in range(N):
            img0 = self.get_center_slice(imgs_ng[idx])
            gt0  = self.get_center_slice(lbls_ng[idx].cpu().numpy())
            pr0  = self.get_center_slice(preds_ng[idx])

            img1 = self.get_center_slice(imgs_g[idx])
            gt1  = self.get_center_slice(lbls_g[idx])
            pr1  = self.get_center_slice(preds_g[idx])

            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            # Row 0: no-grid
            axs[0,0].imshow(img0, cmap='gray');           axs[0,0].set_title('Val Input (no grid)')
            axs[0,1].imshow(img0, cmap='gray'); axs[0,1].imshow(self._color_mask(gt0), alpha=0.5);  axs[0,1].set_title('Val GT (no grid)')
            axs[0,2].imshow(img0, cmap='gray'); axs[0,2].imshow(self._color_mask(pr0), alpha=0.5);  axs[0,2].set_title('Val Pred (no grid)')
            # Row 1: grid-shuffle
            axs[1,0].imshow(img1, cmap='gray');           axs[1,0].set_title('Val Input (grid shuffle)')
            axs[1,1].imshow(img1, cmap='gray'); axs[1,1].imshow(self._color_mask(gt1), alpha=0.5);  axs[1,1].set_title('Val GT (grid shuffle)')
            axs[1,2].imshow(img1, cmap='gray'); axs[1,2].imshow(self._color_mask(pr1), alpha=0.5);  axs[1,2].set_title('Val Pred (grid shuffle)')

            for ax in axs.ravel(): ax.axis('off')
            plt.tight_layout()
            fn = self.save_dir / f"val_epoch{epoch:03d}_idx{idx}_compare.png"
            plt.savefig(fn, bbox_inches='tight')
            plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 7. METRICS: Dice, Macro, Micro, Per-Class
# ─────────────────────────────────────────────────────────────
def per_class_metrics_3d(preds, labels, num_classes, smooth=1e-6, ignore_index=None):
    """
    Returns:
        dice_list, sens_list, spec_list: [per-class],
        macro_* (nanmean over foreground), micro_* (global)
    """
    preds = torch.argmax(preds, dim=1)

    if ignore_index is not None:
        mask = labels != ignore_index
    else:
        mask = torch.ones_like(labels, dtype=torch.bool)

    dice_list, sens_list, spec_list = [], [], []

    for c in range(num_classes):
        pred_c  = ((preds == c) & mask)
        label_c = ((labels == c) & mask)

        tp = (pred_c & label_c).sum().item()
        fp = (pred_c & (~label_c)).sum().item()
        fn = ((~pred_c) & label_c).sum().item()
        tn = ((~pred_c) & (~label_c)).sum().item()

        # --- Option-3 absent-GT handling ---
        gt_present = (label_c.sum().item() > 0)
        if (not gt_present) and (fp == 0):
            dice = float('nan')  # absent in GT and not predicted -> skip
            sens = float('nan')  # undefined without GT
        else:
            dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
            sens = (tp + smooth) / (tp + fn + smooth) if (tp + fn) > 0 else float('nan')

        spec = (tn + smooth) / (tn + fp + smooth) if (tn + fp) > 0 else float('nan')

        dice_list.append(dice)
        sens_list.append(sens)
        spec_list.append(spec)

    # Macro over foreground (exclude background=0); use nanmean to ignore skipped classes
    macro_dice = float(np.nanmean(dice_list[1:])) if len(dice_list) > 1 else float('nan')
    macro_sens = float(np.nanmean(sens_list[1:])) if len(sens_list) > 1 else float('nan')
    macro_spec = float(np.nanmean(spec_list[1:])) if len(spec_list) > 1 else float('nan')

    # Micro (global over foreground classes)
    tp_sum = sum([(((preds == c) & (labels == c) & mask).sum().item()) for c in range(1, num_classes)])
    fp_sum = sum([(((preds == c) & (labels != c) & mask).sum().item()) for c in range(1, num_classes)])
    fn_sum = sum([(((preds != c) & (labels == c) & mask).sum().item()) for c in range(1, num_classes)])
    tn_sum = (((preds == 0) & (labels == 0) & mask).sum().item())  # background

    denom_dice = (2*tp_sum + fp_sum + fn_sum)
    micro_dice = (2*tp_sum + smooth) / (denom_dice + smooth) if denom_dice > 0 else float('nan')
    micro_sens = (tp_sum + smooth) / (tp_sum + fn_sum + smooth) if (tp_sum + fn_sum) > 0 else float('nan')
    micro_spec = (tn_sum + smooth) / (tn_sum + fp_sum + smooth) if (tn_sum + fp_sum) > 0 else float('nan')

    return (dice_list, sens_list, spec_list,
            macro_dice, macro_sens, macro_spec,
            micro_dice, micro_sens, micro_spec)

# You already have per_class_metrics_3d
def per_class_metrics_2d(preds, labels, num_classes, smooth=1e-6, ignore_index=None):
    preds = torch.argmax(preds, dim=1)

    if ignore_index is not None:
        mask = labels != ignore_index
    else:
        mask = torch.ones_like(labels, dtype=torch.bool)

    dice_list, sens_list, spec_list = [], [], []

    for c in range(num_classes):
        pred_c  = ((preds == c) & mask)
        label_c = ((labels == c) & mask)

        tp = (pred_c & label_c).sum().item()
        fp = (pred_c & (~label_c)).sum().item()
        fn = ((~pred_c) & label_c).sum().item()
        tn = ((~pred_c) & (~label_c)).sum().item()

        # --- Option-3 handling for absent GT ---
        gt_present = (label_c.sum().item() > 0)
        if not gt_present and fp == 0:
            dice = float('nan')                     # skip absent+absent
            sens = float('nan')                     # undefined when GT absent
        else:
            dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
            sens = (tp + smooth) / (tp + fn + smooth) if (tp + fn) > 0 else float('nan')

        spec = (tn + smooth) / (tn + fp + smooth) if (tn + fp) > 0 else float('nan')

        dice_list.append(dice)
        sens_list.append(sens)
        spec_list.append(spec)

    # Use nanmean so skipped classes (NaN) don't inflate macro
    macro_dice = float(np.nanmean(dice_list[1:])) if len(dice_list) > 1 else float('nan')
    macro_sens = float(np.nanmean(sens_list[1:])) if len(sens_list) > 1 else float('nan')
    macro_spec = float(np.nanmean(spec_list[1:])) if len(spec_list) > 1 else float('nan')

    # Micro (unchanged): aggregate counts across foreground classes
    tp_sum = sum([(((preds == c) & (labels == c) & mask).sum().item()) for c in range(1, num_classes)])
    fp_sum = sum([(((preds == c) & (labels != c) & mask).sum().item()) for c in range(1, num_classes)])
    fn_sum = sum([(((preds != c) & (labels == c) & mask).sum().item()) for c in range(1, num_classes)])
    tn_sum = (((preds == 0) & (labels == 0) & mask).sum().item())

    micro_dice = (2*tp_sum + smooth) / (2*tp_sum + fp_sum + fn_sum + smooth) if (2*tp_sum + fp_sum + fn_sum) > 0 else float('nan')
    micro_sens = (tp_sum + smooth) / (tp_sum + fn_sum + smooth) if (tp_sum + fn_sum) > 0 else float('nan')
    micro_spec = (tn_sum + smooth) / (tn_sum + fp_sum + smooth) if (tn_sum + fp_sum) > 0 else float('nan')

    return (dice_list, sens_list, spec_list,
            macro_dice, macro_sens, macro_spec,
            micro_dice, micro_sens, micro_spec)


def macro_dice_loss(logits, labels, num_classes, ignore_index=255, smooth=1e-6):
    preds = torch.argmax(logits, dim=1)
    mask = (labels != ignore_index) if ignore_index is not None else torch.ones_like(labels, dtype=torch.bool)
    dice_list = []
    for c in range(1, num_classes):  # skip background
        pred_c = ((preds == c) & mask)
        label_c = ((labels == c) & mask)
        tp = (pred_c & label_c).sum().item()
        fp = (pred_c & (~label_c)).sum().item()
        fn = ((~pred_c) & label_c).sum().item()
        dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
        dice_list.append(dice)
    macro_dice = float(np.mean(dice_list)) if dice_list else 1.0
    return 1.0 - macro_dice

def ce_plus_macro_dice_loss(logits, labels, num_classes, ignore_index=255, smooth=1e-6):
    ce = F.cross_entropy(
        logits, labels,
        ignore_index=ignore_index
    )
    dice = macro_dice_loss(logits, labels, num_classes, ignore_index, smooth)
    return ce + 0.5*dice



# ─────────────────────────────────────────────────────────────
# 9. TRAIN/TEST UTILITIES
# ─────────────────────────────────────────────────────────────

class EpochProgressPrinter(Callback):
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        self.epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        percent = 100 * (trainer.current_epoch + 1) / trainer.max_epochs
        print(f"\n[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] {percent:.1f}% complete")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"    Epoch took {epoch_time:.2f} seconds.")

        # Compute moving average over last 3 epochs
        avg_epoch_time = np.mean(self.epoch_times[-3:]) if len(self.epoch_times) >= 3 else np.mean(self.epoch_times)
        completed = trainer.current_epoch + 1
        remaining = trainer.max_epochs - completed
        est_remain = remaining * avg_epoch_time
        print(f"    Estimated remaining time: ~{est_remain/60:.1f} minutes (moving avg).")

# ─────────────────────────────────────────────────────────────
# Loss registry + focal_plus_gradient_loss (PCCT-style)
# ─────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def _one_hot_ignore(labels: torch.Tensor, num_classes: int, ignore_index: int):
    """
    (B,F,H,W) -> onehot (B,C,F,H,W). Voxels with labels==ignore_index are zeroed out.
    """
    valid = (labels != ignore_index)
    safe = labels.clone()
    safe[~valid] = 0  # put a valid id where ignored; we'll zero it with the mask
    oh = F.one_hot(safe.long(), num_classes=num_classes).permute(0,4,1,2,3).float()
    oh = oh * valid.unsqueeze(1).float()
    return oh, valid

def _spatial_grad_3d(x: torch.Tensor):
    """
    Simple anisotropic TV-like gradient magnitude on (B,C,F,H,W).
    Returns per-voxel magnitude, same shape as x.
    """
    # pad by replication to keep same size
    def shift(dx, dim):
        pad = [0,0,0,0,0,0]
        pad[2*dim+1] = 1
        return F.pad(dx, tuple(pad[::-1]), mode="replicate")
    gx = (x - shift(x, dim=4)).abs()  # W
    gy = (x - shift(x, dim=3)).abs()  # H
    gz = (x - shift(x, dim=2)).abs()  # F (spectral/depth)
    return gx + gy + gz

def focal_plus_gradient_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
    alpha: float | None = None,
    gamma: float = 2.0,
    grad_weight: float = 1.0,
) -> torch.Tensor:
    """
    Multi-class focal CE + mean gradient error term (inspired by PCCT MD-UNet).
    - logits: (B,C,F,H,W)
    - labels: (B,F,H,W)
    - focal: standard γ-focusing (per-voxel), optional α weighting for imbalance
    - gradient term: L1 distance between spatial grads of probs vs one-hot GT
    """
    assert logits.ndim == 5 and labels.ndim == 4, "shapes must be (B,C,F,H,W) vs (B,F,H,W)"
    B, C, Fd, H, W = logits.shape

    # ---- Focal cross-entropy on valid voxels
    # CE expects (B,C,...) + long targets
    ce = F.cross_entropy(logits, labels.long(), ignore_index=ignore_index, reduction='none')  # (B,F,H,W)
    with torch.no_grad():
        valid = (labels != ignore_index).float()                                             # (B,F,H,W)
    # p_t = exp(-CE)
    pt = torch.exp(-ce)
    focal_term = (1 - pt) ** gamma
    if alpha is not None:
        # class-wise alpha weighting
        alpha_vec = torch.full((C,), float(alpha), device=logits.device)
        alpha_vec[0] = 1.0 - float(alpha)  # heuristic: down-weight background if alpha given
        alpha_w = alpha_vec[labels.clamp_min(0).clamp_max(C-1)]  # (B,F,H,W)
        focal_ce = (alpha_w * focal_term * ce * valid).sum() / (valid.sum().clamp_min(1))
    else:
        focal_ce = (focal_term * ce * valid).sum() / (valid.sum().clamp_min(1))

    # ---- Mean Gradient Error between probs and one-hot labels
    probs = torch.softmax(logits, dim=1)                            # (B,C,F,H,W)
    onehot, valid_mask = _one_hot_ignore(labels, num_classes, ignore_index)  # (B,C,F,H,W), (B,F,H,W)
    valid_mask = valid_mask.unsqueeze(1).float()                    # (B,1,F,H,W)

    gp = _spatial_grad_3d(probs) * valid_mask
    gt = _spatial_grad_3d(onehot) * valid_mask
    grad_err = (gp - gt).abs().mean()  # mean over all dims

    return focal_ce + grad_weight * grad_err

# ─────────────────────────────────────────────────────────────
# VMI preprocessing (toy linear synthesis from bins)
# ─────────────────────────────────────────────────────────────
def apply_vmi_preprocess(x: torch.Tensor, weights: list[list[float]], clip=(None, None), return_depth: int = 1):
    """
    x: (B,1,F,H,W) spectral-bin stack
    weights: KxF list mapping bins -> K monoenergetic VMIs
    Returns: (B,1,return_depth,H,W). If K==1 and return_depth>1, we tile along depth.
    """
    assert x.ndim == 5 and x.size(1) == 1, f"expected (B,1,F,H,W), got {tuple(x.shape)}"
    B, _, F, H, W = x.shape
    Wm = torch.tensor(weights, device=x.device, dtype=x.dtype)  # (K,F)
    assert Wm.shape[1] == F, f"weights expect F={F}, got {Wm.shape[1]}"
    x2 = x.view(B, F, H, W)                                    # (B,F,H,W)
    vmi = torch.einsum("kf,bfhw->bkhw", Wm, x2)                # (B,K,H,W)

    lo, hi = clip
    if lo is not None or hi is not None:
        vmi = torch.clamp(vmi, min=lo if lo is not None else vmi.min(), max=hi if hi is not None else vmi.max())

    # Back to (B,1,D,H,W)
    if vmi.shape[1] == 1:
        D = int(return_depth)
        if D <= 0: D = 1
        vmi = vmi[:, 0:1]                                      # (B,1,H,W)
        vmi = vmi.unsqueeze(2)                                 # (B,1,1,H,W)
        if D > 1:
            vmi = vmi.expand(B, 1, D, H, W).contiguous()
    else:
        # If K>1, stack K as depth directly
        vmi = vmi.unsqueeze(1)  # (B,1,K,H,W)
    return vmi

# Keep your existing ce_plus_macro_dice_loss import/def.
# Provide a thin adapter for nnU-Net-style Dice+CE from models.py (if you want to reuse here).
def dice_ce_loss_adapter(logits, labels, num_classes, ignore_index):
    from innovative3D.models import dice_ce_loss as _dice_ce
    return _dice_ce(logits, labels, num_classes, ignore_index=ignore_index)

# Registry so models can pull a loss by name from config.LOSS_NAME
LOSS_REGISTRY = {
    "ce_plus_macro_dice": lambda logits, labels, nc, ignore_index: ce_plus_macro_dice_loss(
        logits, labels, nc, ignore_index=ignore_index
    ),
    "focal_plus_gradient": lambda logits, labels, nc, ignore_index: focal_plus_gradient_loss(
        logits, labels, nc, ignore_index=ignore_index
    ),
    "dice_ce_nnunet": lambda logits, labels, nc, ignore_index: dice_ce_loss_adapter(
        logits, labels, nc, ignore_index
    ),
}
