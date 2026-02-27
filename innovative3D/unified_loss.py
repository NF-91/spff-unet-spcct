# innovative3D/patches/unified_loss.py
# Unify loss across all Lightning modules to ce_plus_macro_dice_loss
# without touching each class definition.

from __future__ import annotations
import types
import pytorch_lightning as pl

from innovative3D.config import IGNORE_INDEX, NUM_CLASSES  # fallback
from innovative3D.helpers import (
    ce_plus_macro_dice_loss,
    per_class_metrics_3d,
    per_class_metrics_2d,
)
# we reuse a couple of small utilities defined in models.py
from innovative3D.models import (
    _pick_first_if_seq,
    _canonicalize_targets_2d,
    _canonicalize_targets_3d,
)
import innovative3D.models as models_mod


def _get_num_classes(self) -> int:
    # Prefer hyperparams if available; fall back to config.NUM_CLASSES
    return int(getattr(getattr(self, "hparams", object()), "num_classes", NUM_CLASSES))


def _unified_shared_step(self: pl.LightningModule, batch, stage: str):
    """
    Stage ∈ {'train','val','test'}.
    Uses ce_plus_macro_dice_loss for loss and logs macro/micro metrics.
    Works for both 2D (B,C,H,W) and 3D (B,C,D,H,W) models.
    If a model returns deep-supervision outputs (tuple/list), we use the main head (index 0).
    """
    # ---- unpack batch ----
    if isinstance(batch, (list, tuple)):
        imgs, lbls = batch
    elif isinstance(batch, dict):
        imgs, lbls = batch.get("image"), batch.get("label")
    else:
        imgs, lbls = batch  # let it raise naturally if shape is wrong

    imgs = _pick_first_if_seq(imgs)
    lbls = _pick_first_if_seq(lbls)

    # ---- forward ----
    logits = self(imgs)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]  # main head

    # ---- canonicalize labels + compute loss/metrics ----
    nc = _get_num_classes(self)
    ign = int(getattr(getattr(self, "hparams", object()), "ignore_index", IGNORE_INDEX))

    if logits.ndim == 5:  # (B,C,D,H,W) → 3D
        tgt = _canonicalize_targets_3d(lbls).to(logits.device).long()
        loss = ce_plus_macro_dice_loss(logits, tgt, nc, ignore_index=ign)

        (dice_list, sens_list, spec_list,
         macro_dice, macro_sens, macro_spec,
         micro_dice, micro_sens, micro_spec) = per_class_metrics_3d(
            logits, tgt, nc, ignore_index=ign
        )
    elif logits.ndim == 4:  # (B,C,H,W) → 2D
        tgt = _canonicalize_targets_2d(lbls).to(logits.device).long()
        loss = ce_plus_macro_dice_loss(logits, tgt, nc, ignore_index=ign)

        (dice_list, sens_list, spec_list,
         macro_dice, macro_sens, macro_spec,
         micro_dice, micro_sens, micro_spec) = per_class_metrics_2d(
            logits, tgt, nc, ignore_index=ign
        )
    else:
        raise RuntimeError(f"Unexpected logits ndim {logits.ndim}; expected 4D or 5D.")

    # ---- logging (epoch-level, like BaseLitModel) ----
    self.log(f"{stage}_loss",       loss,       on_step=False, on_epoch=True,
             prog_bar=(stage == "train"), sync_dist=True)
    self.log(f"{stage}_macro_dice", macro_dice, on_step=False, on_epoch=True,
             prog_bar=(stage != "test"), sync_dist=True)
    self.log(f"{stage}_micro_dice", micro_dice, on_step=False, on_epoch=True,
             prog_bar=True, sync_dist=True)
    self.log(f"{stage}_macro_sens", macro_sens, on_step=False, on_epoch=True,
             prog_bar=True, sync_dist=True)
    self.log(f"{stage}_macro_spec", macro_spec, on_step=False, on_epoch=True,
             prog_bar=True, sync_dist=True)
    self.log(f"{stage}_micro_sens", micro_sens, on_step=False, on_epoch=True,
             prog_bar=True, sync_dist=True)
    self.log(f"{stage}_micro_spec", micro_spec, on_step=False, on_epoch=True,
             prog_bar=True, sync_dist=True)

    # (Optional) per-class logs can be noisy; keep off by default
    # for i, (d, s, sp) in enumerate(zip(dice_list, sens_list, spec_list)):
    #     self.log(f"{stage}_dice_class_{i}", d,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #     self.log(f"{stage}_sens_class_{i}", s,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #     self.log(f"{stage}_spec_class_{i}", sp, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    return loss


def _training_step(self, batch, batch_idx):
    return _unified_shared_step(self, batch, "train")


def _validation_step(self, batch, batch_idx):
    return _unified_shared_step(self, batch, "val")


def _test_step(self, batch, batch_idx):
    return _unified_shared_step(self, batch, "test")


def apply_unified_loss():
    """
    Find every LightningModule class in innovative3D.models and monkey-patch its
    training/validation/test steps to the unified loss/metrics defined above.
    Does NOT touch optimizers/schedulers/forwards.
    Call this ONCE at startup (e.g., in train.py) BEFORE building your models.
    """
    patched = []

    for name in dir(models_mod):
        obj = getattr(models_mod, name)
        if not isinstance(obj, type):
            continue
        if not issubclass(obj, pl.LightningModule):
            continue

        # Skip BaseLitModel itself (it already uses the helper)
        if name == "BaseLitModel":
            continue

        # Bind our functions as methods on the class (monkey-patch)
        obj.training_step   = _training_step
        obj.validation_step = _validation_step
        obj.test_step       = _test_step
        patched.append(name)

    # Optional: small runtime note (visible if someone imports and prints)
    if len(patched) == 0:
        print("[unified_loss] No LightningModule classes found to patch.")
    else:
        print(f"[unified_loss] Patched Lightning steps for: {', '.join(sorted(patched))}")
