# innovative3D/unified_optimizer.py
import torch
import pytorch_lightning as pl

def apply_unified_optimizer(
    lr: float = 1e-4,
    opt_cls = torch.optim.Adam,          # or torch.optim.AdamW
    betas=(0.9, 0.999),
    weight_decay: float = 0.0,           # e.g., 1e-2 for AdamW
    schedule: str = "constant",          # "constant" | "poly" | "cosine"
    poly_power: float = 0.9,             # used only if schedule == "poly"
    disable_lr_hooks: bool = True,
):
    import innovative3D.models as M

    def _cfg(self):
        # build optimizer
        if opt_cls in (torch.optim.Adam, torch.optim.AdamW):
            opt = opt_cls(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            opt = opt_cls(self.parameters(), lr=lr)

        # optional schedulers
        if schedule == "poly":
            # total training steps (fallback if PL can't estimate)
            T = getattr(self.trainer, "estimated_stepping_batches", None)
            if not T:
                steps = int(self.trainer.num_training_batches or 100)
                epochs = int(self.trainer.max_epochs or 100)
                T = steps * epochs

            def poly_lambda(step_idx: int):
                frac = max(0.0, 1.0 - step_idx / float(max(1, T)))
                return frac ** float(poly_power)

            sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=poly_lambda)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

            # (If you prefer epoch-based poly, change interval to "epoch" and
            #  switch to an epoch counter inside poly_lambda.)

        elif schedule == "cosine":
            # epoch-based cosine (simple, no warmup)
            T_max = int(self.trainer.max_epochs or 100)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

        return opt  # constant LR

    for _, cls in vars(M).items():
        if isinstance(cls, type) and issubclass(cls, pl.LightningModule):
            if not hasattr(cls, "_orig_configure_optimizers"):
                cls._orig_configure_optimizers = cls.configure_optimizers
            cls.configure_optimizers = _cfg
            if disable_lr_hooks:
                for hook in ("on_train_batch_start", "on_train_batch_end", "setup"):
                    if hasattr(cls, hook):
                        if not hasattr(cls, f"_orig_{hook}"):
                            setattr(cls, f"_orig_{hook}", getattr(cls, hook))
                        setattr(cls, hook, lambda *a, **k: None)
