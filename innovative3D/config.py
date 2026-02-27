# config.py
import os
from pathlib import Path
from importlib import import_module
import inspect
# ─────────────────────────────────────────────────────────────
# 0) RUNTIME ENV
# ─────────────────────────────────────────────────────────────
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ─────────────────────────────────────────────────────────────
# 1) PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/nadine/datasets/Fivedatasets")

# Default checkpoints location (can be overridden by env below)
_PRIMARY_CKPT_DIR = BASE_DIR / "final checkpoints" / "trial" #"checkpoints_innovative_10Grids_Novel"
_PRIMARY_CKPT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_HEIGHT, IMAGE_WIDTH = 512, 512
NUM_FRAMES = 5
NUM_CLASSES = 13
FINAL_EPOCHS = 200
BEST_LR = 1e-4
IGNORE_INDEX = 255
BATCH_SIZE = 1
NUM_WORKERS = 16
num_workers = NUM_WORKERS  # alias some modules expect

grid_size = 10

SEEDS   =  [42,123,999]
# ─────────────────────────────────────────────────────────────
# 2) DATA ROOTS & LABEL SPACE
# ─────────────────────────────────────────────────────────────
DATA_DIRS = {
    f"set{i+1}": BASE_DIR / d for i, d in enumerate(
        ["firstscan", "filtered", "filtered2", "filtered3", "filtered4"])#, "azzabi792", "azzabi766"])
    
}

global_label_names = {
    0:"BG",1:"HA800",2:"HA400",3:"HA200",4:"HA100",5:"Lung",6:"Liver",7:"Adipose",
    8:"Water",9:"I15",10:"I10",11:"I5",12:"HA50"#, 13:"Muscle"
}
label_colors = {
    0:(0,0,0),1:(255,0,0),2:(255,127,0),3:(255,255,0),4:(0,255,0),5:(0,255,255),
    6:(0,0,255),7:(139,69,19),8:(255,255,255),9:(255,0,255),10:(128,0,128),
    11:(0,128,128),12:(128,128,0)#,13:(0,255,128)
}

# ─────────────────────────────────────────────────────────────
# 3) DATASET CONFIGS
# ─────────────────────────────────────────────────────────────
dataset_configs = [
    {
      "name":"set1",
      "dir": DATA_DIRS["set1"],
      "original_rois":[
        (652,378,186,182,"HA800"),(880,498,186,182,"HA400"),
        (934,750,186,182,"HA200"),(761,950,186,182,"HA100"),
        (513,934,186,182,"Lung"),  (349,727,186,182,"Liver"),
        (416,479,186,182,"Adipose"),(648,670,186,182,"Water")
      ],
      "offset":(-95,-90)
    },
    {
      "name":"set2",
      "dir": DATA_DIRS["set2"],
      "original_rois":[
        (342,569,188,186,"HA800"),(532,385,188,186,"HA100"),
        (786,413,188,186,"Lung"),  (928,637,188,186,"HA200"),
        (840,881,188,186,"Liver"),(594,969,188,186,"HA400"),
        (378,827,188,186,"Adipose"),(631,667,188,186,"Water")
      ],
      "offset":(-95,-90)
    },
    {
      "name":"set3",
      "dir": DATA_DIRS["set3"],
      "original_rois":[
        (828,441,182,180,"HA100"),(930,679,182,180,"HA200"),
        (808,913,182,180,"HA400"),(555,956,182,180,"HA800"),
        (358,784,182,180,"Adipose"),(376,529,182,180,"Lung"),
        (578,375,182,180,"Liver"),(628,668,182,180,"Water")
      ],
      "offset":(-95,-90)
    },
    {
      "name":"set4",
      "dir": DATA_DIRS["set4"],
      "original_rois":[
        (773,409,184,188,"HA800"),(922,620,184,188,"I15"),
        (845,867,184,188,"I10"),(606,964,184,188,"I5"),
        (377,835,184,188,"HA100"),(339,582,184,188,"HA200"),
        (516,390,184,188,"HA400"),(627,660,184,188,"Water")
      ],
      "offset":(-95,-90)
    },
    {
      "name":"set5",
      "dir": DATA_DIRS["set5"],
      "original_rois":[
        (523,388,186,184,"HA800"),(778,409,186,184,"I5"),
        (921,625,186,184,"HA50"),(844,878,186,184,"HA400"),
        (598,965,186,184,"I10"),(373,829,186,184,"HA200"),
        (341,575,186,184,"I15"),(631,666,186,184,"HA100")
      ],
      "offset":(-95,-90)
    },
  
]


# Which datasets go where
TRAIN_INDICES = [0, 1,2, 4]
TEST_INDICES  = [3]

# Build the train and test configs
trainval_sets = [dataset_configs[i] for i in TRAIN_INDICES]

# IMPORTANT: must be named exactly 'test_set' (singular)
test_set = [dataset_configs[i] for i in TEST_INDICES]


# ─────────────────────────────────────────────────────────────
# 4) DATA MODULES
# ─────────────────────────────────────────────────────────────

_DM2D = _DM3D = None

def MultiDicomDataModule2D(*args, **kwargs):
    global _DM2D
    if _DM2D is None:
        from .datasets import MultiDicomDataModule2D as _DM2D  # noqa
    return _DM2D(*args, **kwargs)

def MultiDicomDataModule3D(*args, **kwargs):
    global _DM3D
    if _DM3D is None:
        from .datasets import MultiDicomDataModule3D as _DM3D  # noqa
    return _DM3D(*args, **kwargs)


# ─────────────────────────────────────────────────────────────
# 5) LAZY BUILDERS (avoid circular imports with models.py)
# ─────────────────────────────────────────────────────────────
def build_from_models(func_name: str, **fixed_kwargs):
    def _factory():
        mod = import_module("innovative3D.models")
        fn = getattr(mod, func_name, None)
        if fn is None:
            raise ImportError(f"[config] {func_name} not found in innovative3D.models")
        return fn(**fixed_kwargs)
    return _factory


def build_class(class_name: str, **ctor_kwargs):
    def _factory():
        mod = import_module("innovative3D.models")
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise ImportError(f"[config] {class_name} not found in innovative3D.models")

        # Pass only kwargs the class __init__ accepts.
        # If it has **kwargs, pass everything through unchanged.
        try:
            sig = inspect.signature(cls.__init__)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                             for p in sig.parameters.values())
            if has_var_kw:
                filtered = dict(ctor_kwargs)  # class accepts arbitrary kwargs
            else:
                allowed = {name for name, p in sig.parameters.items() if name != "self"}
                filtered = {k: v for k, v in ctor_kwargs.items() if k in allowed}
        except (TypeError, ValueError):
            # If signature isn't inspectable, be permissive
            filtered = dict(ctor_kwargs)

        return cls(**filtered)
    return _factory

# replace your current build_nnunet_wrap/_factory with this
def build_nnunet_wrap(class_name: str, **nn_kwargs):
    def _factory():
        import inspect
        import pytorch_lightning as pl
        from innovative3D import models as M

        ModelCls = getattr(M, class_name, None)
        if ModelCls is None:
            raise ImportError(f"[config] {class_name} not found in innovative3D.models")

        def _allowed_kwargs(cls, all_kwargs):
            sig = inspect.signature(cls.__init__)
            # keep kwargs if cls accepts **kwargs; else filter to named params
            has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            if has_var_kw:
                return dict(all_kwargs)
            names = {n for n, p in sig.parameters.items()
                     if n != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
            return {k: v for k, v in all_kwargs.items() if k in names}

        # Case 1: class is already a LightningModule → instantiate directly, but only with its allowed kwargs
        if inspect.isclass(ModelCls) and issubclass(ModelCls, pl.LightningModule):
            return ModelCls(**_allowed_kwargs(ModelCls, nn_kwargs))

        # Case 2: backbone nn.Module → wrap with LitNnUNetStyle
        LitWrap = getattr(M, "LitNnUNetStyle", None)
        if LitWrap is None:
            raise ImportError("[config] LitNnUNetStyle not found in innovative3D.models")

        # split kwargs between wrapper and backbone
        wrap_kwargs     = _allowed_kwargs(LitWrap, nn_kwargs)
        backbone_kwargs = _allowed_kwargs(ModelCls, nn_kwargs)

        # ensure num_classes reaches the wrapper
        if "num_classes" in nn_kwargs and "num_classes" not in wrap_kwargs:
            wrap_kwargs["num_classes"] = nn_kwargs["num_classes"]

        # MINIMAL FIX: avoid passing num_classes twice (positional + kw)
        backbone_kwargs.pop("num_classes", None)

        return LitWrap(lambda nc: ModelCls(nc, **backbone_kwargs), **wrap_kwargs)

    return _factory



# ── Training recipe selection
LOSS_NAME = "ce_plus_macro_dice"  # {"ce_plus_macro_dice", "focal_plus_gradient", "dice_ce_nnunet"}

# Focal+Gradient settings (only used when LOSS_NAME == "focal_plus_gradient")
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
GRAD_WEIGHT = 1.0

# ── VMI preprocessing (optional)
USE_VMI = False
VMI_MODE = "linear"
VMI_WEIGHTS = [[0.10, 0.20, 0.40, 0.20, 0.10]]  # shape KxF
VMI_CLIP = (None, None)
VMI_RETURN_DEPTH = 1
VMI_DENOISE = {"enabled": False, "method": "median3d", "kernel": (1, 3, 3)}

# ─────────────────────────────────────────────────────────────
# 6) VARIANTS: REGISTER FACTORIES   (name, builder, DataModuleClass, ckpt_dir)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow environment to override where checkpoints/logs go; fallback to primary
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(_PRIMARY_CKPT_DIR))).resolve()
LOG_DIR = Path(os.getenv("LOG_DIR", str(_PROJECT_ROOT / "runs"))).resolve()

# Back-compat alias some modules expect
CKPT_DIR = CHECKPOINT_DIR

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_ctor(x):
    """Map 'ResUNetPP3D' -> innovative3D.models.ResUNetPP3D if present; else return unchanged."""
    if isinstance(x, str):
        from innovative3D import models as _M
        return getattr(_M, x, x)
    return x

# ─────────────────────────────────────────────────────────────
# 7) VARIANTS REGISTRY
# ─────────────────────────────────────────────────────────────
VARIANTS = []

def _add_variant(name, builder_or_class, dm_cls, ckpt_dir):
    """
    name: str
    builder_or_class: callable that returns a LightningModule (preferred) or the class itself
    dm_cls: DataModule class or callable
    ckpt_dir: Path-like where checkpoints/logs for this variant live
    """
    VARIANTS.append((name, builder_or_class, dm_cls, Path(ckpt_dir)))


def make_cicek_depth_adapter_sgd_wce():
    from innovative3D.models import LitCicek3DUNet_DepthAdapter_Published
    return LitCicek3DUNet_DepthAdapter_Published(
        num_classes=NUM_CLASSES,
        # ------- SGD like the paper -------
        lr=1e-2,                 # typical starting LR for SGD on 3D U-Net
        momentum=0.99,
        nesterov=False,
        weight_decay=0.0,
        # ------- Weighted softmax CE -------
        ignore_index=255,        # unlabeled → ignored
        class_weights=None,      # or e.g., [0.5, 1.5, 2.0, ...]  # len == NUM_CLASSES
        voxel_weight_key=None,   # set to "weight" if your DataModule yields per-voxel weights
        # Dice OFF by default (paper used CE)
        ce_weight=1.0,
        dice_weight=0.0,
        # arch
        use_bn=True,
        target_depth=16,
        include_bg_in_dice=False,
    )


_add_variant(
    "3DUNet",
    make_cicek_depth_adapter_sgd_wce,
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "3DUNet" #"Cicek2016_SGD_WCE"
)




_add_variant("UNETR",
    build_class("LitUNETR_Published",
        num_classes=NUM_CLASSES,
        img_size=(96, 96, 96),
        in_channels=1,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        # AdamW + cosine + warmup (recommended)
        lr=1e-4, weight_decay=1e-2,
        use_cosine_lr=True, warmup_epochs=5,  # <-- add these flags in your Lit class
        # Loss
        use_ce_alongside_dice=True, ce_weight=0.5,
        include_bg_in_dice=False,
        ignore_index=IGNORE_INDEX, 
    ),
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "UNETR" #"UNETR_Published"

)


# -----------------------
# R2U-Net 3D — Alom → R2U3D (Kadia et al.)# bad
# -----------------------
# R2U-Net 3D — Kadia et al. (R2U3D): Adam 1e-3, Dice-only, t=2
_add_variant("R2UNet3D",
    build_class("LitR2UNet3D_Published",
        num_classes=NUM_CLASSES,             # binary head (paper-style). Use C if multi-class.
        in_channels=1,
        base_features=16,
        t=2,
        lr=1e-3, weight_decay=0.0,  # WD=0 for faithfulness
        ignore_index=IGNORE_INDEX,
        include_bg_in_dice=False,
        ce_weight=0.0, dice_weight=1.0, 
        pad_multiple=16
    ),
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "R2UNet3D" #"R2UNet3D_Kadia2021"
) 




_add_variant("SwinUNETR",
    build_class("LitSwinUNETR_Published",
        num_classes=NUM_CLASSES,
        img_size=(64, 64, 64),            # smaller fixed working crop
        in_channels=1,
        feature_size=12,   #48                 # must be divisible by 12 (OK)
        depths=(1, 1, 1, 1),        #(2,2,2,2),      # one block per stage 
        num_heads=(1, 2, 4, 8),  #(3,6,12,24),          # fewer heads → linear memory cut
        window_size=(2, 2, 2),     #(4, 4, 4),             # smallest windows = lowest attn memory
        mlp_ratio=2.0,       #4.0,               # halves MLP activations
        drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0,
        norm_name="instance",
        use_checkpoint=True,               # gradient checkpointing
        lr=8e-4, weight_decay=1e-2, warmup_epochs=5,
        use_ce_alongside_dice=True, ce_weight=0.5,
        ignore_index=IGNORE_INDEX,
        include_bg_in_dice=False
    ),
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "SwinUNETR" #"SwinUNETR_Published"
)



# ResUNet++ 3D — Jha (2019) style where specified; 3D is an adaptation
_add_variant("ResUNet++",
    build_class("LitResUNetPP3D_Published",
        num_classes=NUM_CLASSES,
        in_channels=1,
        base_features=16,                 # impl fallback (paper doesn't fix channels)
        include_bg_in_dice=False,         # impl default
        ce_weight=0.5, dice_weight=0.5,   # Dice+CE common in Jha-style setups
        lr=1e-4, weight_decay=1e-5,       # Adam 1e-4 to mirror Jha; wd is impl default
        ignore_index=IGNORE_INDEX,
        pad_multiple=16
    ),
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "ResUNet++"#"ResUNetPP3D_Jha2019Style"
)



#####################################################################################################################################
# Common kwargs for SPCT-family classes (filtered per class by build_class)
_SPCT_COMMON = dict(
    num_classes=NUM_CLASSES,
    lr=BEST_LR,
    base=32,           # keep compute matched
    ksd=3,             # spectral kernel depth
    use_se=True,
    use_specse=True,
    use_spatial=False,
    use_skip_gate=False,
)


# 1) Main novel model
_add_variant(
    "SPFF-UNet", #0.64 # "SPCT_EFiLM_FourierGate_main"
    build_class("LitSPCT_EFiLM_FourierGate", **_SPCT_COMMON),
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "SPFF-UNet", #EFG_SP_UNet",
)



# 2) Ablation: remove FourierGate (keep SEspec + EFiLM)
_add_variant(
    "E_SP_UNet",
    build_class("LitSPCT_EnergyFiLM", **_SPCT_COMMON),
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "E_SP_UNet", 
)



# 3) Ablation: remove EFiLM (keep FourierGate only)
_add_variant(
    "FG_SP_UNet", 
    build_class("LitSPCT_FourierGate", **_SPCT_COMMON),
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "FG_SP_UNet",
)

# 4) Ablation: SE-only control
_add_variant(
    "SP_UNet", 
    build_class("LitSPCT_SEspec", num_classes=NUM_CLASSES, lr=BEST_LR),
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "SP_UNet", 
)


# Optional: Plain core (turn SE/spec-SE OFF too) for a “vanilla” baseline
_plaincore_kwargs = {
    **_SPCT_COMMON,              # start from the shared defaults
    "base": _SPCT_COMMON.get("base", 32),   # keep compute matched to main
    "ksd":  _SPCT_COMMON.get("ksd", 3),
    # override spectral & SE knobs to get a plain control:
    "use_se": False,
    "use_specse": False,
    "use_spatial": False,
    "use_skip_gate": False,
}

_add_variant(
    "PlainCore_UNet",
    build_class("LitSPCT_ControlUNet", **_plaincore_kwargs),
    MultiDicomDataModule3D,
    CHECKPOINT_DIR / "PlainCore_UNet",
)

# ─────────────────────────────────────────────────────────────
# 7) Convenience
# ─────────────────────────────────────────────────────────────
VARIANT_NAMES = [v[0] for v in VARIANTS]
SELECTED_VARIANT = os.getenv("INNOVATIVE3D_VARIANT")  


