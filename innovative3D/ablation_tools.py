# ablation_tools.py
import os, random, numpy as np, torch
import pytorch_lightning as pl

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); pl.seed_everything(seed, workers=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for reproducibility on CUDA

def run_seeds(model_name: str, BuilderOrClass, DataMod, ckpt_dir, seeds=(0,1,2)):
    results = []
    for s in seeds:
        set_all_seeds(int(s))
        model = BuilderOrClass() if callable(BuilderOrClass) else BuilderOrClass
        dm = DataMod()
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, devices=1, accelerator="gpu", deterministic=True)
        trainer.fit(model, dm)
        # keep best ckpt path if you use ModelCheckpoint
        path = os.path.join(str(ckpt_dir), f"seed{s}")
        os.makedirs(path, exist_ok=True)
        trainer.save_checkpoint(os.path.join(path, "last.ckpt"))
        results.append({"variant": model_name, "seed": s, "ckpt_dir": path})
    return results

# ablation_tools.py
import time, torch
def profile_model(make_model, input_shape=(1,1,96,96,96), warmup=5, iters=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model().to(device).eval()
    x = torch.randn(*input_shape, device=device)

    # Params
    params = sum(p.numel() for p in model.parameters())

    # FLOPs/MACs (best-effort): try thop then fvcore
    flops = None
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(x,), verbose=False)
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            flops = FlopCountAnalysis(model, x).total()
        except Exception:
            pass

    # Latency & peak VRAM
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    # warmup
    with torch.no_grad():
        for _ in range(warmup): _ = model(x)
    torch.cuda.synchronize() if device=="cuda" else None

    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters): _ = model(x)
    torch.cuda.synchronize() if device=="cuda" else None
    dt = (time.time() - t0) / iters

    peak = torch.cuda.max_memory_allocated() if device=="cuda" else 0
    return {"params": params, "flops": flops, "latency_s": dt, "peak_mem_bytes": peak}


# ablation_tools.py
import torch.nn.functional as F

@torch.no_grad()
def eval_with_perturbations(model, batch, intensity_gamma=0.9, noise_std=0.02, scale_hw=0.9):
    x, y = batch if isinstance(batch,(list,tuple)) else (batch["image"], batch["label"])
    x = x[0] if isinstance(x,(list,tuple)) else x  # (B,1,D,H,W)

    # intensity gamma
    x_g = x.clamp_min(0); x_g = x_g ** float(intensity_gamma)

    # gaussian noise
    n = torch.randn_like(x) * float(noise_std)
    x_n = (x + n).clamp(x.min(), x.max())

    # mild anisotropic scale on H,W
    B,C,D,H,W = x.shape
    x_s = F.interpolate(x, size=(D, int(H*scale_hw), int(W*scale_hw)), mode="trilinear", align_corners=False)
    x_s = F.interpolate(x_s, size=(D,H,W), mode="trilinear", align_corners=False)

    outs = []
    for xb in (x, x_g, x_n, x_s):
        logits = model(xb)
        outs.append(logits)
    return tuple(outs)  # (clean, gamma, noise, scale)
