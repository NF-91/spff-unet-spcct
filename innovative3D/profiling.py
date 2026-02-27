# innovative3D/profiling.py
from __future__ import annotations
import time, csv
from pathlib import Path
import torch
import numpy as np

from thop import profile as thop_profile
from innovative3D.config import VARIANTS, CHECKPOINT_DIR, NUM_FRAMES

# Import your ChannelLastLayerNorm3D so we can tell THOP how to handle it
from innovative3D.models import ChannelLastLayerNorm3D

def _zero_ops(m, x, y):
    # Treat as 0-cost (safe + consistent across BN/GN/LN ablations)
    if not hasattr(m, "total_ops"):
        m.total_ops = torch.zeros(1, dtype=torch.float64)
    m.total_ops += torch.zeros(1, dtype=torch.float64)

CUSTOM_OPS = {
    ChannelLastLayerNorm3D: _zero_ops,   # avoid THOP unknown LayerNorm issues
    torch.nn.Identity: _zero_ops,
}

def _bench_latency(model: torch.nn.Module, dummy: torch.Tensor, iters=50, warmup=20):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return (dt / iters) * 1000.0  # ms

def profile_all():
    out_dir = Path(CHECKPOINT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "model_profile.csv"

    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input that matches your pipeline: B=1, C=1, F=5, H=W=512
    dummy = torch.randn(1, 1, NUM_FRAMES, 512, 512, device=device, dtype=torch.float32)

    for name, BuilderOrClass, DataMod, _ in VARIANTS:
        try:
            # Build LightningModule or factory → your train code handles both; reuse that helper:
            from innovative3D.train import _build_lit  # same helper you use at train time
            lit = _build_lit(BuilderOrClass)
            lit.eval().to(device)

            # Prefer the pure nn.Module if available
            core = getattr(lit, "model", lit).eval().to(device)

            # THOP MACs/Params
            with torch.no_grad():
                macs, params = thop_profile(core, inputs=(dummy,), custom_ops=CUSTOM_OPS, verbose=False)
            flops = macs * 2  # 1 MAC = 2 FLOPs convention

            # Latency + peak memory
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            lat_ms = _bench_latency(core, dummy, iters=50, warmup=20)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else np.nan

            rows.append({
                "model": name,
                "params_M": params / 1e6,
                "macs_G": macs / 1e9,
                "flops_G": flops / 1e9,
                "latency_ms_b1": lat_ms,
                "peak_mem_MB": peak_mem,
            })
            print(f"[PROFILE] {name}: params={params/1e6:.2f}M | MACs={macs/1e9:.2f}G | "
                  f"FLOPs={flops/1e9:.2f}G | latency={lat_ms:.1f} ms | peak={peak_mem:.0f} MB")
        except Exception as e:
            print(f"[PROFILE][WARN] {name} failed: {e}")
            rows.append({"model": name, "error": str(e)})

    # write CSV
    if rows:
        fieldnames = sorted(set().union(*[r.keys() for r in rows]))
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[PROFILE] Saved → {out_csv}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False  # reproducible latency
    profile_all()
