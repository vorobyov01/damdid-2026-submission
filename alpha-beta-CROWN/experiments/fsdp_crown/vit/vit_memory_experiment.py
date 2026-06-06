"""FSDP memory experiment on a Vision Transformer.

Builds a small CIFAR-style ViT (BatchNorm-based, the same primitive layout as
the VNN-COMP'23 ``pgd_2_3_16`` benchmark) but with configurable token width
and depth, wraps it in a ``BoundedModule`` and runs ``compute_bounds(CROWN)``
on a single image. The point is to expose the regime where the linear weights
of the model dominate GPU memory: there ``FSDP`` shards them and pays back
its AllGather overhead.

Run:
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 vit_memory_experiment.py

Notes:
    - Only ``compute_bounds`` is measured — no BaB, no alpha-CROWN. That mirrors
      the existing FSDP memory experiment for plain MLPs (Exp. 4 in the thesis).
    - The model is randomly initialized; we do not need a trained ViT to compare
      memory between single-GPU and FSDP runs.
"""
import argparse
import copy
import gc
import json
import math
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../auto_LiRPA"))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.fsdp_utils import fsdp_shard_bounded_module, fsdp_free_gathered_weights


# ---------------------------------------------------------------------------
# A minimal ViT block using BatchNorm1d (over token dim) instead of LayerNorm.
# This mirrors the op layout of pgd_2_3_16 (which already runs through
# auto_LiRPA after the recent JIT batch-dim fixes).
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        h = self.heads
        d = self.head_dim
        q = self.q(x).reshape(B, N, h, d).transpose(1, 2)  # [B, h, N, d]
        k = self.k(x).reshape(B, N, h, d).transpose(1, 2)
        v = self.v(x).reshape(B, N, h, d).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v)                          # [B, h, N, d]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    """ViT block without normalization layers.

    auto_LiRPA gets confused when a Reshape collapses the batch dim into
    the token dim (e.g. ``[B, N, D] -> [B*N, D]`` inside a BN/LN), so for
    the memory experiment we skip normalization entirely. The resulting
    ops (Linear, MatMul, Softmax, ReLU, Reshape over heads only,
    Transpose) are all well supported.
    """
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class TinyViT(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 dim: int = 128, depth: int = 4, heads: int = 4,
                 num_classes: int = 10):
        super().__init__()
        assert image_size % patch_size == 0
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.patch_embed(x)                     # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)            # [B, N, D]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)                           # global average pool
        return self.head(x)


# ---------------------------------------------------------------------------
# Memory measurement (mirrors memory_experiment.py for plain MLPs)
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_bounds(model, x, eps, dev, use_fsdp=False, ws=1, rank=0, method="CROWN"):
    gc.collect()
    torch.cuda.empty_cache()

    x_L = (x - eps).clamp(0, 1)
    x_U = (x + eps).clamp(0, 1)
    dummy = torch.empty_like(x)

    lirpa = BoundedModule(copy.deepcopy(model), dummy, device=dev)

    if use_fsdp and ws > 1:
        fsdp_shard_bounded_module(lirpa, ws, rank, dummy_input=dummy)

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_L.clone(), x_U=x_U.clone())
    bx = BoundedTensor(x.clone(), ptb)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    base_mb = torch.cuda.memory_allocated(dev) / (1024 ** 2)

    lb, ub = lirpa.compute_bounds(x=(bx,), method=method)

    peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    if use_fsdp:
        fsdp_free_gathered_weights(lirpa)

    del lirpa
    gc.collect()
    torch.cuda.empty_cache()
    return lb.detach().cpu(), ub.detach().cpu(), peak_mb, base_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--method", type=str, default="CROWN")
    args = ap.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    if ws > 1:
        dist.init_process_group("nccl", rank=rank, world_size=ws)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)

    torch.manual_seed(0)
    model = TinyViT(image_size=args.image_size, patch_size=args.patch,
                    dim=args.dim, depth=args.depth, heads=args.heads)
    model.eval()

    n_params = count_params(model)
    weight_mb = n_params * 4 / (2 ** 20)
    if rank == 0:
        print(f"Model: TinyViT dim={args.dim} depth={args.depth} heads={args.heads} "
              f"patch={args.patch} -> {n_params / 1e6:.2f} M params ({weight_mb:.1f} MB fp32)")

    x = torch.randn(1, 3, args.image_size, args.image_size, device=dev)
    x = (x - x.min()) / (x.max() - x.min())          # [0, 1]

    # FSDP
    if ws > 1:
        lb_fsdp, ub_fsdp, peak_fsdp, base_fsdp = measure_bounds(
            model, x, args.eps, dev, use_fsdp=True, ws=ws, rank=rank, method=args.method)
        if rank == 0:
            print(f"[FSDP={ws}] base={base_fsdp:.1f} MB  peak={peak_fsdp:.1f} MB  "
                  f"lb[0:5]={lb_fsdp.flatten()[:5].tolist()}")

    # Single GPU (rank 0 only)
    if rank == 0:
        lb_ref, ub_ref, peak_ref, base_ref = measure_bounds(
            model, x, args.eps, dev, use_fsdp=False, ws=1, rank=0, method=args.method)
        print(f"[Single]   base={base_ref:.1f} MB  peak={peak_ref:.1f} MB  "
              f"lb[0:5]={lb_ref.flatten()[:5].tolist()}")

        if ws > 1:
            print(json.dumps({
                "model": f"TinyViT-{args.dim}x{args.depth}",
                "params_M": n_params / 1e6,
                "weights_MB": weight_mb,
                "single": {"base_MB": base_ref, "peak_MB": peak_ref},
                "fsdp_2": {"base_MB": base_fsdp, "peak_MB": peak_fsdp},
                "peak_savings_pct": 100.0 * (peak_ref - peak_fsdp) / max(peak_ref, 1e-9),
                "base_savings_pct": 100.0 * (base_ref - base_fsdp) / max(base_ref, 1e-9),
            }, indent=2))

    if ws > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
