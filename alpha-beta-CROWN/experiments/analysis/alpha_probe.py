#!/usr/bin/env python3
"""Where does BaB memory actually go: alpha tensors, betas, weights or bounds?

Wraps abcrown exactly like run_abcrown_fsdp.py, but instruments
LiRPANet.update_bounds so that after every BaB round the live GPU tensors are
attributed to categories.  The snapshot with the largest alpha footprint is
reported together with the process peak, which turns the paper's qualitative
claim ("alpha tensors, not weights") into a measured breakdown.

  python alpha_probe.py --config <cfg.yaml> [--batch_size N]
  torchrun --nproc_per_node=P alpha_probe.py --config <cfg.yaml>
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../complete_verifier"))

MB = 1024 ** 2
rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
torch.cuda.set_device(rank)

if world_size > 1:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    if rank != 0:
        sys.stdout = open(os.devnull, "w")

torch.cuda.reset_peak_memory_stats()

import beta_CROWN_solver
from abcrown import ABCROWN


def _bytes(t):
    return t.numel() * t.element_size() if torch.is_tensor(t) else 0


def categorise(bounded_module):
    """Attribute live tensors of the bound graph to memory categories."""
    from auto_LiRPA.operators.leaf import BoundParams
    cat = {"weights": 0, "alpha": 0, "beta": 0, "interm_bounds": 0, "lA_uA": 0}
    n_alpha = 0
    for node in bounded_module.nodes():
        if isinstance(node, BoundParams):
            p = getattr(node, "param", None)
            cat["weights"] += _bytes(getattr(p, "data", p))
            continue
        alpha = getattr(node, "alpha", None)
        if isinstance(alpha, dict):
            for v in alpha.values():
                b = _bytes(getattr(v, "data", v))
                cat["alpha"] += b
                n_alpha += 1 if b else 0
        sb = getattr(node, "sparse_betas", None)
        if sb:
            objs = list(sb.values()) if isinstance(sb, dict) else list(sb)
            for o in objs:
                for attr in ("val", "loc", "sign", "bias"):
                    cat["beta"] += _bytes(getattr(getattr(o, attr, None), "data",
                                                  getattr(o, attr, None)))
        for attr in ("lower", "upper"):
            cat["interm_bounds"] += _bytes(getattr(node, attr, None))
        for attr in ("lA", "uA"):
            cat["lA_uA"] += _bytes(getattr(node, attr, None))
    return cat, n_alpha


STATE = {"best": None, "calls": 0}
_orig_update_bounds = beta_CROWN_solver.LiRPANet.update_bounds


def traced_update_bounds(self, *args, **kwargs):
    ret = _orig_update_bounds(self, *args, **kwargs)
    cat, n_alpha = categorise(self.net)
    cat = {k: v / MB for k, v in cat.items()}
    cat["n_alpha_tensors"] = n_alpha
    cat["allocated_MB"] = torch.cuda.memory_allocated() / MB
    cat["peak_so_far_MB"] = torch.cuda.max_memory_allocated() / MB
    STATE["calls"] += 1
    if STATE["best"] is None or cat["alpha"] > STATE["best"]["alpha"]:
        STATE["best"] = cat
    return ret


beta_CROWN_solver.LiRPANet.update_bounds = traced_update_bounds

abcrown = ABCROWN(args=sys.argv[1:])
abcrown.main()

peak_mb = torch.cuda.max_memory_allocated() / MB
b = STATE["best"] or {}

if world_size > 1 and rank != 0:
    sys.stdout = sys.__stdout__
if world_size > 1:
    dist.barrier()

print(f"\n=== MEMORY BREAKDOWN rank={rank} P={world_size} "
      f"(update_bounds calls: {STATE['calls']}) ===")
if b:
    order = ["alpha", "beta", "interm_bounds", "lA_uA", "weights"]
    total = sum(b[k] for k in order)
    for k in order:
        share = 100 * b[k] / total if total else 0
        print(f"  {k:<14} {b[k]:10.2f} MB  ({share:5.1f}% of tracked)")
    print(f"  {'tracked total':<14} {total:10.2f} MB")
    print(f"  alpha tensors: {b['n_alpha_tensors']}")
    print(f"  allocated at that point: {b['allocated_MB']:.2f} MB")
print(f"  PROCESS PEAK: {peak_mb:.2f} MB")

if world_size > 1:
    dist.destroy_process_group()
