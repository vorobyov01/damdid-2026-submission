#!/usr/bin/env python3
"""Which references keep the AllGathered weights alive after a CROWN pass?

  torchrun --nproc_per_node=2 leak_probe.py --h 4096 --d 4
"""
import argparse, copy, gc, os, sys
from collections import Counter

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../auto_LiRPA"))
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators.leaf import BoundParams
import mem_time as mt

MB = 1024 ** 2


def inventory(tag):
    gc.collect()
    seen, groups = set(), Counter()
    for obj in gc.get_objects():
        try:
            t = obj if torch.is_tensor(obj) else None
            if t is None or not t.is_cuda:
                continue
            if t.data_ptr() in seen or t.data_ptr() == 0:
                continue
            seen.add(t.data_ptr())
            mb = t.numel() * t.element_size() / MB
            if mb >= 1.0:
                groups[tuple(t.shape)] += 1
        except Exception:
            continue
    alloc = torch.cuda.memory_allocated() / MB
    print(f"\n[{tag}] allocated={alloc:.1f} MB")
    for shape, n in sorted(groups.items(), key=lambda kv: -kv[1] * torch.Size(kv[0]).numel())[:6]:
        each = torch.Size(shape).numel() * 4 / MB
        print(f"    x{n:<3} {list(shape)}  {each:.1f} MB each -> {n * each:.1f} MB")
    return alloc


def node_state(lirpa, tag):
    print(f"\n[{tag}] BoundParams attribute inventory")
    for node in lirpa.nodes():
        if not isinstance(node, BoundParams):
            continue
        if getattr(node, "_fsdp_world_size", 0) <= 1:
            continue
        held = []
        for attr in ("param", "forward_value", "lower", "upper", "interval"):
            v = getattr(node, attr, None)
            if isinstance(v, tuple):
                v = v[0] if v else None
            t = getattr(v, "data", v)
            if torch.is_tensor(t):
                held.append(f"{attr}{list(t.shape)}={t.numel() * t.element_size() / MB:.0f}MB")
        # is the attribute really gone from the instance dict?
        in_dict = [a for a in ("forward_value", "lower", "upper", "interval")
                   if a in node.__dict__]
        print(f"    {node.name:<10} {' '.join(held) if held else '(clean)'} "
              f"| in __dict__: {in_dict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=4096)
    ap.add_argument("--d", type=int, default=4)
    args = ap.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(42)
    quiet = rank != 0
    if quiet:
        sys.stdout = open(os.devnull, "w")

    model = mt.make_mlp(784, args.h, args.d).eval().to(dev)
    x = torch.randn(1, 1, 28, 28, device=dev).clamp(0, 1)
    dummy = torch.empty_like(x)
    lirpa = BoundedModule(copy.deepcopy(model), dummy, device=dev)
    del model
    from auto_LiRPA.fsdp_utils import fsdp_shard_bounded_module, fsdp_free_gathered_weights
    fsdp_shard_bounded_module(lirpa, ws, rank, dummy_input=dummy)
    gc.collect(); torch.cuda.empty_cache()
    inventory("after shard")
    node_state(lirpa, "after shard")

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=(x - 0.02).clamp(0, 1),
                             x_U=(x + 0.02).clamp(0, 1))
    lb, ub = lirpa.compute_bounds(x=(BoundedTensor(x, ptb),), method="CROWN")
    print(f"\npeak during compute_bounds = "
          f"{torch.cuda.max_memory_allocated(dev) / MB:.1f} MB")
    del lb, ub
    gc.collect(); torch.cuda.empty_cache()
    inventory("after compute_bounds, bounds deleted")
    node_state(lirpa, "after compute_bounds")

    fsdp_free_gathered_weights(lirpa)
    gc.collect(); torch.cuda.empty_cache()
    inventory("after explicit fsdp_free_gathered_weights")

    if quiet:
        sys.stdout = sys.__stdout__
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
