#!/usr/bin/env python3
"""Where does the weight memory actually live, and does FSDP sharding free it?

The paper's baseline metric counts bytes inside BoundParams tensors.  This
script instead measures the allocated CUDA memory before and after sharding and
enumerates every live CUDA tensor above a size threshold, so that any duplicate
copy of the weights (kept by BoundedModule, by the caller's module, or by the
JIT trace) becomes visible.

  python baseline_probe.py --mode single
  torchrun --nproc_per_node=2 baseline_probe.py --mode fsdp
"""
import argparse, copy, gc, os, sys
from collections import Counter

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../auto_LiRPA"))
from auto_LiRPA import BoundedModule
from auto_LiRPA.operators.leaf import BoundParams
import mem_time as mt

MB = 1024 ** 2


def live_cuda_tensors(min_mb=1.0):
    """Inventory of live CUDA tensors above min_mb, grouped by shape."""
    gc.collect()
    seen, groups = set(), Counter()
    total = 0
    for obj in gc.get_objects():
        try:
            t = obj if torch.is_tensor(obj) else getattr(obj, "data", None)
            if not torch.is_tensor(t) or not t.is_cuda:
                continue
            key = t.data_ptr()
            if key in seen or key == 0:
                continue
            seen.add(key)
            mb = t.numel() * t.element_size() / MB
            if mb >= min_mb:
                groups[tuple(t.shape)] += 1
                total += mb
        except Exception:
            continue
    return groups, total


def report(tag, lirpa=None):
    alloc = torch.cuda.memory_allocated() / MB
    bp = 0.0
    if lirpa is not None:
        bp = sum(n.param.data.numel() * n.param.data.element_size()
                 for n in lirpa.nodes() if isinstance(n, BoundParams)) / MB
    groups, tracked = live_cuda_tensors()
    print(f"\n[{tag}] allocated={alloc:.1f} MB  BoundParams={bp:.1f} MB  "
          f"live-tensors>=1MB: {tracked:.1f} MB")
    for shape, n in sorted(groups.items(), key=lambda kv: -kv[1] * torch.Size(kv[0]).numel())[:8]:
        mb = torch.Size(shape).numel() * 4 / MB
        print(f"    x{n}  {list(shape)}  {mb:.1f} MB each -> {n * mb:.1f} MB")
    return alloc, bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "fsdp"], required=True)
    ap.add_argument("--h", type=int, default=4096)
    ap.add_argument("--d", type=int, default=4)
    args = ap.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        dist.init_process_group("nccl", rank=rank, world_size=ws)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(42)

    quiet = (rank != 0)
    if quiet:
        sys.stdout = open(os.devnull, "w")

    model_cpu = mt.make_mlp(784, args.h, args.d).eval()
    w_mb = sum(p.numel() * p.element_size() for p in model_cpu.parameters()) / MB
    print(f"model weights (fp32) = {w_mb:.1f} MB, mode={args.mode}, P={ws}")

    report("empty GPU")
    model = model_cpu.to(dev)
    report("after model.to(cuda)")

    dummy = torch.empty(1, 1, 28, 28, device=dev)
    lirpa = BoundedModule(copy.deepcopy(model), dummy, device=dev)
    report("after BoundedModule (deepcopy of model)", lirpa)

    # the honest baseline: drop the caller's own copy, keep only the verifier
    del model
    gc.collect(); torch.cuda.empty_cache()
    a_before, bp_before = report("caller's model deleted", lirpa)

    if args.mode == "fsdp" and ws > 1:
        from auto_LiRPA.fsdp_utils import fsdp_shard_bounded_module
        fsdp_shard_bounded_module(lirpa, ws, rank, dummy_input=dummy)
        gc.collect(); torch.cuda.empty_cache()
        a_after, bp_after = report(f"after FSDP shard P={ws}", lirpa)
        print(f"\nSUMMARY mode=fsdp P={ws}: "
              f"BoundParams {bp_before:.1f} -> {bp_after:.1f} MB "
              f"({bp_after / bp_before:.3f} of original, ideal {1/ws:.3f}); "
              f"real allocated {a_before:.1f} -> {a_after:.1f} MB "
              f"({a_after / a_before:.3f} of original)")
    else:
        print(f"\nSUMMARY mode=single: BoundParams={bp_before:.1f} MB, "
              f"allocated={a_before:.1f} MB")

    if quiet:
        sys.stdout = sys.__stdout__
    if ws > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
