#!/usr/bin/env python3
"""Peak-memory decomposition and timing for CROWN bound propagation.

Runs ONE mode per process, so single-GPU and FSDP measurements can never
contaminate each other (the original memory_experiment.py measured both in the
same process, FSDP first, which confounds the comparison).

Reported per configuration:
  params      -- bytes held in BoundParams (what FSDP actually shards)
  resident    -- torch.cuda.memory_allocated() right before compute_bounds
  peak        -- max_memory_allocated during compute_bounds
  transient   -- peak - resident (A-matrices, intermediate bounds, gathered weights)
  wall_ms     -- wall-clock of compute_bounds (CUDA-synchronised)
  comm_ms     -- GPU time inside AllGather/AllReduce (CUDA events)

Usage:
  python mem_time.py --mode single --h 4096 --d 4 --repeats 3 --out out.json
  torchrun --nproc_per_node=2 mem_time.py --mode fsdp --h 4096 --d 4 --repeats 3
"""
import argparse, copy, gc, json, os, statistics, sys, time

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../auto_LiRPA"))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators.leaf import BoundParams

MB = 1024 ** 2

# ---------------------------------------------------------------- collectives
COMM = {"calls": 0, "bytes": 0, "events": []}
_orig_all_gather = dist.all_gather
_orig_all_reduce = dist.all_reduce
_orig_all_gather_into = dist.all_gather_into_tensor


def _timed(fn, nbytes_fn):
    def wrapper(*args, **kwargs):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = fn(*args, **kwargs)
        e.record()
        COMM["calls"] += 1
        COMM["bytes"] += nbytes_fn(*args)
        COMM["events"].append((s, e))
        return out
    return wrapper


def install_comm_timers(world_size):
    dist.all_gather = _timed(
        _orig_all_gather,
        lambda tensor_list, tensor, *a: tensor.numel() * tensor.element_size() * world_size)
    dist.all_reduce = _timed(
        _orig_all_reduce,
        lambda tensor, *a: tensor.numel() * tensor.element_size())
    dist.all_gather_into_tensor = _timed(
        _orig_all_gather_into,
        lambda output, input, *a: output.numel() * output.element_size())


def comm_reset():
    COMM["calls"], COMM["bytes"], COMM["events"] = 0, 0, []


def comm_ms():
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in COMM["events"])


# ------------------------------------------------------------- CPU offloading
OFFLOAD = {"calls": 0, "bytes": 0, "events": []}


def install_cpu_offload(lirpa):
    """Keep every weight in pinned host memory, stage it to GPU per layer.

    This is the naive alternative to FSDP that reviewers asked us to compare
    against: it removes the *whole* weight term from GPU residency rather than
    the (1 - 1/P) fraction FSDP removes, and pays a host-to-device copy per
    layer instead of a collective.  We reuse auto_LiRPA's existing FSDP hook
    sites by marking the nodes and replacing the gather with an H2D copy.
    """
    from auto_LiRPA.operators.leaf import BoundParams
    from auto_LiRPA import fsdp_utils

    n = 0
    for node in lirpa.nodes():
        if not isinstance(node, BoundParams):
            continue
        w = node.param.data
        if w.ndim < 2:
            continue
        node.param = nn.Parameter(w.detach().to("cpu").pin_memory(),
                                  requires_grad=False)
        node._fsdp_world_size = 2      # marks the node for the hook sites
        node._fsdp_shard_dim = 0
        node._cpu_offload = True
        node.forward_value = None
        n += 1

    fsdp_utils._OFFLOAD_STATS['calls'] = 0
    fsdp_utils._OFFLOAD_STATS['bytes'] = 0
    return n


def offload_stats():
    from auto_LiRPA import fsdp_utils
    return fsdp_utils._OFFLOAD_STATS


# --------------------------------------------------------------------- tracing
TRACE = []


def install_trace(lirpa):
    """Record allocated bytes around every node's bound_backward call."""
    for node in lirpa.nodes():
        if not hasattr(node, "bound_backward"):
            continue
        orig = node.bound_backward
        nname, ntype = node.name, type(node).__name__

        def wrapper(lA, uA, *inputs, _orig=orig, _n=nname, _t=ntype, **kwargs):
            def nbytes(t):
                return t.numel() * t.element_size() if torch.is_tensor(t) else 0
            before = torch.cuda.memory_allocated()
            out = _orig(lA, uA, *inputs, **kwargs)
            after = torch.cuda.memory_allocated()
            TRACE.append({
                "node": _n, "type": _t,
                "in_A_MB": (nbytes(lA) + nbytes(uA)) / MB,
                "in_A_shape": list(lA.shape) if torch.is_tensor(lA) else None,
                "alloc_before_MB": before / MB,
                "alloc_after_MB": after / MB,
                "peak_so_far_MB": torch.cuda.max_memory_allocated() / MB,
            })
            return out
        node.bound_backward = wrapper


# ----------------------------------------------------------------------- model
def make_mlp(input_dim, hidden_dim, num_layers, output_dim=10):
    layers = [nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def param_bytes(lirpa):
    return sum(n.param.data.numel() * n.param.data.element_size()
               for n in lirpa.nodes() if isinstance(n, BoundParams))


RETAIN = []


def measure(model, x, eps, dev, mode, ws, rank, method, trace=False, retain=False):
    gc.collect(); torch.cuda.empty_cache()
    x_L, x_U = (x - eps).clamp(0, 1), (x + eps).clamp(0, 1)
    dummy = torch.empty_like(x)

    lirpa = BoundedModule(copy.deepcopy(model), dummy, device=dev)
    if mode == "offload":
        OFFLOAD["calls"], OFFLOAD["bytes"], OFFLOAD["events"] = 0, 0, []
        install_cpu_offload(lirpa)
    if mode == "fsdp" and ws > 1:
        from auto_LiRPA.fsdp_utils import fsdp_shard_bounded_module
        fsdp_shard_bounded_module(lirpa, ws, rank, dummy_input=dummy)

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_L.clone(), x_U=x_U.clone())
    bx = BoundedTensor(x.clone(), ptb)

    p_mb = param_bytes(lirpa) / MB
    if trace:
        TRACE.clear()
        install_trace(lirpa)

    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    comm_reset()
    resident = torch.cuda.memory_allocated(dev) / MB

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lb, ub = lirpa.compute_bounds(x=(bx,), method=method)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3

    peak = torch.cuda.max_memory_allocated(dev) / MB
    res = {
        "params_MB": p_mb, "resident_MB": resident, "peak_MB": peak,
        "transient_MB": peak - resident, "wall_ms": wall,
        "comm_ms": (0.0 if mode == "offload"
                    else (comm_ms() if ws > 1 else 0.0)),
        "comm_calls": (offload_stats()["calls"] if mode == "offload"
                       else COMM["calls"]),
        "comm_MB": ((offload_stats()["bytes"] if mode == "offload"
                     else COMM["bytes"]) / MB),
        "lb_sum": float(lb.detach().sum()), "ub_sum": float(ub.detach().sum()),
        "lb_min": float(lb.detach().min()), "ub_max": float(ub.detach().max()),
    }
    if retain:
        # Reproduce the original protocol: the caller keeps the returned bound
        # tensors, which for FSDP still carry an autograd graph.
        RETAIN.append((lirpa, lb, ub))
    else:
        del lirpa, lb, ub
    gc.collect(); torch.cuda.empty_cache()
    return res


def agg(runs, key):
    vals = [r[key] for r in runs]
    return (statistics.mean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "fsdp", "offload"], required=True)
    ap.add_argument("--h", type=int, default=4096)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--input-dim", type=int, default=784)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--method", default="CROWN")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        dist.init_process_group("nccl", rank=rank, world_size=ws)
        install_comm_timers(ws)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(args.seed)

    model = make_mlp(args.input_dim, args.h, args.d).to(dev).eval()
    side = int(round(args.input_dim ** 0.5))
    x = (torch.randn(1, 1, side, side, device=dev).clamp(0, 1)
         if side * side == args.input_dim
         else torch.randn(1, args.input_dim, device=dev).clamp(0, 1))

    # Warm-up: first compute_bounds pays JIT / cuBLAS autotune / NCCL init cost
    for _ in range(args.warmup):
        measure(model, x, args.eps, dev, args.mode, ws, rank, args.method)

    runs = []
    for i in range(args.repeats):
        runs.append(measure(model, x, args.eps, dev, args.mode, ws, rank,
                            args.method, trace=(args.trace and i == 0)))

    # max over ranks for memory, mean for time
    if ws > 1:
        gathered = [None] * ws
        dist.all_gather_object(gathered, runs)
    else:
        gathered = [runs]

    if rank == 0:
        tag = f"{args.mode}{'' if ws == 1 else f'=P{ws}'} h={args.h} d={args.d} {args.method}"
        out = {"mode": args.mode, "P": ws, "h": args.h, "d": args.d,
               "method": args.method, "repeats": args.repeats,
               "per_rank": gathered}
        print(f"\n===== {tag} =====")
        for key in ("params_MB", "resident_MB", "peak_MB", "transient_MB",
                    "wall_ms", "comm_ms", "comm_calls", "comm_MB"):
            # worst rank per repeat, then mean +- std over repeats
            worst = [{key: max(g[i][key] for g in gathered)} for i in range(args.repeats)]
            m, s = agg(worst, key)
            out[key] = {"mean": m, "std": s}
            print(f"  {key:<14} {m:12.2f} +- {s:6.2f}")
        b = gathered[0][0]
        print(f"  bounds: lb_sum={b['lb_sum']:.6f} ub_sum={b['ub_sum']:.6f} "
              f"lb_min={b['lb_min']:.6f} ub_max={b['ub_max']:.6f}")
        if args.trace and TRACE:
            out["trace"] = TRACE
            print(f"\n  --- backward trace ({len(TRACE)} steps), top-8 by peak ---")
            for t in sorted(TRACE, key=lambda z: -z["peak_so_far_MB"])[:8]:
                print(f"    {t['type']:<22} inA={t['in_A_MB']:8.1f} MB "
                      f"shape={t['in_A_shape']} "
                      f"alloc {t['alloc_before_MB']:8.1f}->{t['alloc_after_MB']:8.1f} "
                      f"peak={t['peak_so_far_MB']:8.1f}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  saved -> {args.out}")

    if ws > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
