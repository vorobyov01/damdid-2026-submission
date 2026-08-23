#!/usr/bin/env python3
"""TP peak memory and wall-clock as a function of the number of GPUs.

Same model as experiments/crown/run.py (one wide hidden layer, Column-parallel
followed by Row-parallel), but with repeats, a warm-up, wall-clock timing and
AllReduce timing, so that the memory reduction can be weighed against its
communication cost.  Runs one mode per process.

  python tp_scaling.py --mode single --hidden-dim 131072
  torchrun --nproc_per_node=P tp_scaling.py --mode tp --hidden-dim 131072
"""
import argparse, os, statistics, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../auto_LiRPA"))

import torch
import torch.distributed as dist

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from tp_model import SimpleDenseModel, SimpleTPModel, register_tp_custom_ops

MB = 1024 ** 2
COMM = {"calls": 0, "bytes": 0, "events": []}
_orig_all_reduce = dist.all_reduce


def timed_all_reduce(tensor, *a, **kw):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    out = _orig_all_reduce(tensor, *a, **kw)
    e.record()
    COMM["calls"] += 1
    COMM["bytes"] += tensor.numel() * tensor.element_size()
    COMM["events"].append((s, e))
    return out


def run_once(args, device, rank, ws):
    torch.manual_seed(args.seed)
    if args.mode == "tp":
        register_tp_custom_ops()
        model = SimpleTPModel(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    else:
        model = SimpleDenseModel(args.input_dim, args.hidden_dim, args.output_dim).to(device)

    x = torch.randn(args.batch_size, args.input_dim, device=device)
    lower, upper = x - args.eps, x + args.eps

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    COMM["calls"], COMM["bytes"], COMM["events"] = 0, 0, []

    # Graph construction (JIT trace of the custom TP operators) is a one-off
    # cost and is timed separately from bound propagation itself.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        lirpa = BoundedModule(model, torch.empty_like(x), device=device)
    torch.cuda.synchronize()
    build = (time.perf_counter() - t0) * 1e3

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
        lb, ub = lirpa.compute_bounds(x=(BoundedTensor(x, ptb),), method=args.method)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t1) * 1e3

    res = {
        "peak_MB": torch.cuda.max_memory_allocated(device) / MB,
        "reserved_MB": torch.cuda.max_memory_reserved(device) / MB,
        "build_ms": build,
        "wall_ms": wall,
        "comm_ms": sum(s.elapsed_time(e) for s, e in COMM["events"]),
        "comm_calls": COMM["calls"],
        "comm_MB": COMM["bytes"] / MB,
        "lb_sum": float(lb.detach().sum()), "ub_sum": float(ub.detach().sum()),
    }
    del lirpa, model, lb, ub, x, lower, upper
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "tp"], required=True)
    ap.add_argument("--method", choices=["IBP", "CROWN"], default="CROWN")
    ap.add_argument("--input-dim", type=int, default=4096)
    ap.add_argument("--hidden-dim", type=int, default=131072)
    ap.add_argument("--output-dim", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    rank, ws = 0, 1
    if args.mode == "tp":
        rank = int(os.environ["LOCAL_RANK"])
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        dist.init_process_group("nccl", rank=rank, world_size=ws)
        dist.all_reduce = timed_all_reduce
        if args.hidden_dim % ws:
            raise ValueError("hidden_dim must be divisible by world size")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    for _ in range(args.warmup):
        run_once(args, device, rank, ws)
    runs = [run_once(args, device, rank, ws) for _ in range(args.repeats)]

    if ws > 1:
        gathered = [None] * ws
        dist.all_gather_object(gathered, runs)
    else:
        gathered = [runs]

    if rank == 0:
        print(f"\n===== {args.mode}{'' if ws == 1 else f'=P{ws}'} "
              f"H={args.hidden_dim} D={args.input_dim} N={args.batch_size} "
              f"{args.method} =====")
        for key in ("peak_MB", "reserved_MB", "build_ms", "wall_ms",
                    "comm_ms", "comm_calls", "comm_MB"):
            worst = [max(g[i][key] for g in gathered) for i in range(args.repeats)]
            m = statistics.mean(worst)
            s = statistics.stdev(worst) if len(worst) > 1 else 0.0
            print(f"  {key:<12} {m:12.2f} +- {s:6.2f}")
        print(f"  bounds: lb_sum={gathered[0][0]['lb_sum']:.6f} "
              f"ub_sum={gathered[0][0]['ub_sum']:.6f}")

    if ws > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
