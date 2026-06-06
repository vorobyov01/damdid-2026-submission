"""
Auto-sharding experiment: verify that tp_shard_bounded_module() produces
correct bounds by comparing against single-GPU baseline.

Workflow:
  1) Generate reference (single GPU):
     python run.py --mode single --save ref.pt

  2) Verify correctness with TP=1 (auto-shard, same weights, world_size=1):
     torchrun --nproc_per_node=1 run.py --mode tp --compare ref.pt

  3) Verify with TP=2:
     torchrun --nproc_per_node=2 run.py --mode tp --compare ref.pt

  Expected: TP=1 gives near-zero diff; TP=2 gives near-zero diff for CROWN,
            small diff for alpha-CROWN (different optimization trajectory).
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../auto_LiRPA"))

import torch
import torch.distributed as dist

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.tp_utils import tp_shard_bounded_module


class DenseMLP(torch.nn.Module):
    """Simple dense MLP for testing auto-sharding."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_hidden: int = 1):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(n_hidden):
            layers.append(torch.nn.Linear(prev, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev = hidden_dim
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def init_distributed():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("TP mode requires torchrun (LOCAL_RANK missing).")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size


def print_memory(device, rank):
    if device.type != "cuda":
        return
    alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserv = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    print(f"  [rank={rank}] max_allocated={alloc:.1f}MB, max_reserved={reserv:.1f}MB")


def run_bounds(model, x, lower, upper, device, method):
    lirpa = BoundedModule(model, torch.empty_like(x), device=device)
    return lirpa, method


def compare_bounds(lb, ub, ref_lb, ref_ub, label: str):
    lb_diff = (lb - ref_lb).abs()
    ub_diff = (ub - ref_ub).abs()
    metrics = {
        "lb_max_abs_diff": lb_diff.max().item(),
        "ub_max_abs_diff": ub_diff.max().item(),
        "lb_mean_abs_diff": lb_diff.mean().item(),
        "ub_mean_abs_diff": ub_diff.mean().item(),
    }
    ref_range = (ref_ub - ref_lb).abs().clamp(min=1e-8)
    metrics["lb_max_rel_diff"] = (lb_diff / ref_range).max().item()
    metrics["ub_max_rel_diff"] = (ub_diff / ref_range).max().item()

    print(f"\n  Comparison [{label}]:")
    print(f"    lb max_abs_diff = {metrics['lb_max_abs_diff']:.6e}, "
          f"mean = {metrics['lb_mean_abs_diff']:.6e}")
    print(f"    ub max_abs_diff = {metrics['ub_max_abs_diff']:.6e}, "
          f"mean = {metrics['ub_mean_abs_diff']:.6e}")
    print(f"    lb max_rel_diff = {metrics['lb_max_rel_diff']:.6e}")
    print(f"    ub max_rel_diff = {metrics['ub_max_rel_diff']:.6e}")

    tol = 1e-4
    ok = metrics["lb_max_abs_diff"] < tol and metrics["ub_max_abs_diff"] < tol
    print(f"    {'PASS' if ok else 'FAIL'} (tolerance {tol})")
    return metrics, ok


def main():
    parser = argparse.ArgumentParser(
        description="Auto-sharding TP experiment")
    parser.add_argument("--mode", choices=["single", "tp"], required=True)
    parser.add_argument("--method", default="alpha-CROWN")
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--n-hidden", type=int, default=1,
                        help="Number of hidden layers (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--compare", type=str, default=None)
    args = parser.parse_args()

    rank, world_size = 0, 1
    if args.mode == "tp":
        rank, world_size = init_distributed()
        if args.hidden_dim % world_size != 0:
            raise ValueError(
                f"hidden_dim={args.hidden_dim} must be divisible by "
                f"world_size={world_size}")

    device = make_device(rank)
    set_seed(args.seed)

    if rank == 0:
        print(f"Auto-shard experiment: mode={args.mode}, method={args.method}, "
              f"dims=({args.input_dim},{args.hidden_dim},{args.output_dim}), "
              f"n_hidden={args.n_hidden}, batch={args.batch_size}, "
              f"eps={args.eps}, world_size={world_size}")

    # --- Build inputs ---
    ref = None
    if args.compare and os.path.exists(args.compare):
        ref = torch.load(args.compare, map_location="cpu", weights_only=False)
        x = ref["x"].to(device)
        lower = ref["lower"].to(device)
        upper = ref["upper"].to(device)
        if rank == 0:
            print(f"  Loaded reference inputs from {args.compare}")
    else:
        x = torch.randn(args.batch_size, args.input_dim, device=device)
        lower, upper = x - args.eps, x + args.eps

    # --- Build model (always dense, then auto-shard if TP) ---
    model = DenseMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        n_hidden=args.n_hidden,
    ).to(device)

    if ref is not None and "model_state_dict" in ref:
        model.load_state_dict(ref["model_state_dict"])
        model.to(device)
        if rank == 0:
            print(f"  Loaded reference weights")

    # --- Build BoundedModule and optionally auto-shard ---
    lirpa = BoundedModule(model, torch.empty_like(x), device=device)

    if args.mode == "tp" and world_size > 1:
        sharded = tp_shard_bounded_module(lirpa, world_size, rank,
                                          dummy_input=torch.empty_like(x))
        if rank == 0:
            print(f"  Auto-sharded {len(sharded)} nodes: {sharded}")

    # --- Run methods ---
    methods = ["CROWN", args.method] if args.method != "CROWN" else ["CROWN"]

    results = {}
    for m in methods:
        if rank == 0:
            print(f"\n  Running method={m} ...")

        set_seed(args.seed + 1)
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
        bounded_x = BoundedTensor(x, ptb)

        if m != methods[0]:
            lirpa = BoundedModule(model, torch.empty_like(x), device=device)
            if args.mode == "tp" and world_size > 1:
                tp_shard_bounded_module(lirpa, world_size, rank,
                                        dummy_input=torch.empty_like(x))
            ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
            bounded_x = BoundedTensor(x, ptb)

        lb, ub = lirpa.compute_bounds(x=(bounded_x,), method=m)
        lb, ub = lb.detach().cpu(), ub.detach().cpu()

        if rank == 0:
            print(f"    lb: min={lb.min().item():.6f}, max={lb.max().item():.6f}, "
                  f"mean={lb.mean().item():.6f}")
            print(f"    ub: min={ub.min().item():.6f}, max={ub.max().item():.6f}, "
                  f"mean={ub.mean().item():.6f}")
        print_memory(device, rank)
        results[m] = {"lb": lb, "ub": ub}

    # --- Tightness comparison ---
    if rank == 0 and len(methods) == 2:
        crown_lb = results["CROWN"]["lb"]
        alpha_lb = results[args.method]["lb"]
        crown_ub = results["CROWN"]["ub"]
        alpha_ub = results[args.method]["ub"]
        lb_imp = (alpha_lb - crown_lb).mean().item()
        ub_imp = (crown_ub - alpha_ub).mean().item()
        print(f"\n  Tightness ({args.method} vs CROWN):")
        print(f"    lb improved by {lb_imp:.6f}")
        print(f"    ub improved by {ub_imp:.6f}")

    # --- Save reference ---
    if args.save and rank == 0:
        save_data = {
            "x": x.cpu(), "lower": lower.cpu(), "upper": upper.cpu(),
            "args": vars(args),
            "model_state_dict": model.state_dict(),
        }
        for m, r in results.items():
            save_data[f"lb_{m}"] = r["lb"]
            save_data[f"ub_{m}"] = r["ub"]
        torch.save(save_data, args.save)
        print(f"\n  Saved reference to {args.save}")

    # --- Compare against reference ---
    if ref is not None and rank == 0:
        all_passed = True
        for m in methods:
            ref_lb_key, ref_ub_key = f"lb_{m}", f"ub_{m}"
            if ref_lb_key in ref and ref_ub_key in ref:
                _, ok = compare_bounds(
                    results[m]["lb"], results[m]["ub"],
                    ref[ref_lb_key], ref[ref_ub_key],
                    label=f"{args.mode} vs reference [{m}]")
                if not ok:
                    all_passed = False
            else:
                print(f"\n  No reference for method={m}")
        if all_passed:
            print("\n  ALL COMPARISONS PASSED")
        else:
            print("\n  SOME COMPARISONS FAILED")

    if args.mode == "tp" and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
