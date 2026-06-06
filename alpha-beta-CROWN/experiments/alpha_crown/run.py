"""
α-CROWN verification experiment: numerical comparison of single GPU vs TP.

Workflow for verifying correctness:
  1) Run single GPU — save reference bounds + weights:
     python run.py --mode single --save ref.pt

  2) Run TP=1 (same weights, world_size=1) — compare against reference:
     torchrun --nproc_per_node=1 run.py --mode tp --compare ref.pt

  3) Run TP=2 — compare bounds quality:
     torchrun --nproc_per_node=2 run.py --mode tp --compare ref.pt

  Expected: step 2 gives near-zero diff (TP operators are correct),
            step 3 gives small diff (different sharding changes optimization trajectory).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../auto_LiRPA"))

import torch
import torch.distributed as dist

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from tp_model import (SimpleDenseModel, SimpleTPModel,
                       register_tp_custom_ops, copy_dense_weights_to_tp)


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


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("TP mode requires torchrun (LOCAL_RANK missing).")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size


def print_memory(device: torch.device, rank: int) -> None:
    if device.type != "cuda":
        return
    alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserv_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    print(f"  Memory [rank={rank}] max_allocated={alloc_mb:.1f}MB, max_reserved={reserv_mb:.1f}MB")


def run_bounds(model, x, lower, upper, device, method):
    """Compute bounds for a given model and method. Returns (lb, ub) on CPU."""
    lirpa_model = BoundedModule(model, torch.empty_like(x), device=device)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
    bounded_x = BoundedTensor(x, ptb)
    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=method)
    return lb.detach().cpu(), ub.detach().cpu()


def compare_bounds(lb, ub, ref_lb, ref_ub, label: str) -> dict:
    """Numerically compare bounds against reference. Returns metrics dict."""
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
    if metrics["lb_max_abs_diff"] < tol and metrics["ub_max_abs_diff"] < tol:
        print(f"    ✓ PASS (tolerance {tol})")
    else:
        print(f"    ✗ DIFF exceeds tolerance {tol}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="α-CROWN: numerical comparison of single GPU vs TP")
    parser.add_argument("--mode", choices=["single", "tp"], required=True)
    parser.add_argument("--method", default="alpha-CROWN",
                        help="Bound method: CROWN, alpha-CROWN (default: alpha-CROWN)")
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None,
                        help="Save reference results to .pt file (single mode)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare against reference .pt file")
    args = parser.parse_args()

    rank, world_size = 0, 1
    if args.mode == "tp":
        rank, world_size = init_distributed()
        if args.hidden_dim % world_size != 0:
            raise ValueError(
                f"hidden_dim={args.hidden_dim} must be divisible by world_size={world_size}")

    device = make_device(rank)
    set_seed(args.seed)

    if rank == 0:
        print(f"α-CROWN experiment: mode={args.mode}, method={args.method}, "
              f"dims=({args.input_dim},{args.hidden_dim},{args.output_dim}), "
              f"batch={args.batch_size}, eps={args.eps}, world_size={world_size}")

    # --- Build inputs ---
    # If comparing, load inputs from reference to ensure identical data.
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

    # --- Build model ---
    if args.mode == "tp":
        register_tp_custom_ops()
        tp_model = SimpleTPModel(
            input_dim=args.input_dim, hidden_dim=args.hidden_dim,
            output_dim=args.output_dim).to(device)

        if args.compare and os.path.exists(args.compare):
            dense_sd = ref["model_state_dict"]
            dense_tmp = SimpleDenseModel(
                args.input_dim, args.hidden_dim, args.output_dim)
            dense_tmp.load_state_dict(dense_sd)
            dense_tmp.to(device)
            copy_dense_weights_to_tp(dense_tmp, tp_model)
            del dense_tmp
            if rank == 0:
                print("  Loaded + sharded reference weights into TP model")

        model = tp_model
    else:
        model = SimpleDenseModel(
            input_dim=args.input_dim, hidden_dim=args.hidden_dim,
            output_dim=args.output_dim).to(device)

        if args.compare and os.path.exists(args.compare):
            model.load_state_dict(ref["model_state_dict"])
            model.to(device)
            if rank == 0:
                print(f"  Loaded reference weights from {args.compare}")

    # --- Run methods ---
    methods = ["CROWN", args.method] if args.method != "CROWN" else ["CROWN"]

    results = {}
    for m in methods:
        if rank == 0:
            print(f"\n  Running method={m} ...")

        set_seed(args.seed + 1)
        lb, ub = run_bounds(model, x, lower, upper, device, m)

        if rank == 0:
            print(f"    lb: min={lb.min().item():.6f}, max={lb.max().item():.6f}, "
                  f"mean={lb.mean().item():.6f}")
            print(f"    ub: min={ub.min().item():.6f}, max={ub.max().item():.6f}, "
                  f"mean={ub.mean().item():.6f}")
        print_memory(device, rank)

        results[m] = {"lb": lb, "ub": ub}

    # --- Compare CROWN vs alpha-CROWN (tightness) ---
    if rank == 0 and len(methods) == 2:
        crown_lb = results["CROWN"]["lb"]
        alpha_lb = results[args.method]["lb"]
        crown_ub = results["CROWN"]["ub"]
        alpha_ub = results[args.method]["ub"]

        lb_improvement = (alpha_lb - crown_lb).mean().item()
        ub_improvement = (crown_ub - alpha_ub).mean().item()
        print(f"\n  Tightness improvement ({args.method} vs CROWN):")
        print(f"    lb improved by {lb_improvement:.6f} (positive = tighter)")
        print(f"    ub improved by {ub_improvement:.6f} (positive = tighter)")

    # --- Save reference ---
    if args.save and rank == 0:
        save_data = {
            "x": x.cpu(), "lower": lower.cpu(), "upper": upper.cpu(),
            "args": vars(args),
        }
        if args.mode == "single":
            save_data["model_state_dict"] = model.state_dict()
        for m, r in results.items():
            save_data[f"lb_{m}"] = r["lb"]
            save_data[f"ub_{m}"] = r["ub"]
        torch.save(save_data, args.save)
        print(f"\n  Saved reference to {args.save}")

    # --- Compare against reference ---
    if args.compare and os.path.exists(args.compare) and rank == 0:
        for m in methods:
            ref_key_lb = f"lb_{m}"
            ref_key_ub = f"ub_{m}"
            if ref_key_lb in ref and ref_key_ub in ref:
                compare_bounds(
                    results[m]["lb"], results[m]["ub"],
                    ref[ref_key_lb], ref[ref_key_ub],
                    label=f"{args.mode} vs reference [{m}]")
            else:
                print(f"\n  No reference bounds for method={m} in {args.compare}")

    if args.mode == "tp" and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
