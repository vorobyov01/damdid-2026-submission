"""
VNN-COMP Tensor Parallelism experiment.

Loads an ONNX model (e.g., MNIST-FC from VNN-COMP 2022), creates a
BoundedModule, auto-shards it, and runs incomplete verification (CROWN
and alpha-CROWN) comparing single GPU vs TP=N.

Usage:
  # 1) Download MNIST FC benchmark (run once):
  bash download_mnist_fc.sh

  # 2) Single GPU reference:
  python run.py --mode single --onnx models/mnist-net_256x2.onnx \\
      --save ref.pt --eps 0.026

  # 3) TP=2:
  torchrun --nproc_per_node=2 run.py --mode tp \\
      --onnx models/mnist-net_256x2.onnx --compare ref.pt --eps 0.026
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../auto_LiRPA"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../complete_verifier"))

import torch
import torch.distributed as dist

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.tp_utils import tp_shard_bounded_module


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(rank: int):
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(dev)
        return dev
    return torch.device("cpu")


def init_distributed():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("TP mode requires torchrun.")
    rank = int(os.environ["LOCAL_RANK"])
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=ws)
    return rank, ws


def load_onnx_model(onnx_path: str):
    """Load ONNX model and return (nn.Module, input_shape)."""
    import onnx
    from onnx2pytorch import ConvertModel

    onnx_model = onnx.load(onnx_path)
    pytorch_model = ConvertModel(onnx_model, experimental=True)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.float32)

    inp = onnx_model.graph.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if shape[0] == 0:
        shape[0] = 1
    return pytorch_model, tuple(shape)


def print_memory(device, rank):
    if device.type != "cuda":
        return
    a = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    r = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    print(f"  [rank={rank}] max_allocated={a:.1f}MB, max_reserved={r:.1f}MB")


def compare_bounds(lb, ub, ref_lb, ref_ub, label):
    lb_diff = (lb - ref_lb).abs()
    ub_diff = (ub - ref_ub).abs()
    m = {
        "lb_max": lb_diff.max().item(),
        "ub_max": ub_diff.max().item(),
        "lb_mean": lb_diff.mean().item(),
        "ub_mean": ub_diff.mean().item(),
    }
    ref_range = (ref_ub - ref_lb).abs().clamp(min=1e-8)
    m["lb_rel"] = (lb_diff / ref_range).max().item()
    m["ub_rel"] = (ub_diff / ref_range).max().item()

    print(f"\n  Comparison [{label}]:")
    print(f"    lb max_abs_diff = {m['lb_max']:.6e}, mean = {m['lb_mean']:.6e}")
    print(f"    ub max_abs_diff = {m['ub_max']:.6e}, mean = {m['ub_mean']:.6e}")
    print(f"    lb max_rel_diff = {m['lb_rel']:.6e}")
    print(f"    ub max_rel_diff = {m['ub_rel']:.6e}")

    tol = 1e-3
    ok = m["lb_max"] < tol and m["ub_max"] < tol
    print(f"    {'PASS' if ok else 'FAIL'} (tolerance {tol})")
    return m, ok


def main():
    p = argparse.ArgumentParser(description="VNN-COMP TP experiment")
    p.add_argument("--mode", choices=["single", "tp"], required=True)
    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument("--method", default="alpha-CROWN")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eps", type=float, default=0.026,
                   help="L-inf perturbation radius")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--compare", type=str, default=None)
    args = p.parse_args()

    rank, ws = 0, 1
    if args.mode == "tp":
        rank, ws = init_distributed()

    device = make_device(rank)
    set_seed(args.seed)

    # Load ONNX model
    model, input_shape = load_onnx_model(args.onnx)
    model.to(device)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {args.onnx}")
        print(f"  Input shape: {input_shape}, params: {n_params:,}")
        print(f"  mode={args.mode}, method={args.method}, eps={args.eps}, "
              f"batch={args.batch_size}, ws={ws}")

    # Build input
    ref = None
    if args.compare and os.path.exists(args.compare):
        ref = torch.load(args.compare, map_location="cpu", weights_only=False)
        x = ref["x"].to(device)
        x_L = ref["x_L"].to(device)
        x_U = ref["x_U"].to(device)
        if rank == 0:
            print(f"  Loaded reference from {args.compare}")
    else:
        x = torch.randn(args.batch_size, *input_shape[1:], device=device)
        x = x.clamp(0, 1)
        x_L = (x - args.eps).clamp(0, 1)
        x_U = (x + args.eps).clamp(0, 1)

    # Build BoundedModule
    dummy = torch.empty_like(x)
    lirpa = BoundedModule(model, dummy, device=device)

    if args.mode == "tp" and ws > 1:
        sharded = tp_shard_bounded_module(lirpa, ws, rank, dummy_input=dummy)
        if rank == 0:
            print(f"  Auto-sharded {len(sharded)} nodes: {sharded}")

    # Run methods
    methods = ["CROWN", args.method] if args.method != "CROWN" else ["CROWN"]
    results = {}

    for m in methods:
        if rank == 0:
            print(f"\n  Running {m} ...")

        set_seed(args.seed + 1)

        if m != methods[0]:
            lirpa = BoundedModule(model, dummy, device=device)
            if args.mode == "tp" and ws > 1:
                tp_shard_bounded_module(lirpa, ws, rank, dummy_input=dummy)

        ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_L, x_U=x_U)
        bx = BoundedTensor(x, ptb)

        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        lb, ub = lirpa.compute_bounds(x=(bx,), method=m)
        lb, ub = lb.detach().cpu(), ub.detach().cpu()

        if rank == 0:
            print(f"    lb: min={lb.min():.6f}, max={lb.max():.6f}, "
                  f"mean={lb.mean():.6f}")
            print(f"    ub: min={ub.min():.6f}, max={ub.max():.6f}, "
                  f"mean={ub.mean():.6f}")
            verified = (lb > 0).all().item()
            print(f"    Verified: {verified} (all lb > 0)")
        print_memory(device, rank)
        results[m] = {"lb": lb, "ub": ub}

    # Tightness
    if rank == 0 and len(methods) == 2:
        cr_lb = results["CROWN"]["lb"]
        al_lb = results[args.method]["lb"]
        cr_ub = results["CROWN"]["ub"]
        al_ub = results[args.method]["ub"]
        print(f"\n  Tightness ({args.method} vs CROWN):")
        print(f"    lb improved by {(al_lb - cr_lb).mean():.6f}")
        print(f"    ub improved by {(cr_ub - al_ub).mean():.6f}")

    # Save
    if args.save and rank == 0:
        data = {
            "x": x.cpu(), "x_L": x_L.cpu(), "x_U": x_U.cpu(),
            "args": vars(args),
        }
        for m, r in results.items():
            data[f"lb_{m}"] = r["lb"]
            data[f"ub_{m}"] = r["ub"]
        torch.save(data, args.save)
        print(f"\n  Saved reference to {args.save}")

    # Compare
    if ref is not None and rank == 0:
        all_ok = True
        for m in methods:
            lk, uk = f"lb_{m}", f"ub_{m}"
            if lk in ref and uk in ref:
                _, ok = compare_bounds(
                    results[m]["lb"], results[m]["ub"],
                    ref[lk], ref[uk],
                    label=f"{args.mode} vs ref [{m}]")
                if not ok:
                    all_ok = False
            else:
                print(f"\n  No ref for {m}")
        print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}")

    if args.mode == "tp" and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
