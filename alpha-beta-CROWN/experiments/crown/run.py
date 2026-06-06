"""
CROWN verification experiment: single GPU vs Tensor Parallel.

Demonstrates that TP reduces per-GPU memory proportionally to the number of GPUs.

Examples:
  # Single GPU
  python run.py --mode single --input-dim 4096 --hidden-dim 262144 --batch-size 2048

  # TP = 2
  torchrun --nproc_per_node=2 run.py --mode tp --input-dim 4096 --hidden-dim 262144 --batch-size 2048
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
from tp_model import SimpleDenseModel, SimpleTPModel, register_tp_custom_ops


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


def print_memory(prefix: str, device: torch.device, rank: int) -> None:
    if device.type != "cuda":
        return
    alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserv_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    print(f"{prefix} [rank={rank}] max_allocated={alloc_mb:.1f}MB, max_reserved={reserv_mb:.1f}MB")


def is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="CROWN: single GPU vs TP memory comparison")
    parser.add_argument("--mode", choices=["single", "tp"], required=True)
    parser.add_argument("--method", choices=["IBP", "CROWN"], default="CROWN")
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=262144)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rank, world_size = 0, 1
    if args.mode == "tp":
        rank, world_size = init_distributed()
        if args.hidden_dim % world_size != 0:
            raise ValueError(f"hidden_dim={args.hidden_dim} must be divisible by world_size={world_size}")

    device = make_device(rank)
    set_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        print(f"CROWN experiment: mode={args.mode}, method={args.method}, "
              f"input_dim={args.input_dim}, hidden_dim={args.hidden_dim}, "
              f"batch_size={args.batch_size}, eps={args.eps}")

    if args.mode == "tp":
        register_tp_custom_ops()
        model = SimpleTPModel(input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                              output_dim=args.output_dim).to(device)
    else:
        model = SimpleDenseModel(input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                                 output_dim=args.output_dim).to(device)

    x = torch.randn(args.batch_size, args.input_dim, device=device)
    lower, upper = x - args.eps, x + args.eps

    try:
        with torch.no_grad():
            lirpa_model = BoundedModule(model, torch.empty_like(x), device=device)
            ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
            bounded_x = BoundedTensor(x, ptb)
            lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=args.method)

        if rank == 0:
            print(f"SUCCESS. bounds shape: lb={tuple(lb.shape)}, ub={tuple(ub.shape)}")
        print_memory("Memory", device=device, rank=rank)
        code = 0

    except RuntimeError as exc:
        if is_oom_error(exc):
            print(f"OOM on rank={rank}: {exc}")
            print_memory("Memory before OOM", device=device, rank=rank)
            code = 2
        else:
            raise
    finally:
        if args.mode == "tp" and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    sys.exit(code)


if __name__ == "__main__":
    main()
