"""
OOM experiment for verification:
1) trigger OOM on a single GPU
2) show the same setup works with Tensor Parallel (TP=2)

Examples:
  # 1 GPU (expect OOM on sufficiently large config)
  python3 examples/simple/oom_tp_experiment.py --mode single --method CROWN --input-dim 4096 --hidden-dim 262144 --batch-size 2048

  # 2 GPU TP (same config, should use about half memory per rank)
  torchrun --nproc_per_node=2 examples/simple/oom_tp_experiment.py --mode tp --method CROWN --input-dim 4096 --hidden-dim 262144 --batch-size 2048
"""
import argparse
import os
import random
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

# Add parent directory to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from tp_model import SimpleTPModel, register_tp_custom_ops


class SimpleDenseModel(nn.Module):
    """Single-GPU baseline model (no tensor parallelism)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


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


def init_distributed_for_tp() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("TP mode must be launched with torchrun (LOCAL_RANK is missing).")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size


def build_inputs(batch_size: int, input_dim: int, eps: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, input_dim, device=device)
    lower = x - eps
    upper = x + eps
    return x, lower, upper


def print_memory(prefix: str, device: torch.device, rank: int) -> None:
    if device.type != "cuda":
        if rank == 0:
            print(f"{prefix}: CPU mode, GPU memory stats are unavailable.")
        return
    alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserv_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    print(f"{prefix} [rank={rank}] max_allocated={alloc_mb:.1f}MB, max_reserved={reserv_mb:.1f}MB")


def run_verification(
    mode: str,
    method: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    batch_size: int,
    eps: float,
    seed: int,
) -> int:
    rank = 0
    world_size = 1

    if mode == "tp":
        rank, world_size = init_distributed_for_tp()
        if hidden_dim % world_size != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by world_size={world_size} in TP mode."
            )
    device = make_device(rank)
    set_seed(seed)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        print(
            f"Start experiment: mode={mode}, method={method}, input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, output_dim={output_dim}, batch_size={batch_size}, eps={eps}"
        )

    if mode == "tp":
        register_tp_custom_ops()
        model = SimpleTPModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    else:
        model = SimpleDenseModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

    x, lower, upper = build_inputs(batch_size=batch_size, input_dim=input_dim, eps=eps, device=device)

    try:
        with torch.no_grad():
            lirpa_model = BoundedModule(model, torch.empty_like(x), device=device)
            ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
            bounded_x = BoundedTensor(x, ptb)
            lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=method)

        if rank == 0:
            print(f"Verification SUCCESS. Bounds shape: lb={tuple(lb.shape)}, ub={tuple(ub.shape)}")
        print_memory("Memory usage", device=device, rank=rank)
        return 0

    except RuntimeError as exc:
        if is_oom_error(exc):
            print(f"OOM detected on rank={rank}: {exc}")
            print_memory("Memory usage before OOM", device=device, rank=rank)
            return 2
        raise
    finally:
        if mode == "tp" and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOM vs TP experiment for auto_LiRPA verification.")
    parser.add_argument("--mode", choices=["single", "tp"], required=True, help="single: one GPU dense model, tp: Tensor Parallel model")
    parser.add_argument("--method", choices=["IBP", "CROWN"], default="CROWN")
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=262144)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = run_verification(
        mode=args.mode,
        method=args.method,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        eps=args.eps,
        seed=args.seed,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()

