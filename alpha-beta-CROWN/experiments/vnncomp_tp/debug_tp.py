"""Debug script to locate the TP=2 hang on ONNX model."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../auto_LiRPA"))

import torch
import torch.distributed as dist

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.tp_utils import tp_shard_bounded_module


def log(rank, msg):
    print(f"[rank={rank} t={time.time():.3f}] {msg}", flush=True)


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)

    log(rank, "=== Distributed init OK ===")

    # Test 1: Pure PyTorch model with same architecture as MNIST-FC 256x2
    log(rank, "Building PyTorch model (same architecture as ONNX)...")
    import torch.nn as nn
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    model.to(dev)
    log(rank, "Model built OK")

    x = torch.randn(1, 784, 1, device=dev).clamp(0, 1)
    dummy = torch.empty_like(x)

    log(rank, "Creating BoundedModule...")
    lirpa = BoundedModule(model, dummy, device=dev)
    log(rank, "BoundedModule created OK")

    log(rank, "Auto-sharding...")
    sharded = tp_shard_bounded_module(lirpa, ws, rank, dummy_input=dummy)
    log(rank, f"Sharded: {sharded}")

    log(rank, "Building perturbation...")
    eps = 0.03
    x_L = (x - eps).clamp(0, 1)
    x_U = (x + eps).clamp(0, 1)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_L, x_U=x_U)
    bx = BoundedTensor(x, ptb)

    log(rank, "Computing CROWN bounds...")
    lb, ub = lirpa.compute_bounds(x=(bx,), method="CROWN")
    log(rank, f"CROWN OK: lb={lb.min():.6f}, ub={ub.max():.6f}")

    # Test 2: ONNX model
    log(rank, "=== Now testing ONNX model ===")
    import onnx
    from onnx2pytorch import ConvertModel

    onnx_model = onnx.load("models/mnist-net_256x2.onnx")
    model_onnx = ConvertModel(onnx_model, experimental=True)
    model_onnx.eval().to(dev)
    log(rank, "ONNX model loaded OK")

    log(rank, "Creating BoundedModule from ONNX...")
    lirpa2 = BoundedModule(model_onnx, dummy, device=dev)
    log(rank, "BoundedModule from ONNX created OK")

    # Print graph
    if rank == 0:
        for name, node in lirpa2._modules.items():
            log(rank, f"  node: {type(node).__name__:30s} {name}")

    log(rank, "Auto-sharding ONNX model...")
    sharded2 = tp_shard_bounded_module(lirpa2, ws, rank, dummy_input=dummy)
    log(rank, f"Sharded ONNX: {sharded2}")

    # Print graph after sharding
    if rank == 0:
        for name, node in lirpa2._modules.items():
            log(rank, f"  post-shard: {type(node).__name__:30s} {name}")

    log(rank, "Building perturbation for ONNX...")
    ptb2 = PerturbationLpNorm(norm=float("inf"), x_L=x_L, x_U=x_U)
    bx2 = BoundedTensor(x, ptb2)

    log(rank, "Computing CROWN bounds on ONNX model...")
    lb2, ub2 = lirpa2.compute_bounds(x=(bx2,), method="CROWN")
    log(rank, f"CROWN on ONNX OK: lb={lb2.min():.6f}, ub={ub2.max():.6f}")

    log(rank, "=== ALL DONE ===")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
