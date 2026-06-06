"""BaB memory experiment: wide MLP, single GPU vs FSDP=2.

Generates a wide MLP model + vnnlib spec, runs complete verification (BaB)
in single-GPU mode and FSDP mode, reports peak memory for both.

Usage:
  # Single GPU baseline:
  CUDA_VISIBLE_DEVICES=0 python bab_memory_experiment.py --mode single \
      --hidden-dim 2048 --num-layers 4 --batch-size 512

  # FSDP with 2 GPUs:
  torchrun --nproc_per_node=2 bab_memory_experiment.py --mode fsdp \
      --hidden-dim 2048 --num-layers 4 --batch-size 512
"""
import argparse
import gc
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../complete_verifier"))

INPUT_DIM = 784
OUTPUT_DIM = 10
EPS = 0.02


def make_mlp(input_dim, hidden_dim, num_layers, output_dim):
    layers = [nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def export_onnx(model, input_dim, path):
    dummy = torch.randn(1, input_dim)
    torch.onnx.export(
        model.cpu(), dummy, path,
        input_names=["input"], output_names=["output"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )


def make_vnnlib(input_dim, eps, target_class, path):
    """Generate a simple robustness vnnlib spec in DNF format.

    Asserts that output[target_class] >= output[j] for all j != target_class,
    given L-inf perturbation of size eps around a random input.
    """
    torch.manual_seed(0)
    x0 = torch.rand(input_dim).clamp(eps, 1 - eps)
    with open(path, "w") as f:
        for i in range(input_dim):
            f.write(f"(declare-const X_{i} Real)\n")
        for i in range(OUTPUT_DIM):
            f.write(f"(declare-const Y_{i} Real)\n")
        for i in range(input_dim):
            lo = max(0.0, x0[i].item() - eps)
            hi = min(1.0, x0[i].item() + eps)
            f.write(f"(assert (<= X_{i} {hi:.6f}))\n")
            f.write(f"(assert (>= X_{i} {lo:.6f}))\n")
        # DNF: disjunction of single-element conjunctions
        # Each clause says "non-target class j beats target" → property violated
        clauses = []
        for j in range(OUTPUT_DIM):
            if j != target_class:
                clauses.append(f"(and (>= Y_{j} Y_{target_class}))")
        f.write(f"(assert (or {''.join(clauses)}))\n")


def weight_memory_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "fsdp"], required=True)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bab-timeout", type=int, default=60)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.mode == "fsdp" and world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if rank != 0:
            sys.stdout = open(os.devnull, "w")

    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)

    model = make_mlp(INPUT_DIM, args.hidden_dim, args.num_layers, OUTPUT_DIM)
    w_mb = weight_memory_mb(model)

    tmpdir = tempfile.mkdtemp(prefix="bab_fsdp_")
    onnx_path = os.path.join(tmpdir, "model.onnx")
    vnnlib_path = os.path.join(tmpdir, "prop.vnnlib")

    export_onnx(model, INPUT_DIM, onnx_path)

    # Run model on the reference input to find predicted class
    torch.manual_seed(0)
    x0 = torch.rand(1, INPUT_DIM).clamp(EPS, 1 - EPS)
    with torch.no_grad():
        pred = model(x0).argmax(dim=1).item()
    make_vnnlib(INPUT_DIM, EPS, pred, vnnlib_path)

    del model
    gc.collect()

    if rank == 0:
        print(f"Model: MLP h={args.hidden_dim} d={args.num_layers}")
        print(f"Weight memory: {w_mb:.1f} MB")
        print(f"Mode: {args.mode} (world_size={world_size})")
        print(f"BaB batch_size={args.batch_size}, timeout={args.bab_timeout}s")
        print(f"ONNX: {onnx_path}")
        print(f"VNNLIB: {vnnlib_path}")
        print()

    # Build abcrown config as YAML
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(f"""model:
  onnx_path: {onnx_path}
specification:
  vnnlib_path: {vnnlib_path}
solver:
  batch_size: {args.batch_size}
  beta-crown:
    iteration: 20
    lr_beta: 0.03
    lr_alpha: 0.1
  alpha-crown:
    iteration: 20
    lr_alpha: 0.1
attack:
  pgd_order: skip
bab:
  timeout: {args.bab_timeout}
general:
  device: cuda
  complete_verifier: bab
""")

    torch.cuda.reset_peak_memory_stats(dev)

    from abcrown import ABCROWN
    abcrown = ABCROWN(args=["--config", config_path])
    abcrown.main()

    peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    if rank != 0:
        sys.stdout = sys.__stdout__

    if args.mode == "fsdp" and world_size > 1:
        dist.barrier()
        print(f"[Rank {rank}] Peak GPU: {peak_mb:.1f} MB  (weights={w_mb:.1f} MB)")
        dist.destroy_process_group()
    else:
        print(f"Peak GPU: {peak_mb:.1f} MB  (weights={w_mb:.1f} MB)")


if __name__ == "__main__":
    main()
