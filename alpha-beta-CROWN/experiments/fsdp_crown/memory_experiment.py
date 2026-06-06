"""FSDP memory experiment: compare GPU memory single-GPU vs FSDP=N.

Creates increasingly large MLP models, runs CROWN compute_bounds
with and without FSDP, and reports peak GPU memory per rank.

Run:  torchrun --nproc_per_node=2 memory_experiment.py
"""
import os, sys, copy, gc, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../auto_LiRPA"))

import torch
import torch.nn as nn
import torch.distributed as dist

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.fsdp_utils import fsdp_shard_bounded_module, fsdp_free_gathered_weights


def _param_bytes_mb(lirpa):
    """Count bytes stored in BoundParams (weight shards or full weights).

    This is the correct baseline metric for FSDP: it measures exactly what
    FSDP shards (the weight matrices), ignoring BoundedModule metadata,
    JIT-trace artifacts, and the caller's model copy that stay on GPU
    regardless of sharding.  Expected: single-GPU = W bytes, FSDP=P ~ W/P bytes.
    """
    from auto_LiRPA.operators.leaf import BoundParams
    total = 0
    for node in lirpa.nodes():
        if isinstance(node, BoundParams):
            total += node.param.data.numel() * node.param.data.element_size()
    return total / (1024 ** 2)


def measure_bounds(model, x, eps, dev, use_fsdp=False, ws=1, rank=0, method="CROWN"):
    """Run compute_bounds and return (lb, ub, peak_memory_MB, param_memory_MB).

    Baseline (param_memory_MB) is counted directly from BoundParams bytes so
    that the comparison is not polluted by BoundedModule metadata or the
    caller's model copy that is present on GPU for both modes.
    Peak memory is measured only during compute_bounds.
    """
    gc.collect()
    torch.cuda.empty_cache()

    x_L = (x - eps).clamp(0, 1)
    x_U = (x + eps).clamp(0, 1)
    dummy = torch.empty_like(x)

    lirpa = BoundedModule(copy.deepcopy(model), dummy, device=dev)

    if use_fsdp and ws > 1:
        fsdp_shard_bounded_module(lirpa, ws, rank, dummy_input=dummy)

    ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_L.clone(), x_U=x_U.clone())
    bx = BoundedTensor(x.clone(), ptb)

    # Baseline: bytes actually stored in BoundParams (full weights vs shards).
    mem_before = _param_bytes_mb(lirpa)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    lb, ub = lirpa.compute_bounds(x=(bx,), method=method)

    peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    if use_fsdp:
        fsdp_free_gathered_weights(lirpa)

    del lirpa
    gc.collect()
    torch.cuda.empty_cache()
    return lb, ub, peak_mb, mem_before


def run_experiment(model, x, eps, dev, rank, ws, label, method="CROWN"):
    """Run single-GPU and FSDP, compare bounds and memory."""
    # FSDP first (runs on clean GPU, no prior allocation residue)
    lb_fsdp, ub_fsdp, peak_fsdp, base_fsdp = measure_bounds(
        model, x, eps, dev, use_fsdp=True, ws=ws, rank=rank, method=method)

    # Single GPU reference (runs after FSDP, but that's OK for correctness)
    lb_ref, ub_ref, peak_single, base_single = measure_bounds(
        model, x, eps, dev, use_fsdp=False, ws=1, rank=0, method=method)

    # Cross-rank consistency
    lb_all = [torch.zeros_like(lb_fsdp) for _ in range(ws)]
    ub_all = [torch.zeros_like(ub_fsdp) for _ in range(ws)]
    dist.all_gather(lb_all, lb_fsdp)
    dist.all_gather(ub_all, ub_fsdp)

    stats = torch.tensor([peak_fsdp, base_fsdp], device=dev)
    stats_all = [torch.zeros(2, device=dev) for _ in range(ws)]
    dist.all_gather(stats_all, stats)

    if rank == 0:
        lb_diff = (lb_ref - lb_fsdp).abs().max().item()
        ub_diff = (ub_ref - ub_fsdp).abs().max().item()
        rank_diff = max(
            (lb_all[i] - lb_all[j]).abs().max().item()
            for i in range(ws) for j in range(i + 1, ws)
        )
        max_peak_fsdp = max(s[0].item() for s in stats_all)
        max_base_fsdp = max(s[1].item() for s in stats_all)
        peak_savings = (1.0 - max_peak_fsdp / peak_single) * 100 if peak_single > 0 else 0
        base_savings = (1.0 - max_base_fsdp / base_single) * 100 if base_single > 0 else 0

        ok = lb_diff < 1e-5 and ub_diff < 1e-5 and rank_diff < 1e-6
        status = "PASS" if ok else "FAIL"

        print(f"[{status}] {label} ({method})")
        print(f"  Bounds: |lb_diff|={lb_diff:.2e}, |ub_diff|={ub_diff:.2e}, "
              f"cross-rank={rank_diff:.2e}")
        print(f"  Baseline memory (BoundParams bytes):      "
              f"single={base_single:.1f} MB, FSDP={max_base_fsdp:.1f} MB, "
              f"savings={base_savings:.1f}%")
        print(f"  Peak memory (during compute_bounds):     "
              f"single={peak_single:.1f} MB, FSDP={max_peak_fsdp:.1f} MB, "
              f"savings={peak_savings:.1f}%")
        return {
            "label": label, "method": method, "status": status,
            "lb_diff": lb_diff, "ub_diff": ub_diff,
            "base_single": base_single, "base_fsdp": max_base_fsdp,
            "base_savings_pct": base_savings,
            "peak_single": peak_single, "peak_fsdp": max_peak_fsdp,
            "peak_savings_pct": peak_savings,
        }
    return None


def make_mlp(input_dim, hidden_dim, num_layers, output_dim=10):
    layers = [nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(42)

    results = []
    eps = 0.02

    # --- Experiment 1: fixed depth, growing width ---
    if rank == 0:
        print("=" * 60)
        print("Experiment 1: Fixed depth (4 layers), growing width")
        print("=" * 60)

    for hidden in [256, 1024, 4096, 8192]:
        model = make_mlp(784, hidden, 4).to(dev).eval()
        x = torch.randn(1, 1, 28, 28, device=dev).clamp(0, 1)
        r = run_experiment(model, x, eps, dev, rank, ws,
                          label=f"MLP h={hidden} d=4", method="CROWN")
        if r:
            results.append(r)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            print()

    # --- Experiment 2: fixed width, growing depth ---
    if rank == 0:
        print("=" * 60)
        print("Experiment 2: Fixed width (4096), growing depth")
        print("=" * 60)

    for depth in [2, 4, 6, 8]:
        model = make_mlp(784, 4096, depth).to(dev).eval()
        x = torch.randn(1, 1, 28, 28, device=dev).clamp(0, 1)
        r = run_experiment(model, x, eps, dev, rank, ws,
                          label=f"MLP h=4096 d={depth}", method="CROWN")
        if r:
            results.append(r)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            print()

    # --- Experiment 3: VNN-COMP ONNX models ---
    onnx_models = [
        ("../vnncomp_tp/models/mnist-net_256x2.onnx", "ONNX 256x2"),
        ("../vnncomp_tp/models/mnist-net_256x4.onnx", "ONNX 256x4"),
        ("../vnncomp_tp/models/mnist-net_256x6.onnx", "ONNX 256x6"),
    ]

    if rank == 0:
        print("=" * 60)
        print("Experiment 3: VNN-COMP ONNX models")
        print("=" * 60)

    for onnx_path, label in onnx_models:
        if not os.path.exists(onnx_path):
            if rank == 0:
                print(f"[SKIP] {label}: {onnx_path} not found")
            continue
        try:
            import onnx
            from onnx2pytorch import ConvertModel
            onnx_model = onnx.load(onnx_path)
            m = ConvertModel(onnx_model, experimental=True).eval().to(dev)
            x_onnx = torch.randn(1, 784, 1, device=dev).clamp(0, 1)
            r = run_experiment(m, x_onnx, eps, dev, rank, ws,
                              label=label, method="CROWN")
            if r:
                results.append(r)
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] {label}: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            print()

    # --- Summary ---
    if rank == 0 and results:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Model':<25} {'Status':<6} "
              f"{'Base S':>8} {'Base F':>8} {'Base%':>6} "
              f"{'Peak S':>8} {'Peak F':>8} {'Peak%':>6}")
        print("-" * 85)
        for r in results:
            print(f"{r['label']:<25} {r['status']:<6} "
                  f"{r['base_single']:>8.1f} {r['base_fsdp']:>8.1f} "
                  f"{r['base_savings_pct']:>5.1f}% "
                  f"{r['peak_single']:>8.1f} {r['peak_fsdp']:>8.1f} "
                  f"{r['peak_savings_pct']:>5.1f}%")

        with open("fsdp_memory_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to fsdp_memory_results.json")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
