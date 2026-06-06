"""Verify FSDP CROWN bounds match single-GPU CROWN bounds exactly.

Run:  torchrun --nproc_per_node=2 verify_fsdp.py
"""
import os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../auto_LiRPA"))

import torch
import torch.nn as nn
import torch.distributed as dist

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.fsdp_utils import fsdp_shard_bounded_module, fsdp_free_gathered_weights


def verify_model(model, x, eps, dev, rank, ws, label="", method="CROWN"):
    """Compare single-GPU vs FSDP bounds for a model."""
    x_L = (x - eps).clamp(0, 1)
    x_U = (x + eps).clamp(0, 1)
    dummy = torch.empty_like(x)

    # --- Reference: single GPU, no FSDP ---
    lirpa_ref = BoundedModule(copy.deepcopy(model), dummy, device=dev)
    ptb_ref = PerturbationLpNorm(norm=float("inf"), x_L=x_L.clone(), x_U=x_U.clone())
    bx_ref = BoundedTensor(x.clone(), ptb_ref)
    lb_ref, ub_ref = lirpa_ref.compute_bounds(x=(bx_ref,), method=method)

    # --- FSDP ---
    lirpa_fsdp = BoundedModule(copy.deepcopy(model), dummy, device=dev)
    fsdp_shard_bounded_module(lirpa_fsdp, ws, rank, dummy_input=dummy)
    ptb_fsdp = PerturbationLpNorm(norm=float("inf"), x_L=x_L.clone(), x_U=x_U.clone())
    bx_fsdp = BoundedTensor(x.clone(), ptb_fsdp)
    lb_fsdp, ub_fsdp = lirpa_fsdp.compute_bounds(x=(bx_fsdp,), method=method)

    # Free gathered weights to test the cleanup path
    fsdp_free_gathered_weights(lirpa_fsdp)

    # --- Compare ---
    lb_diff = (lb_ref - lb_fsdp).abs().max().item()
    ub_diff = (ub_ref - ub_fsdp).abs().max().item()

    # Cross-rank consistency
    lb_gathered = [torch.zeros_like(lb_fsdp) for _ in range(ws)]
    ub_gathered = [torch.zeros_like(ub_fsdp) for _ in range(ws)]
    dist.all_gather(lb_gathered, lb_fsdp)
    dist.all_gather(ub_gathered, ub_fsdp)

    tol = 1e-5
    ok = True
    if rank == 0:
        rank_lb_diff = max(
            (lb_gathered[i] - lb_gathered[j]).abs().max().item()
            for i in range(ws) for j in range(i + 1, ws)
        )
        rank_ub_diff = max(
            (ub_gathered[i] - ub_gathered[j]).abs().max().item()
            for i in range(ws) for j in range(i + 1, ws)
        )
        pass_acc = lb_diff < tol and ub_diff < tol
        pass_ranks = rank_lb_diff < 1e-6 and rank_ub_diff < 1e-6

        status = "PASS" if (pass_acc and pass_ranks) else "FAIL"
        print(f"[{status}] {label} ({method}): "
              f"|lb_diff|={lb_diff:.2e}, |ub_diff|={ub_diff:.2e}, "
              f"cross-rank lb={rank_lb_diff:.2e}, ub={rank_ub_diff:.2e}")
        if not pass_acc:
            print(f"  ref lb: {lb_ref.detach().cpu().flatten()[:5].tolist()}")
            print(f"  fsdp lb: {lb_fsdp.detach().cpu().flatten()[:5].tolist()}")
        ok = pass_acc and pass_ranks
    return ok


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(42)

    all_pass = True

    # --- Test 1: MLP 256x2 (1 hidden layer) ---
    model_2 = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10),
    ).to(dev).eval()
    x = torch.randn(1, 1, 28, 28, device=dev).clamp(0, 1)

    for method in ["IBP", "CROWN"]:
        ok = verify_model(model_2, x, eps=0.02, dev=dev, rank=rank, ws=ws,
                          label="MLP 256x2", method=method)
        all_pass = all_pass and ok

    # --- Test 2: MLP 256x4 (3 hidden layers) ---
    model_4 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10),
    ).to(dev).eval()

    for method in ["IBP", "CROWN"]:
        ok = verify_model(model_4, x, eps=0.02, dev=dev, rank=rank, ws=ws,
                          label="MLP 256x4", method=method)
        all_pass = all_pass and ok

    # --- Test 3: MLP 256x6 (5 hidden layers) ---
    model_6 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10),
    ).to(dev).eval()

    for method in ["IBP", "CROWN"]:
        ok = verify_model(model_6, x, eps=0.02, dev=dev, rank=rank, ws=ws,
                          label="MLP 256x6", method=method)
        all_pass = all_pass and ok

    # --- Test 4: ONNX model from VNN-COMP (if available) ---
    onnx_models = [
        ("../vnncomp_tp/models/mnist-net_256x2.onnx", "ONNX 256x2"),
        ("../vnncomp_tp/models/mnist-net_256x4.onnx", "ONNX 256x4"),
    ]
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
            ok = verify_model(m, x_onnx, eps=0.02, dev=dev, rank=rank, ws=ws,
                              label=label, method="CROWN")
            all_pass = all_pass and ok
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] {label}: {e}")
            all_pass = False

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
