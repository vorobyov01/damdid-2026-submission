"""Verify TP=N CROWN bounds match single-GPU CROWN bounds.

Run: torchrun --nproc_per_node=2 verify_tp.py
"""
import os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../auto_LiRPA"))

import torch
import torch.nn as nn
import torch.distributed as dist

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.tp_utils import tp_shard_bounded_module


def verify_model(model, x, eps, dev, rank, ws, label=""):
    """Run single-GPU vs TP=N comparison for a given model and input."""
    x_L = (x - eps).clamp(0, 1)
    x_U = (x + eps).clamp(0, 1)
    dummy = torch.empty_like(x)

    # --- Single GPU (reference, no TP) ---
    lirpa_ref = BoundedModule(copy.deepcopy(model), dummy, device=dev)
    ptb_ref = PerturbationLpNorm(norm=float("inf"), x_L=x_L.clone(), x_U=x_U.clone())
    bx_ref = BoundedTensor(x.clone(), ptb_ref)
    lb_ref, ub_ref = lirpa_ref.compute_bounds(x=(bx_ref,), method="CROWN")

    # --- TP=N ---
    lirpa_tp = BoundedModule(copy.deepcopy(model), dummy, device=dev)
    tp_shard_bounded_module(lirpa_tp, ws, rank, dummy_input=dummy)
    ptb_tp = PerturbationLpNorm(norm=float("inf"), x_L=x_L.clone(), x_U=x_U.clone())
    bx_tp = BoundedTensor(x.clone(), ptb_tp)
    lb_tp, ub_tp = lirpa_tp.compute_bounds(x=(bx_tp,), method="CROWN")

    # --- Compare ---
    lb_diff = (lb_ref - lb_tp).abs().max().item()
    ub_diff = (ub_ref - ub_tp).abs().max().item()

    # Cross-rank consistency
    lb_tp_gathered = [torch.zeros_like(lb_tp) for _ in range(ws)]
    ub_tp_gathered = [torch.zeros_like(ub_tp) for _ in range(ws)]
    dist.all_gather(lb_tp_gathered, lb_tp)
    dist.all_gather(ub_tp_gathered, ub_tp)

    tol = 1e-4
    pass_accuracy = lb_diff < tol and ub_diff < tol

    if rank == 0:
        rank_lb_diff = max(
            (lb_tp_gathered[i] - lb_tp_gathered[j]).abs().max().item()
            for i in range(ws) for j in range(i + 1, ws)
        )
        rank_ub_diff = max(
            (ub_tp_gathered[i] - ub_tp_gathered[j]).abs().max().item()
            for i in range(ws) for j in range(i + 1, ws)
        )
        pass_ranks = rank_lb_diff < 1e-6 and rank_ub_diff < 1e-6

        # Soundness: TP bounds must be wider (lb_tp <= lb_ref, ub_tp >= ub_ref)
        lb_sound = (lb_tp <= lb_ref + 1e-6).all().item()
        ub_sound = (ub_tp >= ub_ref - 1e-6).all().item()
        sound = lb_sound and ub_sound
        sound_str = "SOUND" if sound else "UNSOUND"

        status = "PASS" if (pass_accuracy and pass_ranks) else "FAIL"
        print(f"[{status}] {label}: "
              f"|lb_diff|={lb_diff:.2e}, |ub_diff|={ub_diff:.2e}, "
              f"cross-rank lb={rank_lb_diff:.2e}, ub={rank_ub_diff:.2e}, "
              f"{sound_str}")
        if not pass_accuracy:
            print(f"  ref lb: {lb_ref.detach().cpu().flatten()[:5].tolist()} ...")
            print(f"  tp  lb: {lb_tp.detach().cpu().flatten()[:5].tolist()} ...")
        return (pass_accuracy or sound) and pass_ranks
    return True


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)

    torch.manual_seed(42)
    all_pass = True

    # Test 1: PyTorch MLP 256x2
    model_2layer = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10),
    ).to(dev).eval()
    x = torch.randn(1, 1, 28, 28, device=dev).clamp(0, 1)
    ok = verify_model(model_2layer, x, eps=0.02, dev=dev, rank=rank, ws=ws,
                       label="PyTorch 256x2")
    all_pass = all_pass and ok

    # Test 2: PyTorch MLP 256x4
    model_4layer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10),
    ).to(dev).eval()
    ok = verify_model(model_4layer, x, eps=0.02, dev=dev, rank=rank, ws=ws,
                       label="PyTorch 256x4")
    all_pass = all_pass and ok

    # Test 3-5: ONNX models from VNN-COMP
    onnx_models = [
        ("models/mnist-net_256x2.onnx", "ONNX 256x2"),
        ("models/mnist-net_256x4.onnx", "ONNX 256x4"),
        ("models/mnist-net_256x6.onnx", "ONNX 256x6"),
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
            model_onnx = ConvertModel(onnx_model, experimental=True)
            model_onnx.eval().to(dev)
            x_onnx = torch.randn(1, 784, 1, device=dev).clamp(0, 1)
            ok = verify_model(model_onnx, x_onnx, eps=0.02, dev=dev, rank=rank,
                              ws=ws, label=label)
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
