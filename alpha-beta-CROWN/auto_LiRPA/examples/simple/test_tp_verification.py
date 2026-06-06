"""
Test script for Tensor Parallel bound computation.

This script demonstrates distributed verification using Tensor Parallelism.
Run with: torchrun --nproc_per_node=2 test_tp_verification.py

For single GPU testing, just run: python test_tp_verification.py
"""
import os
import sys

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done before importing torch
try:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

import torch
import torch.distributed as dist

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from tp_model import SimpleTPModel, register_tp_custom_ops


def run_worker(rank, world_size):
    """Worker function for distributed execution."""
    # Only set MASTER_ADDR/PORT when not launched by torchrun (which sets them).
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    if rank == 0:
        print(f"Initialized distributed: world_size={world_size}, backend={backend}")
    
    register_tp_custom_ops()
    model = SimpleTPModel(input_dim=2, hidden_dim=4, output_dim=1).to(device)
    
    # Create input
    x = torch.tensor([[1.0, 1.0]], device=device)
    lower = torch.tensor([[-1.0, -2.0]], device=device)
    upper = torch.tensor([[2.0, 1.0]], device=device)
    
    if rank == 0:
        print(f"\nInput: {x.cpu()}")
        print(f"Input bounds: [{lower.cpu()}, {upper.cpu()}]")
    
    try:
        # Wrap with auto_LiRPA + registered TP custom ops.
        lirpa_model = BoundedModule(
            model, 
            torch.empty_like(x), 
            device=device
        )
        
        if rank == 0:
            print("\nModel wrapped successfully!")
        
        # Regular forward pass
        pred = model(x)
        if rank == 0:
            print(f"Model prediction: {pred.cpu().item():.4f}")
        
        # Define perturbation
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
        bounded_x = BoundedTensor(x, ptb)
        
        # Compute bounds using IBP (simplest method)
        if rank == 0:
            print("\nComputing bounds with IBP...")
        
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
        
        if rank == 0:
            print(f"IBP bounds: lower={lb.cpu().item():.4f}, upper={ub.cpu().item():.4f}")
        
        # Try CROWN
        if rank == 0:
            print("\nComputing bounds with CROWN...")
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
        if rank == 0:
            print(f"CROWN bounds: lower={lb.cpu().item():.4f}, upper={ub.cpu().item():.4f}")
        
        if rank == 0:
            print("\n✓ Test completed successfully!")
    
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        dist.destroy_process_group()


def main():
    """Main function.

    Supports two launch modes:
      1. torchrun --nproc_per_node=2 test_tp_verification.py
         (torchrun sets LOCAL_RANK; we init dist and call run_worker directly)
      2. python test_tp_verification.py
         (spawns child processes manually via mp.spawn)
    """
    # --- torchrun mode: LOCAL_RANK is set by torchrun ---
    if 'LOCAL_RANK' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        run_worker(rank, world_size)
        return

    # --- manual spawn mode ---
    world_size = 2

    # Check available GPUs
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if available_gpus < world_size:
            print(f"Warning: Only {available_gpus} GPU(s) available, but {world_size} required.")
            print("Falling back to single-process mode...")
            world_size = 1

    if world_size == 1:
        # Single GPU/CPU mode for testing
        print("Running in single-process TP mode (world_size=1)...")
        print("For multi-process TP testing, use: torchrun --nproc_per_node=2 test_tp_verification.py\n")

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        register_tp_custom_ops()
        model = SimpleTPModel(input_dim=2, hidden_dim=4, output_dim=1).to(device)
        x = torch.tensor([[1.0, 1.0]], device=device)
        lower = torch.tensor([[-1.0, -2.0]], device=device)
        upper = torch.tensor([[2.0, 1.0]], device=device)

        lirpa_model = BoundedModule(model, torch.empty_like(x), device=device)
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
        bounded_x = BoundedTensor(x, ptb)

        pred = model(x)
        print(f"Model prediction: {pred.item():.4f}")

        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
        print(f"IBP bounds: lower={lb.item():.4f}, upper={ub.item():.4f}")

        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
        print(f"CROWN bounds: lower={lb.item():.4f}, upper={ub.item():.4f}")

        dist.destroy_process_group()
    else:
        # Multi-GPU mode via mp.spawn
        print(f"Running distributed test with {world_size} processes...")
        print("NOTE: For better CUDA support, use: torchrun --nproc_per_node=2 test_tp_verification.py")

        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')
        processes = []
        for rank in range(world_size):
            p = ctx.Process(target=run_worker, args=(rank, world_size))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        all_success = True
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                print(f"Error: Process {i} exited with code {p.exitcode}")
                all_success = False
            else:
                print(f"Process {i} completed successfully")

        if not all_success:
            print("\nSome processes failed. Consider using torchrun instead:")
            print("  torchrun --nproc_per_node=2 test_tp_verification.py")
            sys.exit(1)


if __name__ == '__main__':
    main()


