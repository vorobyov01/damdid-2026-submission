"""
Test script for Tensor Parallel bound computation using torchrun.

This is the recommended way to run distributed tests.
Usage: torchrun --nproc_per_node=2 test_tp_torchrun.py
"""
import os
import sys
import torch
import torch.distributed as dist

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from tp_model import SimpleTPModel, register_tp_custom_ops


def main():
    """Main function - runs on each process."""
    # Initialize distributed (torchrun handles this automatically)
    if not dist.is_initialized():
        # Fallback for single-process mode
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    if torch.cuda.is_available() and world_size > 1:
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        backend = 'nccl'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        backend = 'gloo'
    
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
        
        # Compute bounds using IBP
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
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()


