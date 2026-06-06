"""Run Alpha-Beta-CROWN incomplete verification with FSDP.

Initializes torch.distributed, sets CUDA device per rank, then launches
abcrown. The FSDP hook in LiRPANet.__init__ detects the distributed
environment and shards BoundParams automatically.

Usage:
  # Single GPU (baseline):
  python abcrown.py --config experiments/fsdp_crown/mnist_256x6.yaml

  # FSDP with 2 GPUs:
  torchrun --nproc_per_node=2 experiments/fsdp_crown/run_abcrown_fsdp.py \
      --config experiments/fsdp_crown/mnist_256x6.yaml
"""
import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../complete_verifier"))

rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

torch.cuda.set_device(rank)

if world_size > 1:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Disable ALL TorchScript JIT fusers to avoid a strides assertion
    # bug in the codegen fuser that triggers only on non-primary GPUs.
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    if rank != 0:
        sys.stdout = open(os.devnull, "w")

torch.cuda.reset_peak_memory_stats()

from abcrown import ABCROWN

abcrown = ABCROWN(args=sys.argv[1:])
abcrown.main()

peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

if world_size > 1:
    if rank != 0:
        sys.stdout = sys.__stdout__
    dist.barrier()
    print(f"[Rank {rank}] Peak GPU memory: {peak_mb:.1f} MB")
    dist.destroy_process_group()
else:
    print(f"Peak GPU memory: {peak_mb:.1f} MB")
