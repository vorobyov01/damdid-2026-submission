"""Fully-Sharded Data Parallelism (FSDP) utilities for auto_LiRPA verification.

FSDP shards each BoundLinear and BoundConv's weight across GPUs.  Before any
operation that needs the full weight (IBP forward, CROWN backward), the weight
is reconstructed via AllGather *per-layer*, then freed immediately after.
At any point, only one full weight is in GPU memory.

Sharded layer types:
    - BoundLinear: weight shape [out_features, in_features], shard dim 0.
    - BoundConv:   weight shape [out_channels, in_channels, kH, kW],
                   shard dim 0 (out_channels).

Bias tensors are not sharded (they are 1D and skipped by the shape check).
Both layer types share the same shard / gather / free logic because dim 0
is the natural ``out`` axis in both cases.

Unlike Tensor Parallelism, FSDP introduces *no* accuracy loss -- the
computation is mathematically identical to single-GPU.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def fsdp_shard_bounded_module(model: 'BoundedModule', world_size: int,
                              rank: int, dummy_input: torch.Tensor = None):
    """Shard BoundLinear weights across *world_size* GPUs.

    After sharding, each BoundParams holding a weight matrix stores only
    1/world_size of the rows.  Per-layer AllGather/free hooks ensure that
    only one layer's full weight is in memory at a time during compute_bounds.

    Args:
        model: A ``BoundedModule`` already constructed with full weights.
        world_size: Total number of GPUs.
        rank: Index of the current GPU (0-based).
        dummy_input: A tensor with the correct input shape, used to
            refresh the computation graph shapes after sharding.
    """
    from .operators.linear import BoundLinear
    from .operators.convolution import BoundConv
    from .operators.leaf import BoundParams

    shardable_types = (BoundLinear, BoundConv)
    sharded_count = 0
    sharded_bytes = 0
    for node in model.nodes():
        if not isinstance(node, shardable_types):
            continue
        if len(node.inputs) < 2:
            continue
        w_node = node.inputs[1]
        if not isinstance(w_node, BoundParams):
            continue

        # Guard against double sharding when a BoundParams is referenced
        # by more than one consumer (rare, but possible).
        if getattr(w_node, '_fsdp_world_size', 0) > 1:
            continue

        W = w_node.param.data  # Linear: [out, in].  Conv: [out_c, in_c, kH, kW].
        if W.ndim < 2 or W.shape[0] % world_size != 0:
            continue

        chunk = W.shape[0] // world_size
        shard = W[rank * chunk : (rank + 1) * chunk].contiguous()

        sharded_bytes += (W.numel() - shard.numel()) * W.element_size()

        w_node.param = nn.Parameter(shard, requires_grad=W.requires_grad)
        w_node._fsdp_world_size = world_size
        w_node._fsdp_rank = rank
        w_node._fsdp_shard_dim = 0

        w_node.forward_value = None

        sharded_count += 1

    if rank == 0:
        print(f"[FSDP] Sharded {sharded_count} weight tensors across "
              f"{world_size} GPUs (saved {sharded_bytes / 2**20:.1f} MB / rank)")

    # Refresh graph shapes with full weights, then free them.
    if dummy_input is not None:
        _refresh_graph_shapes(model, dummy_input)
        fsdp_free_gathered_weights(model)


def _refresh_graph_shapes(model: 'BoundedModule', dummy_input: torch.Tensor):
    """Run a forward pass to update output_shape / forward_value metadata."""
    with torch.no_grad():
        model.forward(dummy_input)


_OFFLOAD_STATS = {'calls': 0, 'bytes': 0}


def fsdp_gather_node(node):
    """AllGather a sharded BoundParams to full size.

    Sets forward_value, lower, upper to the gathered full weight so that
    downstream BoundLinear can use it for interval_propagate or bound_backward.
    """
    ws = getattr(node, '_fsdp_world_size', 0)
    if ws <= 1:
        return
    if getattr(node, 'forward_value', None) is not None:
        return  # already gathered
    if getattr(node, '_cpu_offload', False):
        # Naive baseline: the weight lives in pinned host memory and is staged
        # to the device per layer instead of being AllGathered from shards.
        full = node.param.data.to('cuda', non_blocking=True)
        node.forward_value = full
        node.lower = full
        node.upper = full
        node.interval = (full, full)
        _OFFLOAD_STATS['calls'] += 1
        _OFFLOAD_STATS['bytes'] += full.numel() * full.element_size()
        return
    param = node.param.data
    with torch.no_grad():
        # Single flat destination buffer: all_gather_into_tensor concatenates
        # along dim 0, which is the shard dimension for BoundLinear
        # ([out, in]) and BoundConv ([out_c, in_c, kH, kW]) alike.  Building
        # the full weight with torch.cat instead would allocate it twice and,
        # because cat is recorded by autograd, would keep every gathered
        # weight of the pass reachable from the returned bounds.
        assert node._fsdp_shard_dim == 0
        full = torch.empty((param.shape[0] * ws, *param.shape[1:]),
                           dtype=param.dtype, device=param.device)
        dist.all_gather_into_tensor(full, param.contiguous())
    node.forward_value = full
    node.lower = full
    node.upper = full
    node.interval = (full, full)


def fsdp_free_node(node):
    """Free gathered weight on a BoundParams, returning to sharded state.

    We *delete* the cached attributes (rather than setting them to None) so
    that downstream ``hasattr(...)`` checks correctly trigger a re-gather on
    the next access (e.g. when init_alpha runs IBP multiple times, or when a
    BoundParams feeds into more than one Conv/Linear layer).
    """
    ws = getattr(node, '_fsdp_world_size', 0)
    if ws <= 1:
        return
    for name in ('forward_value', 'interval'):
        try:
            delattr(node, name)
        except AttributeError:
            pass
    # lower / upper are properties over _lower / _upper, so delattr on them
    # raises AttributeError and would leave the gathered full weight alive.
    node.delete_lower_and_upper_bounds()


def fsdp_free_gathered_weights(model: 'BoundedModule'):
    """Free full-sized forward_values cached during compute_bounds.

    Call this between compute_bounds invocations to release GPU memory
    occupied by AllGathered weight copies.  The next compute_bounds will
    re-gather them on demand.
    """
    from .operators.leaf import BoundParams

    for node in model.nodes():
        if isinstance(node, BoundParams) and getattr(node, '_fsdp_world_size', 0) > 1:
            fsdp_free_node(node)
