"""Domain-parallel utilities for Branch-and-Bound verification.

Splits a batch of BaB domains across multiple GPUs so each GPU runs
update_bounds on batch/N domains, then gathers the results back.
Combined with FSDP weight sharding, this gives both memory savings
and ~Nx speedup.
"""

import pickle

import torch
import torch.distributed as dist


def _chunk_range(total, rank, world_size):
    """Return (start, end) indices for rank's chunk of total items."""
    chunk = total // world_size
    remainder = total % world_size
    if rank < remainder:
        start = rank * (chunk + 1)
        end = start + chunk + 1
    else:
        start = remainder * (chunk + 1) + (rank - remainder) * chunk
        end = start + chunk
    return start, end


def _ensure_contiguous(t):
    """Call .contiguous() on tensors to avoid TorchScript JIT fuser issues."""
    return t.contiguous() if isinstance(t, torch.Tensor) and not t.is_contiguous() else t


def scatter_domain_dict(d, rank, world_size):
    """Slice domain dict d so that this rank gets its portion of the batch.

    All ranks are assumed to hold the identical full d (from identical
    pick_out + build_history_and_set_bounds).  No communication needed —
    each rank just takes its own slice.

    Returns a new dict with the same keys but batch-sliced values.
    The original d is NOT modified.
    """
    batch = _get_batch_size(d)
    lo, hi = _chunk_range(batch, rank, world_size)

    if lo >= hi:
        # This rank got 0 domains — give it the last domain as a dummy
        # so update_bounds doesn't crash on empty tensors.
        # The caller must trim padded results after gather.
        lo, hi = batch - 1, batch

    out = {}
    for key, val in d.items():
        if key in ('lower_bounds', 'upper_bounds', 'lAs'):
            out[key] = {k: _ensure_contiguous(v[lo:hi]) for k, v in val.items()}
        elif key == 'alphas':
            out[key] = {
                outer_k: {
                    inner_k: _ensure_contiguous(v[:, :, lo:hi])
                    for inner_k, v in outer_v.items()
                }
                for outer_k, outer_v in val.items()
            }
        elif key == 'mask':
            if val is None:
                out[key] = None
            else:
                out[key] = {k: _ensure_contiguous(v[lo:hi]) for k, v in val.items()}
        elif key in ('betas', 'intermediate_betas', 'history',
                     'split_history', 'depths'):
            out[key] = val[lo:hi]
        elif isinstance(val, torch.Tensor):
            out[key] = _ensure_contiguous(val[lo:hi])
        elif val is None:
            out[key] = None
        else:
            out[key] = val
    return out


def gather_result_dict(local_ret, world_size):
    """AllGather result dicts from all ranks, concatenating along batch dim.

    Each rank calls this with its local_ret from update_bounds.
    Returns the full gathered result dict on every rank.
    """
    if world_size <= 1:
        return local_ret

    out = {}
    for key, val in local_ret.items():
        try:
            if key in ('lower_bounds', 'upper_bounds', 'lAs'):
                out[key] = _gather_tensor_dict(val, world_size, cat_dim=0)
            elif key == 'alphas':
                out[key] = {}
                for outer_k, outer_v in val.items():
                    out[key][outer_k] = _gather_tensor_dict(
                        outer_v, world_size, cat_dim=2)
            elif key == 'unstable_bounds':
                if val is None:
                    out[key] = None
                else:
                    out[key] = {}
                    for k, pair in val.items():
                        out[key][k] = [
                            _gather_tensor(pair[0], world_size, cat_dim=0),
                            _gather_tensor(pair[1], world_size, cat_dim=0),
                        ]
            elif key in ('betas', 'intermediate_betas', 'split_history'):
                out[key] = _gather_list(val, world_size)
            elif isinstance(val, torch.Tensor) and val.device.type == 'cuda':
                out[key] = _gather_tensor(val, world_size, cat_dim=0)
            else:
                out[key] = val
        except Exception as e:
            raise RuntimeError(
                f"gather_result_dict failed on key={key!r}, "
                f"type={type(val).__name__}, "
                f"device={getattr(val, 'device', 'N/A')}: {e}"
            ) from e
    return out


def _get_batch_size(d):
    """Infer batch size from the domain dict."""
    for name in ('thresholds', 'cs', 'global_lb'):
        if name in d and isinstance(d[name], torch.Tensor):
            return d[name].shape[0]
    if 'lower_bounds' in d:
        for v in d['lower_bounds'].values():
            return v.shape[0]
    if 'depths' in d and isinstance(d['depths'], list):
        return len(d['depths'])
    raise RuntimeError("Cannot infer batch size from domain dict")


def _gather_tensor(t, world_size, cat_dim=0):
    """AllGather a tensor across all ranks and concatenate."""
    if t.device.type != 'cuda':
        t = t.to('cuda')
    parts = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(parts, t.contiguous())
    return torch.cat(parts, dim=cat_dim)


def _gather_tensor_dict(td, world_size, cat_dim=0):
    """AllGather each tensor in a dict."""
    return {k: _gather_tensor(v, world_size, cat_dim=cat_dim)
            for k, v in td.items()}


def _gather_list(lst, world_size):
    """AllGather a Python list across ranks via pickled GPU byte tensors.

    Avoids all_gather_object which requires a gloo backend.
    Moves CUDA tensors to CPU before pickling and back to the local
    device after unpickling so cross-device tensor references don't leak.
    """
    cpu_lst = _recursive_to_cpu(lst)
    data = pickle.dumps(cpu_lst)
    size = len(data)
    device = torch.device("cuda")

    size_t = torch.tensor([size], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device)
                 for _ in range(world_size)]
    dist.all_gather(all_sizes, size_t)
    max_size = max(s.item() for s in all_sizes)

    buf = torch.zeros(max_size, dtype=torch.uint8, device=device)
    buf[:size] = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)
    all_bufs = [torch.zeros(max_size, dtype=torch.uint8, device=device)
                for _ in range(world_size)]
    dist.all_gather(all_bufs, buf)

    result = []
    for b, sz in zip(all_bufs, all_sizes):
        part = pickle.loads(bytes(b[:int(sz.item())].cpu().numpy()))
        result.extend(part)
    return _recursive_to_device(result, device)


def _recursive_to_cpu(obj):
    """Recursively move tensors to CPU for safe cross-rank pickling."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _recursive_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_recursive_to_cpu(x) for x in obj]
        return type(obj)(converted)
    return obj


def _recursive_to_device(obj, device):
    """Recursively move tensors to target device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _recursive_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_recursive_to_device(x, device) for x in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj
