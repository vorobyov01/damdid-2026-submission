"""
Shared model definitions for TP verification experiments.

Contains:
- SimpleDenseModel: standard nn.Linear baseline (single GPU)
- SimpleTPModel + custom ops: Tensor Parallel model (multi-GPU)
- Utility to copy weights from dense model to TP model for reproducible comparison
"""
import torch
import torch.nn as nn
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Dense baseline model
# ---------------------------------------------------------------------------

class SimpleDenseModel(nn.Module):
    """Single-GPU baseline model (no tensor parallelism)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.relu(self.layer1(x)))


# ---------------------------------------------------------------------------
# ONNX custom ops for TP layers
# ---------------------------------------------------------------------------

class TPLinearColOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_output.matmul(weight)
        grad_weight = grad_output.transpose(-1, -2).matmul(x)
        grad_bias = grad_output.sum(dim=0) if bias is not None else None
        return grad_x, grad_weight, grad_bias

    @staticmethod
    def symbolic(g, x, weight, bias):
        return g.op("customOp::TPLinearCol", x, weight, bias)


class TPLinearRowOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)
        out = torch.nn.functional.linear(x, weight, None)
        if not torch.jit.is_tracing() and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, async_op=False)
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_output.matmul(weight)
        grad_weight = grad_output.transpose(-1, -2).matmul(x)
        grad_bias = grad_output.sum(dim=0) if bias is not None else None
        return grad_x, grad_weight, grad_bias

    @staticmethod
    def symbolic(g, x, weight, bias):
        return g.op("customOp::TPLinearRow", x, weight, bias)


def register_tp_custom_ops():
    """Register TP custom ops so BoundedModule maps them to TP-aware bound classes."""
    from auto_LiRPA import register_custom_op
    from auto_LiRPA.operators.tensor_parallel import BoundLinearTP_Col, BoundLinearTP_Row

    register_custom_op("customOp::TPLinearCol", BoundLinearTP_Col)
    register_custom_op("customOp::TPLinearRow", BoundLinearTP_Row)


# ---------------------------------------------------------------------------
# TP layers
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        assert out_features % self.world_size == 0
        self.local_out_features = out_features // self.world_size

        self.weight = nn.Parameter(torch.randn(self.local_out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(self.local_out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return TPLinearColOp.apply(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        assert in_features % self.world_size == 0
        self.local_in_features = in_features // self.world_size

        self.weight = nn.Parameter(torch.randn(out_features, self.local_in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if not torch.jit.is_tracing() and dist.is_initialized() and self.world_size > 1 and x.size(-1) == self.in_features:
            local_x = x[..., self.rank * self.local_in_features:(self.rank + 1) * self.local_in_features]
        else:
            local_x = x
        return TPLinearRowOp.apply(local_x, self.weight, self.bias)


class SimpleTPModel(nn.Module):
    """
    2-layer MLP with Tensor Parallelism.
    Input -> ColumnParallel -> ReLU -> RowParallel -> Output
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        super().__init__()
        self.layer1 = ColumnParallelLinear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = RowParallelLinear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


# ---------------------------------------------------------------------------
# Weight transfer: dense -> TP (for reproducible numerical comparison)
# ---------------------------------------------------------------------------

def copy_dense_weights_to_tp(dense_model: SimpleDenseModel,
                              tp_model: SimpleTPModel) -> None:
    """Copy weights from a dense model into a TP model (world_size=1 only).

    Allows bit-exact comparison of bounds between dense and TP code paths.
    """
    ws = tp_model.layer1.world_size
    rank = tp_model.layer1.rank

    w1 = dense_model.layer1.weight.data
    b1 = dense_model.layer1.bias.data
    H = w1.shape[0]
    local_H = H // ws
    start, end = rank * local_H, (rank + 1) * local_H

    tp_model.layer1.weight.data.copy_(w1[start:end])
    tp_model.layer1.bias.data.copy_(b1[start:end])

    w2 = dense_model.layer2.weight.data
    b2 = dense_model.layer2.bias.data
    local_in = w2.shape[1] // ws
    s2, e2 = rank * local_in, (rank + 1) * local_in

    tp_model.layer2.weight.data.copy_(w2[:, s2:e2])
    tp_model.layer2.bias.data.copy_(b2)
