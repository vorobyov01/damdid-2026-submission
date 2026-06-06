"""
Simple Tensor Parallel model for testing distributed bound computation.
"""
import torch
import torch.nn as nn
import torch.distributed as dist


class TPLinearColOp(torch.autograd.Function):
    """Column-parallel linear op with ONNX symbolic for auto_LiRPA custom op mapping."""

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
    """Row-parallel linear op with local matmul and cross-rank AllReduce in forward."""

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


class ColumnParallelLinear(nn.Module):
    """
    Column Parallel Linear layer - splits weights along output dimension.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Split output dimension
        assert out_features % self.world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({self.world_size})"
        self.local_out_features = out_features // self.world_size
        
        # Create local weight (sharded)
        self.weight = nn.Parameter(
            torch.randn(self.local_out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(self.local_out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Keep column-parallel activations sharded to avoid extra communication.
        return TPLinearColOp.apply(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    Row Parallel Linear layer - splits weights along input dimension.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Split input dimension
        assert in_features % self.world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({self.world_size})"
        self.local_in_features = in_features // self.world_size
        
        # Create local weight (sharded)
        self.weight = nn.Parameter(
            torch.randn(out_features, self.local_in_features)
        )
        if bias:
            # Bias is replicated (not sharded) for Row Parallel
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # If full activations are provided, take this rank shard.
        if not torch.jit.is_tracing() and dist.is_initialized() and self.world_size > 1 and x.size(-1) == self.in_features:
            local_x = x[..., self.rank * self.local_in_features:(self.rank + 1) * self.local_in_features]
        else:
            local_x = x
        return TPLinearRowOp.apply(local_x, self.weight, self.bias)


class SimpleTPModel(nn.Module):
    """
    Simple 2-layer MLP with Tensor Parallelism.
    Architecture: Input -> ColumnParallel (expand) -> ReLU -> RowParallel (compress) -> Output
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        super().__init__()
        # Column Parallel: expands dimension
        self.layer1 = ColumnParallelLinear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Row Parallel: compresses dimension
        self.layer2 = RowParallelLinear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


