"""
Tensor Parallelism support for auto_LiRPA bound computation.

This module implements distributed bound propagation using Tensor Parallelism,
inspired by Megatron-LM architecture but adapted for backward bound propagation
(CROWN / α-CROWN).

Key design:
  - AllReduce is implemented via DifferentiableAllReduce (torch.autograd.Function)
    so that α-CROWN gradient optimization works correctly through the TP layer.
  - Forward of AllReduce = SUM across ranks.
  - Backward of AllReduce = identity (∂(Σ x_r)/∂x_r = 1).

Bias handling strategy for CROWN backward:
  In the CROWN backward loop, biases accumulate from all layers. With TP, some
  biases are "full" (same on all ranks) and some are "partial" (different per
  rank, needs AllReduce). Rather than AllReducing biases inside each operator,
  we use a coordinated approach:
    - Col backward: AllReduce A matrices only (NOT biases)
    - Row backward: divide bias by world_size (to make it "partial")
    - backward_general loop: track partial vs full bias, AllReduce partial
      biases when a Col layer is reached (see backward_bound.py)
"""
import torch
import torch.distributed as dist
from .linear import BoundLinear


class DifferentiableAllReduce(torch.autograd.Function):
    """AllReduce that preserves the autograd graph.

    Forward: out = AllReduce(input)  — SUM across all ranks.
    Backward: grad passes through unchanged (identity).

    This is correct because AllReduce(x) = Σ_r x_r, so ∂/∂x_r = 1.
    Each rank already holds the correct local partial gradient.
    """

    @staticmethod
    def forward(ctx, tensor):
        output = tensor.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, async_op=False)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _tp_all_reduce(value):
    """Apply differentiable AllReduce to a tensor, tuple, or list of tensors."""
    if isinstance(value, torch.Tensor):
        return DifferentiableAllReduce.apply(value.contiguous())
    if isinstance(value, (tuple, list)):
        items = []
        for item in value:
            if isinstance(item, torch.Tensor):
                item = DifferentiableAllReduce.apply(item.contiguous())
            items.append(item)
        return type(value)(items)
    return value


def _scale_bias(value, scale):
    """Divide a bias value (tensor, tuple, or float) by a scalar."""
    if isinstance(value, torch.Tensor):
        return value / scale
    if isinstance(value, (tuple, list)):
        return type(value)(
            x / scale if isinstance(x, torch.Tensor) else x
            for x in value
        )
    if isinstance(value, (int, float)) and value != 0:
        return value / scale
    return value


class BoundLinearTP_Col(BoundLinear):
    """
    Column Parallel Linear Layer for Tensor Parallelism.

    In forward pass: weights are split along output dimension (columns).
    In backward CROWN: incoming A matrices are split, requires AllReduce to combine.

    This is used for layers that expand dimension (e.g., MLP expansion layers).
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        if attr is None or 'transB' not in attr:
            self.transB = 1
        self._refresh_dist_state()

    def _refresh_dist_state(self):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.use_tp = True
        else:
            self.use_tp = False
            self.world_size = 1
            self.rank = 0

    def forward(self, x, w, b=None):
        self._refresh_dist_state()
        return super().forward(x, w, b)

    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       reduce_bias=True, **kwargs):
        import os, sys
        _debug = os.environ.get('TP_DEBUG', '') == '1'
        result = super().bound_backward(last_lA, last_uA, *x, start_node=start_node,
                                       reduce_bias=reduce_bias, **kwargs)

        self._refresh_dist_state()

        # Skip AllReduce when start_node is in the SAME sharded zone as
        # this Col.  Those backward passes compute LOCAL per-rank bounds
        # and must not mix A matrices across ranks.  If start_node is in
        # a DIFFERENT zone, AllReduce IS needed (the backward path crosses
        # this Col-Row pair on its way to the input).
        start_zone = getattr(start_node, '_tp_sharded_zone_id', None) if start_node is not None else None
        my_zone = getattr(self, '_tp_sharded_zone_id', None)
        if start_zone is not None and start_zone == my_zone:
            if _debug:
                print(f"[TP rank={self.rank}] Col SKIP AllReduce (same zone={my_zone}, start_node={start_node.name})",
                      flush=True, file=sys.stderr)
            return result

        if self.use_tp and self.world_size > 1:
            lA_x, uA_x = result[0][0]

            if lA_x is not None and isinstance(lA_x, torch.Tensor):
                if _debug:
                    print(f"[TP rank={self.rank}] Col AllReduce lA_x shape={lA_x.shape}", flush=True, file=sys.stderr)
                lA_x = DifferentiableAllReduce.apply(lA_x.contiguous())
                if _debug:
                    print(f"[TP rank={self.rank}] Col AllReduce lA_x DONE", flush=True, file=sys.stderr)
            if uA_x is not None and isinstance(uA_x, torch.Tensor):
                if _debug:
                    print(f"[TP rank={self.rank}] Col AllReduce uA_x shape={uA_x.shape}", flush=True, file=sys.stderr)
                uA_x = DifferentiableAllReduce.apply(uA_x.contiguous())
                if _debug:
                    print(f"[TP rank={self.rank}] Col AllReduce uA_x DONE", flush=True, file=sys.stderr)

            result[0][0] = (lA_x, uA_x)

        return result


class BoundLinearTP_Row(BoundLinear):
    """
    Row Parallel Linear Layer for Tensor Parallelism.

    In forward pass: weights are split along input dimension (rows).
    In backward CROWN: incoming A matrices are replicated, output A is automatically split.

    This is used for layers that compress dimension (e.g., MLP compression layers).
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        if attr is None or 'transB' not in attr:
            self.transB = 1
        self._refresh_dist_state()

    def _refresh_dist_state(self):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.use_tp = True
        else:
            self.use_tp = False
            self.world_size = 1
            self.rank = 0

    def forward(self, x, w, b=None):
        self._refresh_dist_state()
        # Megatron-LM pattern: matmul → AllReduce → add bias.
        # super().forward with b=None computes x @ w^T only.
        res = super().forward(x, w, None)
        if self.use_tp and self.world_size > 1 and isinstance(res, torch.Tensor):
            res = DifferentiableAllReduce.apply(res.contiguous())
        if b is not None:
            if self.beta_linear != 1.0:
                b = self.beta_linear * b
            res = res + b
        return res

    def interval_propagate(self, *v, C=None, w=None):
        import os, sys
        _debug = os.environ.get('TP_DEBUG', '') == '1'
        self._refresh_dist_state()
        if not (self.use_tp and self.world_size > 1):
            return super().interval_propagate(*v, C=C, w=w)

        has_bias = len(v) >= 3
        if _debug:
            print(f"[TP rank={self.rank}] Row interval_propagate: has_bias={has_bias}, len(v)={len(v)}", flush=True, file=sys.stderr)

        # Compute IBP without bias so AllReduce doesn't overcount it.
        lower, upper = super().interval_propagate(
            *(v[:2] if has_bias else v), C=C, w=w)

        if isinstance(lower, torch.Tensor):
            if _debug:
                print(f"[TP rank={self.rank}] Row AllReduce lower shape={lower.shape}", flush=True, file=sys.stderr)
            lower = DifferentiableAllReduce.apply(lower.contiguous())
            if _debug:
                print(f"[TP rank={self.rank}] Row AllReduce lower DONE", flush=True, file=sys.stderr)
        if isinstance(upper, torch.Tensor):
            if _debug:
                print(f"[TP rank={self.rank}] Row AllReduce upper shape={upper.shape}", flush=True, file=sys.stderr)
            upper = DifferentiableAllReduce.apply(upper.contiguous())
            if _debug:
                print(f"[TP rank={self.rank}] Row AllReduce upper DONE", flush=True, file=sys.stderr)

        if has_bias:
            bias_lb, bias_ub = v[2][0], v[2][1]
            if self.beta_linear != 1.0:
                bias_lb = self.beta_linear * bias_lb
                bias_ub = self.beta_linear * bias_ub
            if C is not None:
                if isinstance(bias_lb, torch.Tensor):
                    bias_lb = C.matmul(bias_lb)
                if isinstance(bias_ub, torch.Tensor):
                    bias_ub = C.matmul(bias_ub)
            lower = lower + bias_lb
            upper = upper + bias_ub

        return lower, upper

    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       reduce_bias=True, **kwargs):
        self._refresh_dist_state()
        result = super().bound_backward(last_lA, last_uA, *x, start_node=start_node,
                                        reduce_bias=reduce_bias, **kwargs)
        if self.use_tp and self.world_size > 1:
            # Row's bias is "full" (same on all ranks because A is full here).
            # Divide by world_size so that AllReduce in backward_general gives
            # the correct total: sum(bias/ws) = bias.
            lbias = _scale_bias(result[1], self.world_size)
            ubias = _scale_bias(result[2], self.world_size)
            result = (result[0], lbias, ubias)
        return result
