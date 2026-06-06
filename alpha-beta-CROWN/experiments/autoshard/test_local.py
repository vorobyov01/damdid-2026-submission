"""
Local (CPU, no distributed) test of tp_shard_bounded_module logic.

Verifies that:
1. _topological_order works correctly
2. BoundLinear nodes are found and alternated Col/Row
3. Weight sharding produces correct shapes
4. Single-GPU bounds match (world_size=1 means no sharding)

Run: python test_local.py   (from this directory)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../auto_LiRPA"))

import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators.linear import BoundLinear
from auto_LiRPA.operators.tensor_parallel import BoundLinearTP_Col, BoundLinearTP_Row
from auto_LiRPA.operators.leaf import BoundParams
from auto_LiRPA.tp_utils import _topological_order, tp_shard_bounded_module


class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def test_topological_order():
    print("Test: topological order...")
    model = TwoLayerMLP()
    x = torch.randn(2, 4)
    lirpa = BoundedModule(model, x)
    topo = _topological_order(lirpa)

    names = [n.name for n in topo]
    print(f"  Order: {names}")
    assert len(topo) == len(list(lirpa.nodes())), "Topo order missed some nodes"

    linears = [n for n in topo if type(n) is BoundLinear]
    print(f"  BoundLinear nodes: {[n.name for n in linears]}")
    assert len(linears) == 2, f"Expected 2 BoundLinear, got {len(linears)}"
    print("  PASSED")


def test_world_size_1_noop():
    print("\nTest: world_size=1 is no-op...")
    model = TwoLayerMLP()
    x = torch.randn(2, 4)
    lirpa = BoundedModule(model, x)

    result = tp_shard_bounded_module(lirpa, world_size=1, rank=0)
    assert result == [], f"Expected empty list for ws=1, got {result}"

    linears = [n for n in lirpa.nodes() if isinstance(n, BoundLinear)]
    for n in linears:
        assert type(n) is BoundLinear, f"Node {n.name} should still be BoundLinear"
    print("  PASSED")


def test_shard_shapes_2layer():
    print("\nTest: shard shapes (2-layer, ws=2)...")
    model = TwoLayerMLP()
    x = torch.randn(2, 4)
    lirpa = BoundedModule(model, x)

    topo = _topological_order(lirpa)
    linears = [n for n in topo if type(n) is BoundLinear]

    orig_shapes = {}
    for n in linears:
        w = n.inputs[1]
        assert isinstance(w, BoundParams)
        orig_shapes[n.name] = w.param.shape
        print(f"  Before shard: {n.name} weight shape = {w.param.shape}")

    sharded = tp_shard_bounded_module(lirpa, world_size=2, rank=0)
    assert len(sharded) == 2

    for n_name in sharded:
        n = lirpa[n_name]
        w = n.inputs[1]
        print(f"  After shard: {n.name} type={type(n).__name__} "
              f"weight shape = {w.param.shape}")

    fc1 = lirpa[sharded[0]]
    fc2 = lirpa[sharded[1]]

    assert isinstance(fc1, BoundLinearTP_Col), f"First should be Col, got {type(fc1).__name__}"
    assert isinstance(fc2, BoundLinearTP_Row), f"Second should be Row, got {type(fc2).__name__}"

    w1 = fc1.inputs[1].param
    assert w1.shape[0] == 4, f"Col shard: expected out=4, got {w1.shape[0]}"
    assert w1.shape[1] == 4, f"Col shard: expected in=4, got {w1.shape[1]}"

    b1 = fc1.inputs[2].param
    assert b1.shape[0] == 4, f"Col shard: expected bias=4, got {b1.shape[0]}"

    w2 = fc2.inputs[1].param
    assert w2.shape[0] == 2, f"Row shard: expected out=2, got {w2.shape[0]}"
    assert w2.shape[1] == 4, f"Row shard: expected in=4, got {w2.shape[1]}"

    if len(fc2.inputs) > 2:
        b2 = fc2.inputs[2].param
        assert b2.shape[0] == 2, f"Row shard: bias should be full (2), got {b2.shape[0]}"

    print("  PASSED")


def test_shard_shapes_3layer():
    print("\nTest: shard shapes (3-layer, ws=2)...")
    model = ThreeLayerMLP()
    x = torch.randn(2, 4)
    lirpa = BoundedModule(model, x)

    topo = _topological_order(lirpa)
    linears = [n for n in topo if type(n) is BoundLinear]
    assert len(linears) == 3, f"Expected 3 linears, got {len(linears)}"

    sharded = tp_shard_bounded_module(lirpa, world_size=2, rank=0)
    # Only 2 of 3 are sharded (last one left un-sharded for output correctness)
    assert len(sharded) == 2, f"Expected 2 sharded, got {len(sharded)}"

    types = [type(lirpa[name]).__name__ for name in sharded]
    print(f"  Sharded types: {types}")
    assert types == ["BoundLinearTP_Col", "BoundLinearTP_Row"], \
        f"Expected Col/Row, got {types}"

    # Verify the last linear is still regular BoundLinear
    last_linear = linears[-1]
    assert type(last_linear) is BoundLinear, \
        f"Last linear should be un-sharded BoundLinear, got {type(last_linear).__name__}"
    print(f"  Last linear ({last_linear.name}) is un-sharded: OK")
    print("  PASSED")


def test_bounds_unchanged_ws1():
    print("\nTest: bounds unchanged with world_size=1 (no-op)...")
    model = TwoLayerMLP()
    x = torch.randn(2, 4)
    lower, upper = x - 0.1, x + 0.1
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)

    lirpa1 = BoundedModule(model, torch.empty_like(x))
    bx1 = BoundedTensor(x, ptb)
    lb1, ub1 = lirpa1.compute_bounds(x=(bx1,), method="CROWN")

    lirpa2 = BoundedModule(model, torch.empty_like(x))
    tp_shard_bounded_module(lirpa2, world_size=1, rank=0)
    bx2 = BoundedTensor(x, ptb)
    lb2, ub2 = lirpa2.compute_bounds(x=(bx2,), method="CROWN")

    diff_lb = (lb1 - lb2).abs().max().item()
    diff_ub = (ub1 - ub2).abs().max().item()
    print(f"  lb diff = {diff_lb:.2e}, ub diff = {diff_ub:.2e}")
    assert diff_lb < 1e-6 and diff_ub < 1e-6, "Bounds changed with ws=1!"
    print("  PASSED")


if __name__ == "__main__":
    test_topological_order()
    test_world_size_1_noop()
    test_shard_shapes_2layer()
    test_shard_shapes_3layer()
    test_bounds_unchanged_ws1()
    print("\n=== ALL LOCAL TESTS PASSED ===")
