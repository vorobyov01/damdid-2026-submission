# DAMDID revision: measured results (4x RTX PRO 4000 Blackwell 24GB, torch 2.8.0+cu128, NCCL, PCIe)

All runs: seed 42, eps=0.02, input 784, batch 1, CROWN, 1 warm-up + 3 repeats,
one mode per process unless stated.  std over repeats was 0.00 MB for every
memory figure.

## 1. The published 34-39% FSDP peak saving is a measurement artefact

memory_experiment.py measures both modes in one process, FSDP first, and returns
the bound tensors to the caller.  For FSDP those tensors carry an autograd graph
that keeps ~1.9 GB of intermediates alive, so the *second* measurement counts
them as its own peak.

| protocol (h=4096, d=4)                     | single  | FSDP=2 | "saving" |
|--------------------------------------------|---------|--------|----------|
| original script (reproduces the paper)      | 3799.6  | 2526.9 | +33.5%   |
| one mode per process                        | 1944.9  | 2526.9 | -29.9%   |
| same process, bounds released               | 1944.9  | 2526.9 | -29.9%   |
| same process, bounds retained, single 2nd   | 4208.6  | 2526.9 | +40.0%   |
| same process, bounds retained, FSDP 2nd     | 1944.9  | 4158.7 | -113.8%  |

transient (peak - resident) is invariant: 1527.7 MB single, 2109.7 MB FSDP.

## 2. Corrected FSDP measurements, before the fix

| config          | single peak | FSDP=2 | FSDP=4 | single wall | FSDP=2 wall (comm) |
|-----------------|-------------|--------|--------|-------------|--------------------|
| h=1024, d=4     |   149.6     |  205.2 |  201.7 |   23.6 ms   |   31.2 ms (15.8)   |
| h=4096, d=4     |  1944.9     | 2526.9 | 2526.8 |   74.7 ms   |  254.7 ms (182.3)  |
| h=8192, d=4     |  7593.9     | 9781.7 | 9781.4 |  418.1 ms   |  999.3 ms (590.4)  |
| h=4096, d=8     |  7934.0     |10232.0 |10231.8 |  381.1 ms   |  881.5 ms (516.9)  |

Paper's single-GPU figures were 3800 / 14759 / 16726 -> inflated 1.94-2.11x.
Paper's FSDP figures reproduce exactly (2527 / 9782 / 10232).
P=4 never improves peak over P=2.

## 3. Baseline "exactly 1/P" is bookkeeping, not memory

BoundParams bytes 204.5 -> 102.3 MB (exactly 1/2), but really allocated memory
417.2 -> 417.2 MB (ratio 1.000): the AllGathered full weights were never freed.

## 4. Root cause and fix

Leak inventory after one CROWN pass (h=4096, d=4, FSDP=2, bounds deleted):
  18 x [4096,1,4096]  = 1152 MB   CROWN A-matrices (legitimate)
   9 x [4096,4096]    =  576 MB   leaked full weights
   3 x [2048,4096]    =   96 MB   the shards themselves

Two defects in auto_LiRPA/fsdp_utils.py:
1. fsdp_gather_node built the full weight as [empty_like x P] + torch.cat:
   two full-weight allocations per gather, and cat is recorded by autograd.
   Fix: all_gather_into_tensor into one buffer under no_grad.
2. fsdp_free_node released lower/upper with delattr, but Bound.lower/.upper are
   @property over _lower/_upper (operators/base.py:201), so delattr always
   raised AttributeError and the except-clause swallowed it.
   Fix: node.delete_lower_and_upper_bounds() (already provided by the base class).

Effect of fix 1: h=4096 d=4 peak 2526.9 -> 2476.0; h=8192 9781.7 -> 9551.8;
h=4096 d=8 10232.0 -> 10181.1.  Fix 2 cleans the node attributes (the
BoundParams inventory after a pass now holds shards only) but does not move the
peak: at the peak instant the retained copies are held by auto_LiRPA's own
bookkeeping (9 live [4096,4096] tensors), not by the BoundParams nodes.
Bitwise identity preserved after both fixes: ALL PASS at P=2 and at P=4.

Corrected FSDP peaks after both fixes: 2476.0 (h=4096 d=4), 9551.8 (h=8192 d=4),
10181.1 (h=4096 d=8), 200.6 (h=1024 d=4) -- still above the single-GPU peak in
every configuration, and identical at P=2 and P=4.

## 5. Peak is dominated by A-matrices, not weights (CROWN, incomplete)

Backward trace, h=4096 d=4 single: allocated grows 417 -> 1844 MB in 128 MB
steps, every step an A-matrix pair of shape [4096,1,4096]; weights are 204 MB of
the 1945 MB peak.

## 6. TP: near-linear scaling of both peak memory and propagation time

H=131072, D=4096, N=2048, eps=0.01, CROWN, graph construction timed separately
from bound propagation (the construction cost is one-off; the earlier combined
figure hid this):

| mode   | peak (MB) | reduction | build (ms) | propagation (ms) | speed-up | comm (ms) |
|--------|-----------|-----------|------------|------------------|----------|-----------|
| single |  13513.2  |  1.00x    |   4279.5   |      546.6       |  1.00x   |     0     |
| TP=2   |   6856.7  |  1.97x    |  22679.8   |      287.7       |  1.90x   |    89.2   |
| TP=4   |   3528.4  |  3.83x    |  11392.1   |      156.5       |  3.49x   |   139.9   |

single at H=131072 equals the paper's TP=2 rank figure at H=262144 (13513 MB):
the TP memory model is exact.  Collectives are 7 calls, <1% of propagation time.
Graph construction under TP is 5.3x slower than dense at P=2 (JIT tracing of the
custom Col/Row operators) but is paid once per model.

CAVEAT, and it must go into the paper: on this very model the TP bounds are
meaningless -- lb_sum = -3.35e8 versus -6.65e3 for single GPU -- because the
ReLU sits inside the sharded zone and its intermediate bounds fall back to IBP.
The 2x memory reduction of the paper's Experiment 1 was reported on a model
whose bounds are useless.  TP is only usable when the sharded zone contains no
intermediate activation.

## 6b. alpha-CROWN under TP is sound and matches to machine precision

experiments/alpha_crown/run.py, single-GPU reference vs TP=2:
lb max_abs_diff = 4.47e-08, mean 2.14e-08, max_rel_diff = 3.64e-07, PASS at
tolerance 1e-4; the TP lower bound came out 0.000804 tighter.  This closes
reviewer 1's remark that TP soundness was demonstrated for CROWN only.

## 7. BaB memory is linear in the number of domains; weights are noise

mnist-net_256x6, eps=0.05, force_synchronous, 10 BaB rounds, single GPU:

| batch | alpha | beta | interm bounds | lA/uA | weights | process peak |
|-------|-------|------|---------------|-------|---------|--------------|
|   128 |  3.79 | 0.01 |  9.06         |  9.08 | 2.03    |    68.73     |
|   256 |  7.58 | 0.02 | 18.13         | 18.16 | 2.03    |   116.24     |
|   512 | 15.16 | 0.05 | 36.26         | 36.33 | 2.03    |   211.93     |

peak(B) ~ 21.0 + 0.373*B MB (paper's Table 5: 211.7 @512, 1537.3 @4096 -> same law).
Every batch-scaled term doubles with B; weights stay 2.03 MB.
Hence splitting the *domain* axis over P GPUs gives per-GPU peak 21 + 0.373*B/P:
45% saving at P=2 and 68% at P=4 for B=512, and the projection is validated by
the measured single-GPU point at B/P (116.24 measured vs 116.5 predicted).

## 8. BaB under FSDP, same workload (mnist-net_256x6, batch 512)

single 211.93 MB, FSDP=2 214.88 MB, FSDP=4 214.87 MB per rank: sharding 2.03 MB
of weights cannot pay for the AllGather buffers, and P=4 changes nothing.
Breakdown at batch 4096: alpha 121.25 MB (17.2%), interm bounds and lA/uA ~40%
each, weights 2.03 MB (0.13%).  The paper's claim that alpha tensors occupy
~1.5 GB conflates alphas with the whole peak: 1.5 GB is the peak, alphas are
121 MB of it.  The correct statement is that every batch-scaled term (alphas,
intermediate bounds, A-matrices) grows linearly in the number of domains while
weights stay constant.

## 9. Naive baseline: CPU offloading beats FSDP on both axes

Weights pinned in host memory, staged to the device per layer through the same
hook sites (host-to-device copy instead of the collective).  GPU-resident weight
bytes: 0, against S_W/P for FSDP.  Bounds identical to the single-GPU reference.

| config      | single peak | FSDP=2 peak | offload peak | single ms | FSDP=2 ms | offload ms |
|-------------|-------------|-------------|--------------|-----------|-----------|------------|
| h=1024 d=4  |    149.6    |    200.6    |    184.6     |   23.6    |   30.1    |    21.5    |
| h=4096 d=4  |  1 944.9    |  2 476.0    |  2 271.6     |   74.7    |  236.6    |   103.3    |
| h=8192 d=4  |  7 593.9    |  9 551.8    |  8 759.0     |  418.1    |  983.3    |   528.4    |
| h=4096 d=8  |  7 934.0    | 10 181.1    |  9 720.7     |  381.1    |  867.4    |   477.6    |

H2D traffic: 14 transfers per pass (44 at d=8), 51 / 637 / 2427 / 2350 MB.
FSDP traffic: 19 collectives (53 at d=8), 66 / 842 / 3219 / 2811 MB.
Both stay above the single-GPU peak because auto_LiRPA retains the staged copies
past the layer that needs them: 531 MB at h=4096 d=4, more than twice the
204.5 MB of weights either mechanism removes.

Activation checkpointing does not apply: the CROWN backward stores no forward
activations to recompute; its A-matrices are live because they are still needed.

## 10. Consistency of the ranges quoted in the paper

peak increase FSDP=2 vs single: +34.1% (h=1024), +27.3% (h=4096 d=4),
+25.8% (h=8192), +28.3% (h=4096 d=8)  -> "26-34%".
propagation slowdown: 1.28x (h=1024), 3.17x, 2.35x, 2.28x -> "up to 3.2x",
"2.3-3.2x for h >= 4096".
collective share of wall-clock: 57.5%, 70.5%, 59.3%, 58.9% -> "57-70%".

## 11. Where the alpha-CROWN deviation actually comes from

A distributed run of abcrown does not reproduce a plain single-GPU run bit for
bit in the optimised alpha/beta path: 3 of 9 spec bounds differ, max |d| =
6.10e-05, max rel = 1.17e-07.  It is NOT the sharding.  run_abcrown_fsdp.py
disables the TorchScript fusers when world_size > 1 (a workaround for a strides
assertion on non-primary GPUs).  Applying exactly that fuser configuration on a
single GPU, with no process group at all, reproduces the distributed values
digit for digit:

  fusers disabled, 1 GPU : [-467.7776184082031, -545.41748046875, ...]
  distributed (P=2)      : [-467.7776184082031, -545.41748046875, ...]
  plain single GPU       : [-467.777587890625,  -545.4174194335938, ...]

Verified with the published code and with both fixes applied: identical
deviation, so the fixes are not implicated.

## 12. Domain-parallel BaB: how far the implementation got

Scattering the domains BEFORE build_history_and_set_bounds (the earlier attempt
sliced them after, when the model state was already sized for the full batch)
makes per-rank propagation correct: with bab.domain_parallel enabled, each rank
builds its own state and completes a BaB round on its own chunk with sensible
bounds.  Two further findings:

1. Domain-parallel does not compose with FSDP.  FSDP's per-layer AllGather is
   interleaved with data-dependent control flow, so once ranks hold different
   subdomains their collective sequences diverge and both GPUs spin at 100%
   utilisation with 512 MiB allocated -- an NCCL deadlock.  FSDP_DISABLE=1 (new
   flag) removes it, and sharding 2.03 MB of weights was pointless anyway.
2. What still stalls is the gather of per-rank results and domain dicts after
   propagation.  Round 1 propagates correctly and then hangs in the gather.

The code is published behind bab.domain_parallel (off by default) with that
limitation documented in the source.  The paper therefore reports the projection
from the batch sweep, which needs no working scatter to be valid.
