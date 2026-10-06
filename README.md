# Scaling Neural Network Verification with Tensor Parallelism and Fully Sharded Data Parallelism

Code accompanying the paper

> **Scaling Neural Network Verification with Tensor Parallelism and Fully Sharded Data Parallelism**
> Sergei Vorobyov, Eugene Ilyushin — DAMDID/RCDL 2026.

This repository adapts two parameter-sharding strategies from large-scale model
training — **Tensor Parallelism (TP)** and **Fully Sharded Data Parallelism
(FSDP)** — to GPU bound-propagation verification, in order to scale formal
verification beyond single-GPU memory limits.

It is a **fork of [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
and [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)**
(winner of VNN-COMP 2021–2025). All upstream code is reused under its original
BSD 3-Clause license; see [License & Attribution](#license--attribution).

---

## What this fork adds

- **Tensor Parallelism for bound propagation.** New operators
  `BoundLinearTP_Col` / `BoundLinearTP_Row` shard both weight and CROWN
  `A`-matrices across GPUs (one `AllReduce` per Column–Row pair), with automatic
  graph sharding via `tp_shard_bounded_module`. Peak memory drops 1.97× at
  `P=2` and 3.83× at `P=4`, and propagation gets 1.90× and 3.49× faster. Bounds
  are exact for the first sharded zone; later zones fall back to IBP.
- **FSDP for bound propagation.** `fsdp_shard_bounded_module` shards only weight
  matrices (per-layer `AllGather`) and produces **bitwise-identical** bounds.
  Weights are a small share of the peak in bound propagation, so FSDP does not
  lower peak memory: on wide MLPs it raises it by 26–34%, and staging the
  weights from host memory does better.
- **Memory measurements** (`experiments/analysis/`): a per-node trace of a
  CROWN pass, a batch sweep for β-CROWN+BaB (peak ≈ 21.3 + 0.372·B MB, weights
  0.13%) and a CPU-offload baseline. The collated numbers are in
  `experiments/analysis/RESULTS.md`.
- **FSDP in complete verification** (β-CROWN + Branch-and-Bound), with a
  `force_synchronous` flag that makes single-GPU and FSDP runs do the same work.
- **Convolutional sharding** (`BoundConv`, sharded over output channels);
  checked on CIFAR-100 ResNets (VNN-COMP'24).
- **JIT fixes for Transformers** (`BoundReshape`, `BoundConcat`,
  `BoundConstantOfShape`) that let BaB run on a ViT (VNN-COMP'23) with sharded
  weights.

---

## Repository layout

```
alpha-beta-CROWN/                         # forked verifier (BSD-3-Clause)
├── auto_LiRPA/                           # forked bound-propagation library
│   └── auto_LiRPA/
│       ├── operators/tensor_parallel.py  # BoundLinearTP_Col/Row, DifferentiableAllReduce
│       ├── tp_utils.py                   # tp_shard_bounded_module, zone labelling
│       ├── fsdp_utils.py                 # fsdp_shard_bounded_module, gather/free hooks
│       ├── backward_bound.py             # CROWN backward (TP zones + FSDP hooks)
│       ├── interval_bound.py             # IBP forward (+ FSDP hooks)
│       └── bound_general.py              # BoundedModule (JIT trace, batch-dim propagation)
├── complete_verifier/                    # abcrown + Domain-Parallel BaB
│   ├── bab_parallel.py                   # scatter/gather of BaB domains
│   └── bab.py                            # DP integration + anti-deadlock
└── experiments/                          # experiments from the paper
    ├── tp_model.py                       # shared TP / dense model, copy-weights
    ├── analysis/                         # §4, §5.1, §5.4, §5.5: memory and time, RESULTS.md
    ├── alpha_crown/run.py                # §5.2: α-CROWN, single GPU vs. TP
    ├── vnncomp_tp/verify_tp.py           # §5.2: TP correctness & soundness
    ├── crown/run.py                      # TP vs. single-GPU memory at H=262144 (A40)
    └── fsdp_crown/
        ├── verify_fsdp.py                # §5.3: bitwise-identical bounds
        ├── run_abcrown_fsdp.py           # torchrun wrapper for abcrown+FSDP
        ├── mnist_fc_fair_{512,4096}.yaml # §5.5: same workload, single GPU vs. FSDP in BaB
        ├── vit/                          # §5.6: ViT (VNN-COMP'23)
        ├── cifar100/                     # §5.6: ResNet conv sharding (VNN-COMP'24)
        └── memory_experiment.py          # older harness, superseded by analysis/mem_time.py
```

---

## Installation

A CUDA GPU is required; multi-GPU experiments need 2 or 4 GPUs. Most
measurements in the paper used 4× NVIDIA RTX PRO 4000 Blackwell (24 GB); the
ViT and CIFAR-100 checks used 2× NVIDIA A40 (48 GB).

```bash
git clone https://github.com/vorobyov01/damdid-2026-submission.git
cd damdid-2026-submission

# uv.lock pins torch==2.8.x+cu128 (the build that matches a CUDA 12.8 driver).
# Always use `uv sync`, not a manual pip install.
uv sync
source .venv/bin/activate

# Extra dependencies for complete_verifier (not in uv.lock):
pip install git+https://github.com/Verified-Intelligence/onnx2pytorch@fe7281b9b6c8c28f61e72b8f3b0e3181067c7399
uv pip install onnx onnxruntime onnxoptimizer skl2onnx psutil appdirs packaging sortedcontainers timm pandas scipy -q
```

> **Gurobi** is optional: it is only needed for MIP/LP-based verification, which
> this work does not use. The BaB-only path runs without it (install a free stub
> for `gurobipy` if imports fail).

Smoke test (single GPU):

```bash
python alpha-beta-CROWN/auto_LiRPA/examples/simple/toy.py
```

On machines **without NVLink**, prefix distributed runs with
`NCCL_P2P_DISABLE=1`.

---

## Reproducing the experiments

Section numbers refer to the paper. Paths below assume the repository root.
Run each mode in its own process: bound tensors returned under FSDP keep an
autograd graph alive, and in a shared process that memory would be counted in
the next measurement.

### Memory and time of FSDP — §4, §5.4

```bash
cd alpha-beta-CROWN/experiments/analysis

# single GPU, CPU offload and FSDP=2/4 at h=4096, d=4 (repeat for other h, d)
python mem_time.py --mode single  --h 4096 --d 4
python mem_time.py --mode offload --h 4096 --d 4
torchrun --nproc_per_node=2 mem_time.py --mode fsdp --h 4096 --d 4
torchrun --nproc_per_node=4 mem_time.py --mode fsdp --h 4096 --d 4

# per-node memory trace of one CROWN pass (§4)
python mem_time.py --mode single --h 4096 --d 4 --trace
```

### Tensor Parallelism — §5.1, §5.2

```bash
cd alpha-beta-CROWN/experiments

# memory and time scaling, graph construction timed separately (§5.1);
# the single-GPU and TP runs use different random weights, so compare
# memory and time only
python analysis/tp_scaling.py --mode single --hidden-dim 131072
torchrun --nproc_per_node=2 analysis/tp_scaling.py --mode tp --hidden-dim 131072
torchrun --nproc_per_node=4 analysis/tp_scaling.py --mode tp --hidden-dim 131072

# α-CROWN: bounds must match the single-GPU reference (§5.2)
python alpha_crown/run.py --mode single --method alpha-CROWN --save ref.pt
torchrun --nproc_per_node=2 alpha_crown/run.py --mode tp --method alpha-CROWN --compare ref.pt

# TP correctness & soundness on VNN-COMP MNIST-FC ONNX models (§5.2)
bash vnncomp_tp/download_mnist_fc.sh
torchrun --nproc_per_node=2 vnncomp_tp/verify_tp.py
```

### FSDP bounds — §5.3

```bash
cd alpha-beta-CROWN/experiments/fsdp_crown

# bounds bitwise-identical to single-GPU (IBP + CROWN)
torchrun --nproc_per_node=2 verify_fsdp.py
```

### Complete verification and other architectures — §5.5, §5.6

```bash
cd alpha-beta-CROWN/complete_verifier

# §5.5: same workload on one GPU and under FSDP=2
CUDA_VISIBLE_DEVICES=0 python ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/mnist_fc_fair_512.yaml
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
  ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/mnist_fc_fair_512.yaml

# §5.5: memory by tensor class for one batch size
python ../experiments/analysis/alpha_probe.py \
  --config ../experiments/fsdp_crown/mnist_fc_fair_512.yaml --batch_size 4096

# §5.6: ViT (VNN-COMP'23)
bash ../experiments/fsdp_crown/vit/download_vit.sh
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
  ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/vit/vit_pgd_fair.yaml

# §5.6: CIFAR-100 ResNet conv sharding (VNN-COMP'24)
bash ../experiments/fsdp_crown/cifar100/download_cifar100.sh
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
  ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/cifar100/cifar100_large_fair.yaml
```

The same-workload configs fix the batch size, disable
`auto_enlarge_batch_size` / `early_stop` / `pruning_in_iteration`, and cap BaB
rounds, so single-GPU and FSDP=2 do the same work and peak memory is directly
comparable.

---

## License & Attribution

This project is a derivative work of α,β-CROWN and auto_LiRPA, both released by
the α,β-CROWN Team under the **BSD 3-Clause License**. In accordance with that
license:

- The original copyright notices, license texts (`alpha-beta-CROWN/LICENSE`,
  `alpha-beta-CROWN/complete_verifier/LICENSE`,
  `alpha-beta-CROWN/auto_LiRPA/LICENSE`), the `CONTRIBUTORS` file, and the
  per-file source headers are **retained unchanged**.
- The name of the α,β-CROWN Team and its contributors is **not** used to endorse
  or promote this fork.

Modifications in this repository (TP/FSDP bound propagation, convolutional
sharding, Transformer JIT fixes, Domain-Parallel BaB, and the `experiments/`
folder) are:

```
Portions Copyright (C) 2026 Sergei Vorobyov, Eugene Ilyushin
```

and are likewise distributed under the BSD 3-Clause License.

---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{vorobyov2026scaling,
  title     = {Scaling Neural Network Verification with Tensor Parallelism
               and Fully Sharded Data Parallelism},
  author    = {Vorobyov, Sergei and Ilyushin, Eugene},
  booktitle = {Data Analytics and Management in Data Intensive Domains
               (DAMDID/RCDL 2026)},
  year      = {2026}
}
```

and the upstream α,β-CROWN / auto_LiRPA papers (see the
[upstream README](https://github.com/Verified-Intelligence/alpha-beta-CROWN#citation)),
in particular CROWN (Zhang et al., 2018), auto_LiRPA (Xu et al., 2020),
α-CROWN (Xu et al., 2021), and β-CROWN (Wang et al., 2021).
