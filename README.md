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
  graph sharding via `tp_shard_bounded_module`. Gives ≈2× peak-memory reduction
  at `P=2`; sound on VNN-COMP MNIST-FC.
- **FSDP for bound propagation.** `fsdp_shard_bounded_module` shards only weight
  matrices (per-layer `AllGather`), producing **bitwise-identical** bounds.
  Baseline weight storage drops by exactly `1/P` (50% at `P=2`); peak memory by
  34–39% on wide MLPs.
- **FSDP in complete verification** (β-CROWN + Branch-and-Bound), with a
  `force_synchronous` flag for fair single-GPU vs. FSDP comparison.
- **Convolutional sharding** (`BoundConv`, sharded over output channels);
  validated on CIFAR-100 ResNets (VNN-COMP'24).
- **JIT fixes for Transformers** (`BoundReshape`, `BoundConcat`,
  `BoundConstantOfShape`) enabling FSDP verification of a ViT (VNN-COMP'23).

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
    ├── crown/run.py                      # Exp. 1: TP OOM vs. single-GPU memory
    ├── alpha_crown/run.py                # α-CROWN numeric single vs. TP comparison
    ├── vnncomp_tp/verify_tp.py           # Exp. 2–3: TP correctness & soundness
    └── fsdp_crown/                       # Exp. 4–8: FSDP
        ├── verify_fsdp.py                # Exp. 4: bitwise-identical bounds
        ├── memory_experiment.py          # Exp. 5: baseline / peak memory
        ├── run_abcrown_fsdp.py           # torchrun wrapper for abcrown+FSDP
        ├── mnist_fc_fair_{512,4096}.yaml # Exp. 6: fair FSDP vs. single in BaB
        ├── vit/                          # Exp. 7: ViT (VNN-COMP'23)
        └── cifar100/                     # Exp. 8: ResNet conv sharding (VNN-COMP'24)
```

---

## Installation

A CUDA GPU is required; multi-GPU experiments need ≥2 GPUs. The runs in the
paper used 2× NVIDIA A40 (48 GB).

```bash
git clone https://github.com/vorobyov01/damdid-2026-submission.git
cd damdid-2026-submission

# uv.lock pins torch==2.8.x+cu128 (the only build compatible with the CUDA 12.8
# driver on NVIDIA A40). Always use `uv sync`, not a manual pip install.
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

On pods **without NVLink**, prefix distributed runs with `NCCL_P2P_DISABLE=1`.

---

## Reproducing the experiments

All multi-GPU runs use `torchrun --nproc_per_node=2`. Paths below assume the
repository root.

### TP — Experiments 1–3

```bash
cd alpha-beta-CROWN/experiments

# Exp. 1: model that OOMs on one GPU, fits under TP=2 (peak-memory reduction)
python crown/run.py        --mode single --input-dim 4096 --hidden-dim 262144 --batch-size 2048
torchrun --nproc_per_node=2 crown/run.py --mode tp --input-dim 4096 --hidden-dim 262144 --batch-size 2048

# α-CROWN numeric comparison (bounds must match the single-GPU reference)
python alpha_crown/run.py --mode single --method alpha-CROWN --save ref.pt
torchrun --nproc_per_node=2 alpha_crown/run.py --mode tp --method alpha-CROWN --compare ref.pt

# Exp. 2–3: TP correctness & soundness on VNN-COMP MNIST-FC ONNX models
bash vnncomp_tp/download_mnist_fc.sh
torchrun --nproc_per_node=2 vnncomp_tp/verify_tp.py
```

### FSDP — Experiments 4–5

```bash
cd alpha-beta-CROWN/experiments/fsdp_crown

# Exp. 4: bounds bitwise-identical to single-GPU (IBP + CROWN)
torchrun --nproc_per_node=2 verify_fsdp.py

# Exp. 5: baseline (exact 1/P) and peak (34–39%) memory savings
torchrun --nproc_per_node=2 memory_experiment.py   # writes fsdp_memory_results.json
```

### FSDP + complete verification — Experiments 6–8

```bash
cd alpha-beta-CROWN/complete_verifier

# Exp. 6: fair single-GPU vs. FSDP=2 in β-CROWN+BaB (identical workload)
CUDA_VISIBLE_DEVICES=0 python ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/mnist_fc_fair_512.yaml
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
  ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/mnist_fc_fair_512.yaml

# Exp. 7: ViT (VNN-COMP'23) — bitwise-identical bounds under FSDP
bash ../experiments/fsdp_crown/vit/download_vit.sh
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
  ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/vit/vit_pgd_fair.yaml

# Exp. 8: CIFAR-100 ResNet conv sharding (VNN-COMP'24)
bash ../experiments/fsdp_crown/cifar100/download_cifar100.sh
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
  ../experiments/fsdp_crown/run_abcrown_fsdp.py \
  --config ../experiments/fsdp_crown/cifar100/cifar100_large_fair.yaml
```

The fair-comparison configs fix the batch size, disable
`auto_enlarge_batch_size` / `early_stop` / `pruning_in_iteration`, and cap BaB
rounds, so single-GPU and FSDP=2 perform an identical workload and peak memory
is directly comparable.

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
