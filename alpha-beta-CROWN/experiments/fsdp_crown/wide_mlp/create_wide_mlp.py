"""Create a wide MLP whose linear weights dominate GPU memory and
export it together with a single VNN-LIB robustness spec.

Run:
    python create_wide_mlp.py --hidden 4096 --depth 4 --eps 0.01

Produces:
    onnx/wide_mlp_<h>x<d>.onnx
    vnnlib/wide_mlp_<h>x<d>.vnnlib
"""
import argparse
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torchvision


def build_mlp(input_dim: int, hidden: int, depth: int, num_classes: int) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
        in_dim = hidden
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)


def get_mnist_image(idx: int = 0):
    ds = torchvision.datasets.MNIST(
        root="/tmp/mnist", train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    img, label = ds[idx]
    return img.view(-1).numpy().astype(np.float32), int(label)


def write_vnnlib(path: str, x0: np.ndarray, label: int, eps: float, num_classes: int):
    n = x0.shape[0]
    lo = np.clip(x0 - eps, 0.0, 1.0)
    hi = np.clip(x0 + eps, 0.0, 1.0)
    with open(path, "w") as f:
        for i in range(n):
            f.write(f"(declare-const X_{i} Real)\n")
        for j in range(num_classes):
            f.write(f"(declare-const Y_{j} Real)\n")
        f.write("\n; Input box\n")
        for i in range(n):
            f.write(f"(assert (>= X_{i} {lo[i]:.6f}))\n")
            f.write(f"(assert (<= X_{i} {hi[i]:.6f}))\n")
        f.write("\n; Output: at least one wrong-class logit must dominate the true class\n")
        f.write("(assert (or\n")
        for j in range(num_classes):
            if j == label:
                continue
            f.write(f"  (and (>= Y_{j} Y_{label}))\n")
        f.write("))\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input-idx", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_dim = 28 * 28
    num_classes = 10
    model = build_mlp(input_dim, args.hidden, args.depth, num_classes)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MLP {args.hidden}x{args.depth}, params = {n_params / 1e6:.2f} M, "
          f"weights = {n_params * 4 / 2**20:.1f} MB (fp32)")

    onnx_dir = os.path.join(args.out_dir, "onnx")
    vnnlib_dir = os.path.join(args.out_dir, "vnnlib")
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(vnnlib_dir, exist_ok=True)

    name = f"wide_mlp_{args.hidden}x{args.depth}"
    onnx_path = os.path.join(onnx_dir, f"{name}.onnx")
    vnnlib_path = os.path.join(vnnlib_dir, f"{name}.vnnlib")

    dummy = torch.zeros(1, input_dim)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["X"], output_names=["Y"],
        dynamic_axes={"X": {0: "batch"}, "Y": {0: "batch"}},
        opset_version=17,
    )
    print(f"Wrote {onnx_path} ({os.path.getsize(onnx_path) / 2**20:.1f} MB)")

    x0, label = get_mnist_image(args.input_idx)
    write_vnnlib(vnnlib_path, x0, label, args.eps, num_classes)
    print(f"Wrote {vnnlib_path} (label={label}, eps={args.eps})")


if __name__ == "__main__":
    main()
