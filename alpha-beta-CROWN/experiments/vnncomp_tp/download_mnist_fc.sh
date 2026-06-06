#!/bin/bash
# Download MNIST-FC models from VNN-COMP 2022 benchmark.
# These are small fully-connected networks perfect for TP testing.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"

BASE_URL="https://raw.githubusercontent.com/VNN-COMP/vnncomp2022_benchmarks/main/benchmarks/mnist_fc"

echo "Downloading MNIST-FC models from VNN-COMP 2022..."

for model in mnist-net_256x2 mnist-net_256x4 mnist-net_256x6; do
    DEST="$MODELS_DIR/${model}.onnx"
    if [ -f "$DEST" ]; then
        echo "  $model.onnx already exists, skipping."
    else
        echo "  Downloading $model.onnx.gz ..."
        curl -fSL "$BASE_URL/onnx/${model}.onnx.gz" -o "$DEST.gz"
        gunzip -f "$DEST.gz"
        echo "  Done."
    fi
done

# Also download a vnnlib instance for reference
VNNLIB_DIR="$SCRIPT_DIR/vnnlib"
mkdir -p "$VNNLIB_DIR"

INSTANCES_URL="$BASE_URL/instances.csv"
INSTANCES_FILE="$VNNLIB_DIR/instances.csv"
if [ ! -f "$INSTANCES_FILE" ]; then
    echo "Downloading instances.csv..."
    curl -fSL "$INSTANCES_URL" -o "$INSTANCES_FILE"
fi

# Download vnnlib specs (gzipped on the new repo)
for spec in prop_0_0.03 prop_0_0.05; do
    SPEC_DEST="$VNNLIB_DIR/${spec}.vnnlib"
    if [ -f "$SPEC_DEST" ]; then
        echo "  $spec.vnnlib already exists, skipping."
    else
        echo "  Downloading $spec.vnnlib.gz ..."
        if curl -fSL "$BASE_URL/vnnlib/${spec}.vnnlib.gz" -o "$SPEC_DEST.gz"; then
            gunzip -f "$SPEC_DEST.gz"
        else
            echo "  (failed: $spec, not critical)"
            rm -f "$SPEC_DEST.gz"
        fi
    fi
done

echo ""
echo "Models downloaded to: $MODELS_DIR"
ls -la "$MODELS_DIR"
