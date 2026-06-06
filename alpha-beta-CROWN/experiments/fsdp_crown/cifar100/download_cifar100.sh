#!/bin/bash
# Download CIFAR-100 ResNet medium/large from VNN-COMP 2024.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ONNX_DIR="$SCRIPT_DIR/onnx"
VNNLIB_DIR="$SCRIPT_DIR/vnnlib"
mkdir -p "$ONNX_DIR" "$VNNLIB_DIR"

BASE_URL="https://raw.githubusercontent.com/ChristopherBrix/vnncomp2024_benchmarks/main/benchmarks/cifar100"

echo "Downloading CIFAR-100 ResNet ONNX..."
for model in CIFAR100_resnet_medium CIFAR100_resnet_large; do
    DEST="$ONNX_DIR/${model}.onnx"
    if [ -f "$DEST" ]; then
        echo "  $model.onnx already exists, skipping."
    else
        curl -fSL "$BASE_URL/onnx/${model}.onnx.gz" -o "$DEST.gz"
        gunzip -f "$DEST.gz"
        echo "  $model: $(du -h "$DEST" | cut -f1)"
    fi
done

INSTANCES_FILE="$SCRIPT_DIR/instances.csv"
if [ ! -f "$INSTANCES_FILE" ]; then
    curl -fSL "$BASE_URL/instances.csv" -o "$INSTANCES_FILE"
fi

echo "Downloading first 5 vnnlib specs (and the first matching one for each model)..."
seen_medium=0
seen_large=0
while IFS=',' read -r onnx vnnlib_path timeout; do
    spec_name="$(basename "$vnnlib_path")"
    spec_dest="$VNNLIB_DIR/$spec_name"
    if [ "$seen_medium" -ge 3 ] && [ "$seen_large" -ge 3 ]; then
        break
    fi
    if [[ "$onnx" == *medium* ]]; then
        seen_medium=$((seen_medium + 1))
    elif [[ "$onnx" == *large* ]]; then
        seen_large=$((seen_large + 1))
    fi
    if [ ! -f "$spec_dest" ]; then
        if curl -fSL "$BASE_URL/$vnnlib_path.gz" -o "$spec_dest.gz" 2>/dev/null; then
            gunzip -f "$spec_dest.gz"
        else
            curl -fSL "$BASE_URL/$vnnlib_path" -o "$spec_dest" || rm -f "$spec_dest"
        fi
    fi
done < "$INSTANCES_FILE"

echo ""
echo "Done."
ls -la "$ONNX_DIR"
echo "VNNLIB:"
ls -la "$VNNLIB_DIR" | head -10
