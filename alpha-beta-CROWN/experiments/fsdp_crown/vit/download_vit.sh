#!/bin/bash
# Download Vision Transformer benchmark from VNN-COMP 2023.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ONNX_DIR="$SCRIPT_DIR/onnx"
VNNLIB_DIR="$SCRIPT_DIR/vnnlib"
mkdir -p "$ONNX_DIR" "$VNNLIB_DIR"

BASE_URL="https://raw.githubusercontent.com/ChristopherBrix/vnncomp2023_benchmarks/main/benchmarks/vit"

echo "Downloading ViT models from VNN-COMP 2023..."

for model in pgd_2_3_16 ibp_3_3_8; do
    DEST="$ONNX_DIR/${model}.onnx"
    if [ -f "$DEST" ]; then
        echo "  $model.onnx already exists, skipping."
    else
        echo "  Downloading $model.onnx.gz ..."
        curl -fSL "$BASE_URL/onnx/${model}.onnx.gz" -o "$DEST.gz"
        gunzip -f "$DEST.gz"
    fi
done

# Download a few vnnlib specs from instances.csv (first 3)
INSTANCES_FILE="$SCRIPT_DIR/instances.csv"
if [ ! -f "$INSTANCES_FILE" ]; then
    curl -fSL "$BASE_URL/instances.csv" -o "$INSTANCES_FILE"
fi

# Each line: onnx/X.onnx,vnnlib/Y.vnnlib,timeout
while IFS=',' read -r onnx vnnlib_path timeout; do
    spec_name="$(basename "$vnnlib_path")"
    spec_dest="$VNNLIB_DIR/$spec_name"
    if [ ! -f "$spec_dest" ]; then
        if curl -fSL "$BASE_URL/$vnnlib_path.gz" -o "$spec_dest.gz"; then
            gunzip -f "$spec_dest.gz"
        else
            curl -fSL "$BASE_URL/$vnnlib_path" -o "$spec_dest" || rm -f "$spec_dest"
        fi
    fi
done < <(head -5 "$INSTANCES_FILE")

echo ""
echo "Done. ONNX:"
ls -la "$ONNX_DIR"
echo "VNNLIB (first 5):"
ls -la "$VNNLIB_DIR" | head -8
