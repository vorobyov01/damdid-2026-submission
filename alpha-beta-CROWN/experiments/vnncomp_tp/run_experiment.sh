#!/bin/bash
# Full VNN-COMP TP experiment: download models, run single-GPU reference,
# then TP=2 and compare bounds.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EPS=0.026
BATCH=1

echo "=============================================="
echo "  VNN-COMP Tensor Parallelism Experiment"
echo "=============================================="

# Step 1: Download models
echo ""
echo "[Step 1] Downloading MNIST-FC models..."
bash download_mnist_fc.sh

# Step 2: Kill stale processes
echo ""
echo "[Step 2] Cleaning up stale processes..."
killall -9 python3 torchrun 2>/dev/null || true
sleep 1

MODELS=(
    "models/mnist-net_256x2.onnx"
    "models/mnist-net_256x4.onnx"
    "models/mnist-net_256x6.onnx"
)
MODEL_NAMES=("256x2" "256x4" "256x6")
ALL_PASS=true

for i in "${!MODELS[@]}"; do
    ONNX="${MODELS[$i]}"
    NAME="${MODEL_NAMES[$i]}"

    if [ ! -f "$ONNX" ]; then
        echo "  [SKIP] $ONNX not found"
        continue
    fi

    REF="/tmp/vnncomp_ref_${NAME}.pt"

    echo ""
    echo "=============================================="
    echo "  Model: $NAME ($ONNX)"
    echo "=============================================="

    # Step 3: Single GPU reference
    echo ""
    echo "[Step 3] Single GPU reference ($NAME)..."
    python3 run.py --mode single --onnx "$ONNX" \
        --method alpha-CROWN --eps "$EPS" --batch-size "$BATCH" --save "$REF"

    # Step 4: TP=2
    echo ""
    echo "[Step 4] TP=2 ($NAME)..."
    torchrun --nproc_per_node=2 --master_port=29520 run.py --mode tp \
        --onnx "$ONNX" --method alpha-CROWN --eps "$EPS" --batch-size "$BATCH" \
        --compare "$REF"

    echo ""
    echo "  ---"
done

# Cleanup
for NAME in "${MODEL_NAMES[@]}"; do
    rm -f "/tmp/vnncomp_ref_${NAME}.pt"
done

echo ""
echo "=============================================="
echo "  Experiment complete."
echo "=============================================="
