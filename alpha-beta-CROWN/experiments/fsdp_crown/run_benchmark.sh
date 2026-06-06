#!/bin/bash
# Run mnist_fc verification: single GPU vs FSDP=2
# Usage: bash run_benchmark.sh [incomplete|complete]
set -e

MODE="${1:-incomplete}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFIER_DIR="$SCRIPT_DIR/../../complete_verifier"

if [ "$MODE" = "complete" ]; then
    CONFIG="$SCRIPT_DIR/mnist_fc_complete.yaml"
    echo "=== Complete verification (BaB) ==="
else
    CONFIG="$SCRIPT_DIR/mnist_256x6.yaml"
    echo "=== Incomplete verification ==="
fi

echo "Config: $CONFIG"
echo ""

killall -9 python3 torchrun 2>/dev/null || true
sleep 2

echo "=========================================="
echo "  Single GPU baseline"
echo "=========================================="
cd "$VERIFIER_DIR"
python abcrown.py --config "$CONFIG" 2>&1 | tee /tmp/abcrown_single.log
echo ""

echo "=========================================="
echo "  FSDP = 2 GPUs"
echo "=========================================="
killall -9 python3 torchrun 2>/dev/null || true
sleep 2
torchrun --nproc_per_node=2 "$SCRIPT_DIR/run_abcrown_fsdp.py" \
    --config "$CONFIG" 2>&1 | tee /tmp/abcrown_fsdp.log
echo ""

echo "=========================================="
echo "  Summary"
echo "=========================================="
echo "--- Single GPU result ---"
grep -E "Result:|safe|unsafe|timeout|unknown|Peak GPU" /tmp/abcrown_single.log || true
echo ""
echo "--- FSDP=2 result ---"
grep -E "Result:|safe|unsafe|timeout|unknown|Peak GPU" /tmp/abcrown_fsdp.log || true
