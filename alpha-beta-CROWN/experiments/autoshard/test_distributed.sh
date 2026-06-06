#!/bin/bash
# Run the full auto-shard test suite on the GPU VM.
# Usage: bash test_distributed.sh
set -e

cd "$(dirname "$0")"

echo "=== Step 0: Local CPU tests ==="
python test_local.py

echo ""
echo "=== Step 1: Single GPU reference ==="
python run.py --mode single --method CROWN --save ref.pt
echo ""

echo "=== Step 2: TP=1 vs reference (CROWN) ==="
torchrun --nproc_per_node=1 --master_port=29520 run.py --mode tp --method CROWN --compare ref.pt
echo ""

echo "=== Step 3: TP=2 vs reference (CROWN) ==="
torchrun --nproc_per_node=2 --master_port=29521 run.py --mode tp --method CROWN --compare ref.pt
echo ""

echo "=== Step 4: Single GPU reference (alpha-CROWN) ==="
python run.py --mode single --method alpha-CROWN --save ref_alpha.pt
echo ""

echo "=== Step 5: TP=2 vs reference (alpha-CROWN) ==="
torchrun --nproc_per_node=2 --master_port=29522 run.py --mode tp --method alpha-CROWN --compare ref_alpha.pt
echo ""

echo "=== Step 6: 3-layer MLP (CROWN) ==="
python run.py --mode single --method CROWN --n-hidden 2 --hidden-dim 16 --save ref_3layer.pt
torchrun --nproc_per_node=2 --master_port=29523 run.py --mode tp --method CROWN --n-hidden 2 --hidden-dim 16 --compare ref_3layer.pt
echo ""

echo "=== ALL DISTRIBUTED TESTS DONE ==="
