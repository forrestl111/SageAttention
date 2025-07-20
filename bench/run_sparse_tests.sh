#!/bin/bash

# Sparse SageAttention Test Runner
# This script runs various tests for sparse SageAttention

echo "🔍 Sparse SageAttention Test Suite Runner"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "bench_sparse_sage.py" ]; then
    echo "❌ Error: bench_sparse_sage.py not found. Please run from SageAttention/bench directory."
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run test with error handling
run_test() {
    local test_name="$1"
    local cmd="$2"
    
    echo -e "\n${BLUE}=== Running: $test_name ===${NC}"
    echo "Command: $cmd"
    echo
    
    if eval "$cmd"; then
        echo -e "${GREEN}✅ $test_name completed successfully${NC}"
    else
        echo -e "${RED}❌ $test_name failed${NC}"
        return 1
    fi
}

# Default configuration
BATCH_SIZE=2
NUM_HEADS=8
HEAD_DIM=128
SEQ_LENS="1024 2048"
TOP_K_RATIOS="0.25 0.5 0.75"

echo "Configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Heads: $NUM_HEADS"
echo "  Head Dim: $HEAD_DIM"
echo "  Sequence Lengths: $SEQ_LENS"
echo "  Top-K Ratios: $TOP_K_RATIOS"

# Test 1: Basic Correctness Test
run_test "Basic Correctness Test" \
    "python bench_sparse_sage.py --batch_size $BATCH_SIZE --num_heads $NUM_HEADS --head_dim $HEAD_DIM --test_correctness"

# Test 2: Different Sparse Patterns
run_test "Sparse Pattern Comparison" \
    "python bench_sparse_sage.py --batch_size $BATCH_SIZE --num_heads $NUM_HEADS --head_dim $HEAD_DIM --test_patterns"

# Test 3: Performance Benchmark (small scale for quick test)
run_test "Performance Benchmark (Quick)" \
    "python bench_sparse_sage.py --batch_size $BATCH_SIZE --num_heads $NUM_HEADS --head_dim $HEAD_DIM --seq_lens $SEQ_LENS --top_k_ratios $TOP_K_RATIOS --benchmark_perf"

# Test 4: Different Quantization Granularities
echo -e "\n${YELLOW}=== Testing Different Quantization Granularities ===${NC}"

run_test "Per-Warp Quantization" \
    "python bench_sparse_sage.py --batch_size $BATCH_SIZE --num_heads $NUM_HEADS --head_dim $HEAD_DIM --quant_gran per_warp --test_correctness"

run_test "Per-Thread Quantization" \
    "python bench_sparse_sage.py --batch_size $BATCH_SIZE --num_heads $NUM_HEADS --head_dim $HEAD_DIM --quant_gran per_thread --test_correctness"

# Test 5: Different Sparse Patterns with Performance
echo -e "\n${YELLOW}=== Testing Different Patterns with Performance ===${NC}"

PATTERNS=("random" "local" "strided" "block_diagonal")

for pattern in "${PATTERNS[@]}"; do
    run_test "Pattern: $pattern" \
        "python bench_sparse_sage.py --batch_size $BATCH_SIZE --num_heads $NUM_HEADS --head_dim $HEAD_DIM --pattern $pattern --seq_lens 1024 --top_k_ratios 0.5 --benchmark_perf"
done

# Test 6: Stress Test (if requested)
if [ "$1" = "--stress" ]; then
    echo -e "\n${YELLOW}=== Stress Test (Large Configurations) ===${NC}"
    
    run_test "Large Scale Test" \
        "python bench_sparse_sage.py --batch_size 4 --num_heads 32 --head_dim 128 --seq_lens 4096 8192 --top_k_ratios 0.25 0.5 --benchmark_perf"
fi

# Test 7: Memory Usage Analysis
echo -e "\n${YELLOW}=== Memory Usage Analysis ===${NC}"

run_test "Memory Analysis" \
    "python bench_sparse_sage.py --batch_size 1 --num_heads 4 --head_dim 64 --seq_lens 2048 4096 --top_k_ratios 0.1 0.25 0.5 0.75 --benchmark_perf"

echo -e "\n${GREEN}🎉 All tests completed!${NC}"
echo
echo "Usage examples:"
echo "  ./run_sparse_tests.sh                # Run standard tests"
echo "  ./run_sparse_tests.sh --stress       # Include stress tests"
echo 
echo "Individual test examples:"
echo "  python bench_sparse_sage.py --test_correctness"
echo "  python bench_sparse_sage.py --test_patterns"
echo "  python bench_sparse_sage.py --benchmark_perf --seq_lens 1024 2048 --top_k_ratios 0.25 0.5"
echo "  python bench_sparse_sage.py --pattern local --quant_gran per_thread --test_correctness" 