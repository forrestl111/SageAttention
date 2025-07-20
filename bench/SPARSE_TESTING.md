# Sparse SageAttention Testing Suite

This directory contains comprehensive tests for validating the correctness and performance of Sparse SageAttention.

## Overview

The sparse attention implementation allows selecting only a subset of key-value blocks for each query block, significantly reducing memory usage and computation while maintaining attention quality for many applications.

## Files

- `bench_sparse_sage.py` - Main testing script
- `run_sparse_tests.sh` - Automated test runner
- `SPARSE_TESTING.md` - This documentation

## Quick Start

### 1. Basic Usage

```bash
# Make the script executable
chmod +x run_sparse_tests.sh

# Run all standard tests
./run_sparse_tests.sh

# Run with stress tests (larger configurations)
./run_sparse_tests.sh --stress
```

### 2. Individual Tests

```bash
# Test basic correctness
python bench_sparse_sage.py --test_correctness

# Test different sparse patterns
python bench_sparse_sage.py --test_patterns

# Run performance benchmarks
python bench_sparse_sage.py --benchmark_perf
```

## Test Categories

### 1. Correctness Tests (`--test_correctness`)

Validates that sparse attention produces mathematically correct results:
- ✅ Kernel execution without errors
- ✅ Shape compatibility
- ✅ Memory safety
- ✅ Block index validation

### 2. Pattern Tests (`--test_patterns`)

Tests different sparse attention patterns:
- **Random**: Random selection of K blocks
- **Local**: Attend to nearby blocks (sliding window)
- **Strided**: Uniform sampling across sequence
- **Block Diagonal**: Block-structured patterns

### 3. Performance Tests (`--benchmark_perf`)

Measures performance compared to dense attention:
- 🚀 Execution time comparison
- 💾 Memory usage analysis
- ⚡ Speedup measurements
- 📊 Scaling behavior

## Configuration Options

### Basic Parameters

```bash
python bench_sparse_sage.py \
    --batch_size 4 \
    --num_heads 32 \
    --head_dim 128 \
    --seq_lens 1024 2048 4096 \
    --top_k_ratios 0.25 0.5 0.75
```

### Sparse Patterns

```bash
# Test different patterns
--pattern random          # Random block selection
--pattern local           # Local/sliding window
--pattern strided         # Strided sampling
--pattern block_diagonal  # Block diagonal structure
```

### Quantization Granularity

```bash
--quant_gran per_warp     # Per-warp quantization (default)
--quant_gran per_thread   # Per-thread quantization
```

## Expected Results

### Memory Savings
- **25% sparse**: ~75% memory reduction
- **50% sparse**: ~50% memory reduction  
- **75% sparse**: ~25% memory reduction

### Performance Characteristics
- **Small sequences (≤2K)**: Minimal speedup due to overhead
- **Medium sequences (2K-8K)**: Moderate speedup (1.2-2x)
- **Large sequences (≥8K)**: Significant speedup (2-4x)

### Pattern Performance
- **Random**: Best memory distribution
- **Local**: Best cache locality
- **Strided**: Balanced coverage
- **Block Diagonal**: Best for structured data

## Example Output

```
🔍 Sparse SageAttention Test Suite
CUDA Device: NVIDIA H100 80GB HBM3
CUDA Capability: (9, 0)

=== Correctness Test ===
Config: B=2, H=8, S=2048, D=128
Pattern: random, Top-K ratio: 0.5
✅ Sparse attention executed successfully
Block indices shape: torch.Size([2, 8, 32, 8])
Top-K blocks per query: 8/16 (50.0%)
Memory savings: 50.0% (using 256/512 blocks)

=== Performance Benchmark ===
Config: B=4, H=32, D=128
Pattern: random, Quant granularity: per_warp

Testing sequence length: 1024
  Top-K 25%: Dense=1.23ms, Sparse=0.95ms, Speedup=1.29x, Memory savings=75.0%
  Top-K 50%: Dense=1.23ms, Sparse=1.08ms, Speedup=1.14x, Memory savings=50.0%
  Top-K 75%: Dense=1.23ms, Sparse=1.18ms, Speedup=1.04x, Memory savings=25.0%
```

## Troubleshooting

### Common Issues

1. **"SM90 sparse attention kernel not found!"**
   - Ensure you're running on SM90+ GPU (H100, etc.)
   - Verify CUDA compilation flags include SM90

2. **"block_index contains invalid block index"**
   - Block indices must be in range [0, num_k_blocks-1]
   - Check sparse pattern generation logic

3. **Out of memory errors**
   - Reduce batch size or sequence length
   - Increase sparsity (lower top_k_ratio)

### Performance Tips

1. **For maximum speedup**: Use high sparsity (≤50%) on long sequences
2. **For accuracy**: Use local or structured patterns
3. **For memory efficiency**: Use higher sparsity ratios

## Implementation Details

### Block Layout
- Query blocks: 64 tokens per block (CTA_Q=64)
- Key/Value blocks: 128 tokens per block (CTA_K=128)
- Block indices shape: [batch, heads, q_blocks, top_k]

### Supported Configurations
- **Batch sizes**: 1-16
- **Sequence lengths**: 1K-32K+ (divisible by 64)
- **Head dimensions**: 64, 128 (extendable)
- **Sparsity ratios**: 10%-90%

### Memory Requirements
```
Dense attention:     O(B × H × S²)
Sparse attention:    O(B × H × S² × sparsity_ratio)
Block indices:       O(B × H × (S/64) × top_k × 4 bytes)
```

## Contributing

To add new test patterns or configurations:

1. Extend `create_sparse_block_indices()` for new patterns
2. Add pattern choice to argument parser
3. Update documentation with expected behavior
4. Test on multiple GPU architectures

## References

- [SageAttention Paper](https://arxiv.org/abs/2410.02367)
- [Sparse Attention Survey](https://arxiv.org/abs/2009.14794)
- [CUDA WGMMA Documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/) 