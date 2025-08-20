"""
Sparse SageAttention Benchmark and Correctness Test

This script validates the correctness and performance of sparse SageAttention 
by comparing it with dense SageAttention and measuring speedup.
"""

import argparse
import torch
from typing import List, Tuple

# Import SageAttention modules
from sageattention import sageattn
from sageattention.core import get_cuda_arch_versions
from utils import bench, calc_diff

def get_block_sizes_for_arch(device_idx: int = 0) -> Tuple[int, int]:
    """
    Get the appropriate CTA_Q and CTA_K block sizes based on GPU architecture.
    
    Args:
        device_idx: CUDA device index
        
    Returns:
        Tuple of (CTA_Q, CTA_K) for the given architecture
    """
    arch = get_cuda_arch_versions()[device_idx]
    
    if arch == "sm89":
        # SM89 architecture
        return 128, 64
    elif arch in "sm90":
        # SM90+ architectures
        return 64, 128
    else:
        raise NotImplementedError(f"Unsupported architecture: {arch}")

def create_sparse_block_indices(batch_size: int, num_heads: int, num_q_blocks: int, 
                                num_k_blocks: int, top_k: int, 
                                pattern: str = "random") -> torch.Tensor:
    """
    Create sparse block indices for sparse attention.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        num_q_blocks: Number of query blocks
        num_k_blocks: Number of key blocks
        top_k: Number of top K blocks to attend to
        pattern: Sparse pattern ("random", "local", "strided", "block_diagonal")
    
    Returns:
        block_index tensor of shape [batch_size, num_heads, num_q_blocks, top_k]
    """
    block_indices = torch.zeros(batch_size, num_heads, num_q_blocks, top_k, dtype=torch.int32, device="cuda")
    
    for b in range(batch_size):
        for h in range(num_heads):
            for q_idx in range(num_q_blocks):
                if pattern == "random":
                    # Random selection of K blocks
                    selected = torch.randperm(num_k_blocks, device="cuda")[:top_k]
                    block_indices[b, h, q_idx] = selected.sort()[0]
                    
                elif pattern == "local":
                    # Local attention pattern - attend to nearby blocks
                    window_size = top_k // 2
                    start = max(0, q_idx - window_size)
                    end = min(num_k_blocks, q_idx + window_size + 1)
                    candidates = list(range(start, end))
                    
                    # If not enough candidates, add random blocks
                    while len(candidates) < top_k:
                        rand_block = torch.randint(0, num_k_blocks, (1,)).item()
                        if rand_block not in candidates:
                            candidates.append(rand_block)
                    
                    selected = torch.tensor(candidates[:top_k], device="cuda")
                    block_indices[b, h, q_idx] = selected.sort()[0]
                    
                elif pattern == "strided":
                    # Strided pattern - attend to every few blocks
                    stride = max(1, num_k_blocks // top_k)
                    selected = torch.arange(0, num_k_blocks, stride, device="cuda")[:top_k]
                    selected_list = selected.tolist()
                    # Fill remaining with random if needed
                    while len(selected_list) < top_k:
                        rand_block = torch.randint(0, num_k_blocks, (1,)).item()
                        if rand_block not in selected_list:
                            selected_list.append(rand_block)
                    selected = torch.tensor(selected_list[:top_k], device="cuda")
                    block_indices[b, h, q_idx] = selected.sort()[0]
                    
                elif pattern == "block_diagonal":
                    # Block diagonal pattern
                    block_size = max(1, num_k_blocks // top_k)
                    start_block = (q_idx // block_size) * block_size
                    selected = torch.arange(start_block, min(start_block + top_k, num_k_blocks), device="cuda")
                    selected_list = selected.tolist()
                    # Fill remaining with nearby blocks if needed
                    while len(selected_list) < top_k:
                        candidates = list(range(max(0, start_block - 1), min(num_k_blocks, start_block + top_k + 1)))
                        added = False
                        for cand in candidates:
                            if cand not in selected_list and len(selected_list) < top_k:
                                selected_list.append(cand)
                                added = True
                                break
                        if not added and len(selected_list) < top_k:
                            # Add random block if no nearby candidates
                            for _ in range(10):  # Try up to 10 times to find a unique block
                                rand_block = torch.randint(0, num_k_blocks, (1,)).item()
                                if rand_block not in selected_list:
                                    selected_list.append(rand_block)
                                    break
                    selected = torch.tensor(selected_list[:top_k], device="cuda")
                    block_indices[b, h, q_idx] = selected.sort()[0]
    
    return block_indices

def generate_test_tensors(batch_size: int, num_heads: int, seq_len: int, head_dim: int, 
                         tensor_layout: str = "HND", dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, ...]:
    """Generate test tensors for attention computation."""
    
    if tensor_layout == "NHD":
        # [batch_size, seq_len, num_heads, head_dim] 
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    else:  # HND
        # [batch_size, num_heads, seq_len, head_dim]
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    
    return q, k, v

def test_correctness(batch_size: int = 2, num_heads: int = 8, seq_len: int = 2048, 
                    head_dim: int = 128, top_k_ratio: float = 0.5, 
                    pattern: str = "random", tensor_layout: str = "HND", 
                    qk_quant_gran: str = "per_thread", dtype: torch.dtype = torch.float16) -> float:
    """
    Test correctness of sparse attention by comparing with dense attention
    on the same subset of blocks.
    """
    assert tensor_layout == "HND", "Only HND layout is supported for correctness test"

    print(f"\n=== Correctness Test ===")
    print(f"Config: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
    print(f"Pattern: {pattern}, Top-K ratio: {top_k_ratio}")
    print(f"Layout: {tensor_layout}, Quantization: {qk_quant_gran}, Dtype: {dtype}")
    
    # Generate test data
    print("batch_size: ", batch_size, "num_heads: ", num_heads, "seq_len: ", seq_len, "head_dim: ", head_dim, "tensor_layout: ", tensor_layout, "dtype: ", dtype)
    q, k, v = generate_test_tensors(batch_size, num_heads, seq_len, head_dim, tensor_layout, dtype)
    print("q: ", q.shape, "k: ", k.shape, "v: ", v.shape)
    # Create sparse block indices - get block sizes based on GPU architecture
    CTA_Q, CTA_K = get_block_sizes_for_arch(q.device.index)
    num_q_blocks = (seq_len + CTA_Q - 1) // CTA_Q
    num_k_blocks = (seq_len + CTA_K - 1) // CTA_K
    
    arch = get_cuda_arch_versions()[q.device.index]
    print(f"🔧 Detected GPU architecture: {arch}, using block sizes CTA_Q={CTA_Q}, CTA_K={CTA_K}")
    top_k = max(1, min(num_k_blocks, int(num_k_blocks * top_k_ratio)))
    
    block_indices = create_sparse_block_indices(
        batch_size, num_heads, num_q_blocks, num_k_blocks, top_k, pattern)
    
    # Scale parameter
    sm_scale = 1 / (head_dim ** 0.5)
    
    try:        
        # Run sparse attention
        print("🔄 Running sparse attention...")
        o_sparse = sageattn(
            q, k, v,
            tensor_layout=tensor_layout,
            is_causal=False,
            qk_quant_gran=qk_quant_gran,
            sm_scale=sm_scale,
            smooth_k=True,
            return_lse=False,
            block_index=block_indices,
        )
        torch.cuda.synchronize()
        print("✅ Sparse attention executed successfully")
        # Create reference sparse output by manually masking dense output
        print("🔄 Creating reference sparse output for verification...")
        o_reference = torch.zeros_like(o_sparse)
        
        # For verification, we simulate sparse attention by selectively copying
        # from dense attention based on the block indices
        seq_dim = 1 if tensor_layout == "NHD" else 2
        # For each batch and head, compute reference output block separately
        for b in range(batch_size):
            for h in range(num_heads):
                for q_block_idx in range(num_q_blocks):
                    q_start = q_block_idx * CTA_Q
                    q_end = min(q_start + CTA_Q, seq_len)
                    # Get selected key blocks for this query block for this b, h
                    selected_k_blocks = block_indices[b, h, q_block_idx]  # [top_k]
                    k_segments = []
                    v_segments = []
                    for k_rel_idx in range(top_k):
                        k_block_idx = selected_k_blocks[k_rel_idx].item()
                        k_start = k_block_idx * CTA_K
                        k_end = min(k_start + CTA_K, seq_len)
                        k_segments.append(k[b:b+1, h:h+1, k_start:k_end, :])
                        v_segments.append(v[b:b+1, h:h+1, k_start:k_end, :])
                    k_concat = torch.cat(k_segments, dim=2)  # [1, 1, total_k_len, head_dim]
                    v_concat = torch.cat(v_segments, dim=2)  # [1, 1, total_k_len, head_dim]
                    q_block = q[b:b+1, h:h+1, q_start:q_end, :]  # [1, 1, q_block_size, head_dim]
                    print(f"q_block: {q_block.shape}, k_concat: {k_concat.shape}, v_concat: {v_concat.shape}, tensor_layout")
                    o_block = sageattn(
                        q_block, k_concat, v_concat,
                        tensor_layout=tensor_layout,
                        is_causal=False,
                        qk_quant_gran=qk_quant_gran,
                        sm_scale=sm_scale,
                        smooth_k=True,
                        return_lse=False,
                        block_index=None,
                    )
                    o_reference[b:b+1, h:h+1, q_start:q_end, :] = o_block

        print("✅ Reference computation completed")
        
        # Verify correctness by comparing outputs
        print("🔍 Verifying correctness...")
        
        # Calculate similarity metrics
        diff = calc_diff(o_sparse, o_reference)
        similarity = 1 - diff
        
        abs_error = torch.mean(torch.abs(o_sparse - o_reference)).item()
        relative_error = (abs_error / torch.mean(torch.abs(o_reference)).item()) * 100
        max_error = torch.max(torch.abs(o_sparse - o_reference)).item()
        
        print(f"📊 Correctness Results:")
        print(f"  - Cosine Similarity: {similarity:.6f} ({similarity*100:.4f}%)")
        print(f"  - Mean Absolute Error: {abs_error:.6e}")
        print(f"  - Relative Error: {relative_error:.4f}%")
        print(f"  - Max Absolute Error: {max_error:.6e}")
        import ipdb; ipdb.set_trace()

        if similarity < 0.99:
            import ipdb; ipdb.set_trace()
            raise ValueError(f"Cosine similarity is less than 0.99: {similarity:.6f}")
        
        # Calculate approximate memory savings
        dense_blocks = num_q_blocks * num_k_blocks
        sparse_blocks = num_q_blocks * top_k
        memory_savings = 1 - (sparse_blocks / dense_blocks)
        
        print(f"Block indices shape: {block_indices.shape}")
        print(f"Top-K blocks per query: {top_k}/{num_k_blocks} ({top_k_ratio:.1%})")
        print(f"Memory savings: {memory_savings:.1%} (using {sparse_blocks}/{dense_blocks} blocks)")
        
        # Note about verification approach
        print(f"\nℹ️  Note: This test verifies that sparse attention runs successfully")
        print(f"   and produces reasonable outputs. For exact correctness verification,")
        print(f"   a reference sparse implementation would be needed.")
        
        return memory_savings
        
    except Exception as e:
        print(f"❌ Attention computation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def benchmark_performance(batch_size: int = 4, num_heads: int = 32, head_dim: int = 128,
                         seq_lens: List[int] = [1024, 2048, 4096, 8192],
                         top_k_ratios: List[float] = [0.25, 0.5, 0.75],
                         pattern: str = "random", tensor_layout: str = "HND",
                         qk_quant_gran: str = "per_thread", dtype: torch.dtype = torch.float16) -> None:
    """
    Benchmark performance comparison between dense and sparse attention.
    """
    print(f"\n=== Performance Benchmark ===")
    print(f"Config: B={batch_size}, H={num_heads}, D={head_dim}")
    print(f"Pattern: {pattern}, Layout: {tensor_layout}, Quantization: {qk_quant_gran}")
    
    results = {
        'seq_len': [],
        'top_k_ratio': [],
        'dense_time': [],
        'sparse_time': [],
        'speedup': [],
        'memory_savings': []
    }
    
    for seq_len in seq_lens:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Generate test data
        q, k, v = generate_test_tensors(batch_size, num_heads, seq_len, head_dim, tensor_layout, dtype)
        
        sm_scale = 1 / (head_dim ** 0.5)
        
        # Dense attention baseline
        def dense_fn():
            return sageattn(
                q, k, v,
                tensor_layout=tensor_layout,
                is_causal=False,
                qk_quant_gran=qk_quant_gran,
                sm_scale=sm_scale,
                smooth_k=True,
                return_lse=False,
                block_index=None,
            )
        
        dense_time = bench(dense_fn, num_warmups=5, num_tests=20)
        
        # Test different sparsity levels - get block sizes based on GPU architecture
        CTA_Q, CTA_K = get_block_sizes_for_arch(q.device.index)
        num_q_blocks = (seq_len + CTA_Q - 1) // CTA_Q
        num_k_blocks = (seq_len + CTA_K - 1) // CTA_K
        
        arch = get_cuda_arch_versions()[q.device.index]
        print(f"🔧 Performance test using architecture: {arch}, block sizes CTA_Q={CTA_Q}, CTA_K={CTA_K}")
        
        for top_k_ratio in top_k_ratios:
            top_k = max(1, min(num_k_blocks, int(num_k_blocks * top_k_ratio)))
            
            # Create sparse block indices
            block_indices = create_sparse_block_indices(
                batch_size, num_heads, num_q_blocks, num_k_blocks, top_k, pattern)
            
            # Sparse attention
            def sparse_fn():
                return sageattn(
                    q, k, v,
                    tensor_layout=tensor_layout,
                    is_causal=False,
                    qk_quant_gran=qk_quant_gran,
                    sm_scale=sm_scale,
                    smooth_k=True,
                    return_lse=False,
                    block_index=block_indices,
                )
            
            try:
                sparse_time = bench(sparse_fn, num_warmups=5, num_tests=20)
                speedup = dense_time / sparse_time
                
                # Calculate memory savings
                dense_blocks = num_q_blocks * num_k_blocks
                sparse_blocks = num_q_blocks * top_k
                memory_savings = 1 - (sparse_blocks / dense_blocks)
                
                results['seq_len'].append(seq_len)
                results['top_k_ratio'].append(top_k_ratio)
                results['dense_time'].append(dense_time)
                results['sparse_time'].append(sparse_time)
                results['speedup'].append(speedup)
                results['memory_savings'].append(memory_savings)
                
                print(f"  Top-K {top_k_ratio:.0%}: Dense={dense_time:.2f}ms, "
                      f"Sparse={sparse_time:.2f}ms, Speedup={speedup:.2f}x, "
                      f"Memory savings={memory_savings:.1%}")
                
            except Exception as e:
                print(f"  Top-K {top_k_ratio:.0%}: Failed - {e}")
    
    # Print summary
    print(f"\n=== Performance Summary ===")
    for i, seq_len in enumerate(seq_lens):
        seq_results = [(r, s, m) for j, (sl, r, s, m) in enumerate(
            zip(results['seq_len'], results['top_k_ratio'], 
                results['speedup'], results['memory_savings'])) if sl == seq_len]
        
        if seq_results:
            print(f"Seq {seq_len}:")
            for ratio, speedup, mem_save in seq_results:
                print(f"  {ratio:.0%} sparse: {speedup:.2f}x speedup, {mem_save:.1%} memory saved")

def test_different_patterns(batch_size: int = 2, num_heads: int = 8, 
                           seq_len: int = 2048, head_dim: int = 128, 
                           top_k_ratio: float = 0.5, tensor_layout: str = "HND") -> None:
    """Test different sparse attention patterns."""
    print(f"\n=== Pattern Comparison ===")
    
    patterns = ["random", "local", "strided", "block_diagonal"]
    
    for pattern in patterns:
        try:
            memory_savings = test_correctness(
                batch_size, num_heads, seq_len, head_dim, top_k_ratio, pattern, tensor_layout)
            print(f"Pattern '{pattern}': {memory_savings:.1%} memory savings")
        except Exception as e:
            print(f"Pattern '{pattern}': Failed - {e}")

def main():
    parser = argparse.ArgumentParser(description='Sparse SageAttention Benchmark')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--seq_lens', nargs='+', type=int, default=[1024, 2048, 4096], 
                       help='Sequence lengths to test')
    parser.add_argument('--top_k_ratios', nargs='+', type=float, default=[0.25, 0.5, 0.75], 
                       help='Top-K ratios to test')
    parser.add_argument('--pattern', type=str, default='random', 
                       choices=['random', 'local', 'strided', 'block_diagonal'],
                       help='Sparse attention pattern')
    parser.add_argument('--tensor_layout', type=str, default='HND', 
                       choices=['NHD', 'HND'], help='Tensor layout')
    parser.add_argument('--qk_quant_gran', type=str, default='per_thread', 
                       choices=['per_warp', 'per_thread'], help='Quantization granularity')
    parser.add_argument('--dtype', type=str, default='float16', 
                       choices=['float16', 'bfloat16'], help='Data type')
    parser.add_argument('--test_correctness', action='store_true', 
                       help='Run correctness tests')
    parser.add_argument('--test_patterns', action='store_true', 
                       help='Test different sparse patterns')
    parser.add_argument('--benchmark_perf', action='store_true', 
                       help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16
    
    print("🔍 Sparse SageAttention Test Suite")
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    
    # Display GPU architecture and block sizes
    arch = get_cuda_arch_versions()[0]
    CTA_Q, CTA_K = get_block_sizes_for_arch(0)
    print(f"🏗️ GPU Architecture: {arch}")
    print(f"📏 Block Sizes: CTA_Q={CTA_Q}, CTA_K={CTA_K}")
    
    try:
        if args.test_correctness or not any([args.test_patterns, args.benchmark_perf]):
            test_correctness(args.batch_size, args.num_heads, 
                           args.seq_lens[0] if args.seq_lens else 2048, 
                           args.head_dim, args.top_k_ratios[0] if args.top_k_ratios else 0.5, 
                           args.pattern, args.tensor_layout, args.qk_quant_gran, dtype)
        
        if args.test_patterns:
            test_different_patterns(args.batch_size, args.num_heads, 
                                  args.seq_lens[0] if args.seq_lens else 2048, 
                                  args.head_dim, args.top_k_ratios[0] if args.top_k_ratios else 0.5,
                                  args.tensor_layout)
        
        if args.benchmark_perf:
            benchmark_performance(args.batch_size, args.num_heads, args.head_dim,
                                args.seq_lens, args.top_k_ratios, args.pattern, 
                                args.tensor_layout, args.qk_quant_gran, dtype)
                                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
