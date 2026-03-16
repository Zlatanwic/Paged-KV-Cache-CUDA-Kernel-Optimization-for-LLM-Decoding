"""Test V2 fused paged attention against PyTorch baseline and V1."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline_pytorch"))

import torch
import time
from model import (
    PagedKVCacheConfig, generate_paged_kv_cache,
    generate_block_table, paged_attention_naive,
)
from build import load_paged_gather

print("Compiling CUDA kernels...")
cuda_module = load_paged_gather()
print("Done.\n")


def test_v2_correctness(num_seqs, ctx_len, block_size, num_heads, head_dim, frag=0.0):
    """Compare V2 output with PyTorch baseline."""
    torch.manual_seed(42)
    num_blocks = (ctx_len // block_size + 1) * num_seqs + 16
    config = PagedKVCacheConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_seqs=num_seqs,
        max_context_len=ctx_len,
        device="cuda",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
    max_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    block_table = generate_block_table(
        num_seqs, max_blocks_per_seq, num_blocks, context_lengths, block_size,
        fragmentation=frag,
    )

    # PyTorch baseline
    ref = paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)

    # CUDA V2
    out = cuda_module.paged_attention_v2(
        query, k_cache, v_cache, block_table, context_lengths
    )

    max_diff = (out - ref).abs().max().item()
    atol = 1e-3 if ctx_len > 1024 else 1e-4
    ok = max_diff < atol
    tag = "OK" if ok else "FAIL"
    desc = f"seqs={num_seqs}, ctx={ctx_len}, bs={block_size}, h={num_heads}, d={head_dim}, frag={frag}"
    print(f"  [{tag}] {desc}  (max_diff={max_diff:.2e})")
    if not ok:
        # Print more debug info
        rel_err = ((out - ref).abs() / (ref.abs() + 1e-8)).max().item()
        print(f"         rel_err={rel_err:.2e}")
    return ok


def benchmark_all():
    """Benchmark PyTorch baseline vs V1 vs V2."""
    print("\n=== Timing: PyTorch vs V1 vs V2 ===")
    torch.manual_seed(0)

    configs = [
        (1, 1024, 16, 16, 128),
        (1, 4096, 16, 16, 128),
        (1, 4096, 32, 16, 128),
    ]

    for num_seqs, ctx_len, block_size, num_heads, head_dim in configs:
        num_blocks = (ctx_len // block_size) * num_seqs + 16
        config = PagedKVCacheConfig(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_seqs=num_seqs,
            max_context_len=ctx_len,
            device="cuda",
        )
        k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
        context_lengths = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
        max_blocks_per_seq = ctx_len // block_size
        block_table = generate_block_table(
            num_seqs, max_blocks_per_seq, num_blocks, context_lengths, block_size,
        )

        # Warmup
        for _ in range(5):
            paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)
            cuda_module.paged_attention_v1(query, k_cache, v_cache, block_table, context_lengths, ctx_len)
            cuda_module.paged_attention_v2(query, k_cache, v_cache, block_table, context_lengths)
        torch.cuda.synchronize()

        num_runs = 20

        # PyTorch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)
            torch.cuda.synchronize()
        pytorch_ms = (time.perf_counter() - t0) / num_runs * 1000

        # V1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            cuda_module.paged_attention_v1(query, k_cache, v_cache, block_table, context_lengths, ctx_len)
            torch.cuda.synchronize()
        v1_ms = (time.perf_counter() - t0) / num_runs * 1000

        # V2
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            cuda_module.paged_attention_v2(query, k_cache, v_cache, block_table, context_lengths)
            torch.cuda.synchronize()
        v2_ms = (time.perf_counter() - t0) / num_runs * 1000

        print(f"\n  ctx={ctx_len}, bs={block_size}, h={num_heads}, d={head_dim}")
        print(f"    PyTorch: {pytorch_ms:.3f} ms")
        print(f"    V1:      {v1_ms:.3f} ms  ({pytorch_ms/v1_ms:.2f}x vs PyTorch)")
        print(f"    V2:      {v2_ms:.3f} ms  ({pytorch_ms/v2_ms:.2f}x vs PyTorch, {v1_ms/v2_ms:.2f}x vs V1)")


if __name__ == "__main__":
    print("=== V2 Correctness Tests ===")
    all_ok = True
    all_ok &= test_v2_correctness(1, 32, 16, 4, 64)
    all_ok &= test_v2_correctness(1, 48, 16, 4, 64)        # partial block
    all_ok &= test_v2_correctness(2, 64, 16, 4, 64)         # multi-seq
    all_ok &= test_v2_correctness(1, 64, 16, 4, 64, frag=1.0)  # fragmented
    all_ok &= test_v2_correctness(1, 1024, 16, 16, 128)
    all_ok &= test_v2_correctness(1, 4096, 16, 16, 128)
    all_ok &= test_v2_correctness(1, 4096, 32, 16, 128)
    all_ok &= test_v2_correctness(1, 4096, 64, 16, 128)

    if all_ok:
        print("\nAll V2 correctness tests passed!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)

    benchmark_all()
