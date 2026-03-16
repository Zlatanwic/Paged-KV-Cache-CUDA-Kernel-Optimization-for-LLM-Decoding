"""Benchmark: fragmentation impact on V2 kernel performance.

Measures how block placement fragmentation affects latency and
links to KV cache eviction research question.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline_pytorch"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cuda_kernel"))

import torch
import time
import csv
from model import PagedKVCacheConfig, generate_paged_kv_cache, generate_block_table
from build import load_paged_gather

print("Compiling CUDA kernels...")
cuda_module = load_paged_gather()
print("Done.\n")


def make_data_fixed(num_seqs, ctx_len, block_size, num_heads, head_dim, seed=42):
    """Create KV cache once, return reusable data (no block table yet)."""
    torch.manual_seed(seed)
    num_blocks = (ctx_len // block_size + 1) * num_seqs + 64
    config = PagedKVCacheConfig(
        num_blocks=num_blocks, block_size=block_size, num_heads=num_heads,
        head_dim=head_dim, num_seqs=num_seqs, max_context_len=ctx_len, device="cuda",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
    max_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    return k_cache, v_cache, query, context_lengths, num_blocks, max_blocks_per_seq


def make_block_table(num_seqs, max_blocks_per_seq, num_blocks, context_lengths, block_size, frag):
    """Create block table with given fragmentation level."""
    return generate_block_table(
        num_seqs, max_blocks_per_seq, num_blocks, context_lengths, block_size,
        fragmentation=frag,
    )


def bench(fn, *args, warmup=10, runs=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(*args)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000  # ms


# Experiment matrix
num_heads = 16
head_dim = 128
frag_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

configs = [
    # (num_seqs, ctx_len, block_size)
    (1,  4096,  16),
    (1,  16384, 16),
    (1,  4096,  32),
    (1,  16384, 32),
    (8,  4096,  16),
    (8,  16384, 16),
]

results = []

print("=" * 80)
print("FRAGMENTATION IMPACT BENCHMARK")
print("=" * 80)

for num_seqs, ctx_len, block_size in configs:
    print(f"\n  seqs={num_seqs}, ctx={ctx_len}, bs={block_size}, h={num_heads}, d={head_dim}")

    # Allocate KV cache once, only vary block table
    k, v, q, cl, num_blocks, max_bps = make_data_fixed(
        num_seqs, ctx_len, block_size, num_heads, head_dim)

    base_ms = None
    for frag in frag_levels:
        bt = make_block_table(num_seqs, max_bps, num_blocks, cl, block_size, frag)
        ms = bench(cuda_module.paged_attention_v2, q, k, v, bt, cl)

        if frag == 0.0:
            base_ms = ms

        degradation = (ms - base_ms) / base_ms * 100 if base_ms else 0.0
        print(f"    frag={frag:.2f}  {ms:.4f} ms  ({degradation:+.1f}%)")

        results.append({
            "num_seqs": num_seqs,
            "ctx_len": ctx_len,
            "block_size": block_size,
            "fragmentation": frag,
            "latency_ms": round(ms, 4),
            "degradation_pct": round(degradation, 2),
        })

    # Free GPU memory between configs
    del k, v, q, cl
    torch.cuda.empty_cache()

# Save CSV
csv_path = os.path.join(os.path.dirname(__file__), "..", "profile", "fragmentation_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {csv_path}")
