"""Test and benchmark FP16 V2 kernel vs FP32 V2."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline_pytorch"))

import torch
import time
from model import PagedKVCacheConfig, generate_paged_kv_cache, generate_block_table
from build import load_paged_gather

print("Compiling CUDA kernels...")
cuda_module = load_paged_gather()
print("Done.\n")


def make_test_data(num_seqs, ctx_len, block_size, num_heads, head_dim, seed=42):
    torch.manual_seed(seed)
    num_blocks = (ctx_len // block_size + 1) * num_seqs + 16
    config = PagedKVCacheConfig(
        num_blocks=num_blocks, block_size=block_size, num_heads=num_heads,
        head_dim=head_dim, num_seqs=num_seqs, max_context_len=ctx_len, device="cuda",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
    max_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    block_table = generate_block_table(
        num_seqs, max_blocks_per_seq, num_blocks, context_lengths, block_size,
    )
    return k_cache, v_cache, query, block_table, context_lengths


# --- Correctness ---
print("=== FP16 Correctness ===")
all_ok = True
for ctx, bs in [(32, 16), (64, 16), (1024, 16), (4096, 16), (4096, 32)]:
    k32, v32, q32, bt, cl = make_test_data(1, ctx, bs, 16, 128)

    # FP32 reference
    ref = cuda_module.paged_attention_v2(q32, k32, v32, bt, cl)

    # FP16 version
    k16, v16, q16 = k32.half(), v32.half(), q32.half()
    out = cuda_module.paged_attention_v2_fp16(q16, k16, v16, bt, cl)

    # Compare in float32
    max_diff = (out.float() - ref).abs().max().item()
    # FP16 has ~1e-3 precision, longer contexts accumulate more error
    atol = 5e-3 if ctx > 1024 else 2e-3
    ok = max_diff < atol
    tag = "OK" if ok else "FAIL"
    print(f"  [{tag}] ctx={ctx}, bs={bs}  (max_diff={max_diff:.2e}, atol={atol:.0e})")
    all_ok &= ok

if not all_ok:
    print("\nSome tests FAILED!")
    sys.exit(1)
print("\nAll correctness tests passed!\n")


# --- Benchmark ---
def bench(fn, *args, warmup=10, runs=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(*args)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


print("=" * 70)
print("BENCHMARK: FP32 V2 vs FP16 V2")
print("=" * 70)

configs = [
    (1,  1024,  16),
    (1,  4096,  16),
    (1,  16384, 16),
    (1,  4096,  32),
    (8,  4096,  16),
    (8,  16384, 16),
]

for num_seqs, ctx_len, block_size in configs:
    k32, v32, q32, bt, cl = make_test_data(num_seqs, ctx_len, block_size, 16, 128)
    k16, v16, q16 = k32.half(), v32.half(), q32.half()

    ms_fp32 = bench(cuda_module.paged_attention_v2, q32, k32, v32, bt, cl)
    ms_fp16 = bench(cuda_module.paged_attention_v2_fp16, q16, k16, v16, bt, cl)

    speedup = ms_fp32 / ms_fp16
    mem_fp32 = (k32.nelement() + v32.nelement()) * 4 / 1e6  # MB
    mem_fp16 = (k16.nelement() + v16.nelement()) * 2 / 1e6

    print(f"\n  seqs={num_seqs}, ctx={ctx_len}, bs={block_size}")
    print(f"    FP32: {ms_fp32:.3f} ms  (KV cache: {mem_fp32:.1f} MB)")
    print(f"    FP16: {ms_fp16:.3f} ms  (KV cache: {mem_fp16:.1f} MB)")
    print(f"    Speedup: {speedup:.2f}x")

    del k32, v32, q32, k16, v16, q16
    torch.cuda.empty_cache()
