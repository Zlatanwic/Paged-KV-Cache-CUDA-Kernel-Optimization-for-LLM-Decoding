/*
 * PyTorch binding for paged KV cache CUDA kernels.
 */

#include <torch/extension.h>
#include <vector>
#include <cmath>

// Declared in paged_gather.cu
extern "C" void launch_paged_gather(
    const float* kv_cache, float* output, const int* block_table,
    const int* context_lengths, int num_seqs, int num_heads,
    int block_size, int head_dim, int max_blocks_per_seq, int max_ctx_len
);

extern "C" void launch_naive_attention(
    const float* query, const float* k, const float* v, float* output,
    const int* context_lengths, int num_seqs, int num_heads,
    int head_dim, int max_ctx_len, float scale
);

extern "C" void launch_fused_paged_attention(
    const float* query, const float* k_cache, const float* v_cache,
    float* output, const int* block_table, const int* context_lengths,
    int num_seqs, int num_heads, int block_size, int head_dim,
    int max_blocks_per_seq, float scale
);

extern "C" void launch_fused_paged_attention_layout_b(
    const float* query, const float* k_cache, const float* v_cache,
    float* output, const int* block_table, const int* context_lengths,
    int num_seqs, int num_heads, int block_size, int head_dim,
    int max_blocks_per_seq, float scale
);

extern "C" void launch_fused_paged_attention_v3(
    const float* query, const float* k_cache, const float* v_cache,
    float* output, const int* block_table, const int* context_lengths,
    int num_seqs, int num_heads, int block_size, int head_dim,
    int max_blocks_per_seq, float scale
);

extern "C" void launch_fused_paged_attention_fp16(
    const void* query, const void* k_cache, const void* v_cache,
    void* output, const int* block_table, const int* context_lengths,
    int num_seqs, int num_heads, int block_size, int head_dim,
    int max_blocks_per_seq, float scale
);

// Helper: prepare block_table and context_lengths on GPU
static auto prepare_inputs(torch::Tensor block_table, torch::Tensor context_lengths,
                           torch::Device device) {
    auto bt = block_table.to(device).to(torch::kInt32).contiguous();
    auto cl = context_lengths.to(device).to(torch::kInt32).contiguous();
    return std::make_pair(bt, cl);
}

// Gather K/V from paged cache into contiguous tensors
std::vector<torch::Tensor> paged_gather(
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lengths, int max_ctx_len
) {
    int num_seqs = block_table.size(0);
    int max_blocks_per_seq = block_table.size(1);
    int num_heads = k_cache.size(1);
    int block_size = k_cache.size(2);
    int head_dim = k_cache.size(3);

    auto options = k_cache.options();
    auto k_out = torch::zeros({num_seqs, num_heads, max_ctx_len, head_dim}, options);
    auto v_out = torch::zeros({num_seqs, num_heads, max_ctx_len, head_dim}, options);

    auto [bt, cl] = prepare_inputs(block_table, context_lengths, k_cache.device());

    launch_paged_gather(k_cache.data_ptr<float>(), k_out.data_ptr<float>(),
        bt.data_ptr<int>(), cl.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, max_ctx_len);
    launch_paged_gather(v_cache.data_ptr<float>(), v_out.data_ptr<float>(),
        bt.data_ptr<int>(), cl.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, max_ctx_len);

    return {k_out, v_out};
}

// V1: gather + naive attention (two-step)
torch::Tensor paged_attention_v1(
    torch::Tensor query, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lengths, int max_ctx_len
) {
    int num_seqs = block_table.size(0);
    int num_heads = k_cache.size(1);
    int head_dim = k_cache.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto gathered = paged_gather(k_cache, v_cache, block_table, context_lengths, max_ctx_len);
    auto output = torch::zeros({num_seqs, num_heads, head_dim}, query.options());
    auto cl = context_lengths.to(query.device()).to(torch::kInt32).contiguous();

    launch_naive_attention(query.data_ptr<float>(),
        gathered[0].data_ptr<float>(), gathered[1].data_ptr<float>(),
        output.data_ptr<float>(), cl.data_ptr<int>(),
        num_seqs, num_heads, head_dim, max_ctx_len, scale);

    return output;
}

// V2: Fused paged attention, Layout A [num_blocks, num_heads, block_size, head_dim]
torch::Tensor paged_attention_v2(
    torch::Tensor query, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lengths
) {
    int num_seqs = block_table.size(0);
    int max_blocks_per_seq = block_table.size(1);
    int num_heads = k_cache.size(1);
    int block_size = k_cache.size(2);
    int head_dim = k_cache.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto output = torch::zeros({num_seqs, num_heads, head_dim}, query.options());
    auto [bt, cl] = prepare_inputs(block_table, context_lengths, query.device());

    launch_fused_paged_attention(query.data_ptr<float>(),
        k_cache.data_ptr<float>(), v_cache.data_ptr<float>(),
        output.data_ptr<float>(), bt.data_ptr<int>(), cl.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, scale);

    return output;
}

// V2-B: Fused paged attention, Layout B [num_blocks, block_size, num_heads, head_dim]
torch::Tensor paged_attention_v2_layout_b(
    torch::Tensor query,
    torch::Tensor k_cache,   // [num_blocks, block_size, num_heads, head_dim]
    torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lengths
) {
    int num_seqs = block_table.size(0);
    int max_blocks_per_seq = block_table.size(1);
    int num_heads = k_cache.size(2);   // Layout B: dim 2 is num_heads
    int block_size = k_cache.size(1);  // Layout B: dim 1 is block_size
    int head_dim = k_cache.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto output = torch::zeros({num_seqs, num_heads, head_dim}, query.options());
    auto [bt, cl] = prepare_inputs(block_table, context_lengths, query.device());

    launch_fused_paged_attention_layout_b(query.data_ptr<float>(),
        k_cache.data_ptr<float>(), v_cache.data_ptr<float>(),
        output.data_ptr<float>(), bt.data_ptr<int>(), cl.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, scale);

    return output;
}

// V3: Fused with shared memory K loading, Layout A
torch::Tensor paged_attention_v3(
    torch::Tensor query, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lengths
) {
    int num_seqs = block_table.size(0);
    int max_blocks_per_seq = block_table.size(1);
    int num_heads = k_cache.size(1);
    int block_size = k_cache.size(2);
    int head_dim = k_cache.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto output = torch::zeros({num_seqs, num_heads, head_dim}, query.options());
    auto [bt, cl] = prepare_inputs(block_table, context_lengths, query.device());

    launch_fused_paged_attention_v3(query.data_ptr<float>(),
        k_cache.data_ptr<float>(), v_cache.data_ptr<float>(),
        output.data_ptr<float>(), bt.data_ptr<int>(), cl.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, scale);

    return output;
}

// V2-FP16: Fused paged attention with half-precision KV cache
torch::Tensor paged_attention_v2_fp16(
    torch::Tensor query, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lengths
) {
    int num_seqs = block_table.size(0);
    int max_blocks_per_seq = block_table.size(1);
    int num_heads = k_cache.size(1);
    int block_size = k_cache.size(2);
    int head_dim = k_cache.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto output = torch::zeros({num_seqs, num_heads, head_dim}, query.options());
    auto [bt, cl] = prepare_inputs(block_table, context_lengths, query.device());

    launch_fused_paged_attention_fp16(query.data_ptr(),
        k_cache.data_ptr(), v_cache.data_ptr(),
        output.data_ptr(), bt.data_ptr<int>(), cl.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, scale);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_gather", &paged_gather, "Gather K/V from paged cache");
    m.def("paged_attention_v1", &paged_attention_v1, "V1: gather + naive attention");
    m.def("paged_attention_v2", &paged_attention_v2, "V2: Fused, Layout A");
    m.def("paged_attention_v2_layout_b", &paged_attention_v2_layout_b, "V2-B: Fused, Layout B");
    m.def("paged_attention_v3", &paged_attention_v3, "V3: Fused + shared memory K, Layout A");
    m.def("paged_attention_v2_fp16", &paged_attention_v2_fp16, "V2-FP16: Fused, half precision");
}
