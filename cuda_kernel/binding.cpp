/*
 * PyTorch binding for paged KV cache CUDA kernels.
 */

#include <torch/extension.h>
#include <vector>
#include <cmath>

// Declared in paged_gather.cu
extern "C" void launch_paged_gather(
    const float* kv_cache,
    float* output,
    const int* block_table,
    const int* context_lengths,
    int num_seqs,
    int num_heads,
    int block_size,
    int head_dim,
    int max_blocks_per_seq,
    int max_ctx_len
);

extern "C" void launch_fused_paged_attention(
    const float* query,
    const float* k_cache,
    const float* v_cache,
    float* output,
    const int* block_table,
    const int* context_lengths,
    int num_seqs,
    int num_heads,
    int block_size,
    int head_dim,
    int max_blocks_per_seq,
    float scale
);

extern "C" void launch_naive_attention(
    const float* query,
    const float* k,
    const float* v,
    float* output,
    const int* context_lengths,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_ctx_len,
    float scale
);

// Gather K/V from paged cache into contiguous tensors
std::vector<torch::Tensor> paged_gather(
    torch::Tensor k_cache,          // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor v_cache,          // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor block_table,      // [num_seqs, max_blocks_per_seq]
    torch::Tensor context_lengths,  // [num_seqs]
    int max_ctx_len
) {
    int num_seqs = block_table.size(0);
    int max_blocks_per_seq = block_table.size(1);
    int num_heads = k_cache.size(1);
    int block_size = k_cache.size(2);
    int head_dim = k_cache.size(3);

    auto options = k_cache.options();
    auto k_out = torch::zeros({num_seqs, num_heads, max_ctx_len, head_dim}, options);
    auto v_out = torch::zeros({num_seqs, num_heads, max_ctx_len, head_dim}, options);

    auto block_table_cuda = block_table.to(k_cache.device()).to(torch::kInt32).contiguous();
    auto ctx_lens_cuda = context_lengths.to(k_cache.device()).to(torch::kInt32).contiguous();

    launch_paged_gather(
        k_cache.data_ptr<float>(),
        k_out.data_ptr<float>(),
        block_table_cuda.data_ptr<int>(),
        ctx_lens_cuda.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, max_ctx_len
    );

    launch_paged_gather(
        v_cache.data_ptr<float>(),
        v_out.data_ptr<float>(),
        block_table_cuda.data_ptr<int>(),
        ctx_lens_cuda.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, max_ctx_len
    );

    return {k_out, v_out};
}

// End-to-end V1: gather + naive attention (two-step)
torch::Tensor paged_attention_v1(
    torch::Tensor query,            // [num_seqs, num_heads, head_dim]
    torch::Tensor k_cache,          // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor v_cache,          // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor block_table,      // [num_seqs, max_blocks_per_seq]
    torch::Tensor context_lengths,  // [num_seqs]
    int max_ctx_len
) {
    int num_seqs = block_table.size(0);
    int num_heads = k_cache.size(1);
    int head_dim = k_cache.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Step 1: Gather K/V into contiguous buffers
    auto gathered = paged_gather(k_cache, v_cache, block_table, context_lengths, max_ctx_len);
    auto& k_contig = gathered[0];  // [num_seqs, num_heads, max_ctx_len, head_dim]
    auto& v_contig = gathered[1];

    // Step 2: Run attention on gathered K/V
    auto options = query.options();
    auto output = torch::zeros({num_seqs, num_heads, head_dim}, options);

    auto ctx_lens_cuda = context_lengths.to(query.device()).to(torch::kInt32).contiguous();

    launch_naive_attention(
        query.data_ptr<float>(),
        k_contig.data_ptr<float>(),
        v_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        ctx_lens_cuda.data_ptr<int>(),
        num_seqs, num_heads, head_dim, max_ctx_len, scale
    );

    return output;
}

// V2: Fused paged attention (no intermediate buffer)
torch::Tensor paged_attention_v2(
    torch::Tensor query,            // [num_seqs, num_heads, head_dim]
    torch::Tensor k_cache,          // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor v_cache,          // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor block_table,      // [num_seqs, max_blocks_per_seq]
    torch::Tensor context_lengths   // [num_seqs]
) {
    int num_seqs = block_table.size(0);
    int max_blocks_per_seq = block_table.size(1);
    int num_heads = k_cache.size(1);
    int block_size = k_cache.size(2);
    int head_dim = k_cache.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto options = query.options();
    auto output = torch::zeros({num_seqs, num_heads, head_dim}, options);

    auto block_table_cuda = block_table.to(query.device()).to(torch::kInt32).contiguous();
    auto ctx_lens_cuda = context_lengths.to(query.device()).to(torch::kInt32).contiguous();

    launch_fused_paged_attention(
        query.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        output.data_ptr<float>(),
        block_table_cuda.data_ptr<int>(),
        ctx_lens_cuda.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, scale
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_gather", &paged_gather, "Gather K/V from paged cache into contiguous tensors");
    m.def("paged_attention_v1", &paged_attention_v1, "End-to-end V1: gather + naive attention");
    m.def("paged_attention_v2", &paged_attention_v2, "V2: Fused paged attention with online softmax");
}
