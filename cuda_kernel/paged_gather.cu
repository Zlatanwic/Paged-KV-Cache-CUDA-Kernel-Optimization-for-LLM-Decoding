/*
 * Paged KV Cache Gather Kernel (V1 - Naive)
 *
 * KV cache layout: [num_blocks, num_heads, block_size, head_dim]
 * Block table:     [num_seqs, max_blocks_per_seq]
 * Output K/V:      [num_seqs, num_heads, max_context_len, head_dim]
 */

#include <cuda_runtime.h>

// Naive gather kernel: each thread copies one element
__global__ void paged_gather_kernel(
    const float* __restrict__ kv_cache,
    float* __restrict__ output,
    const int* __restrict__ block_table,
    const int* __restrict__ context_lengths,
    int num_heads,
    int block_size,
    int head_dim,
    int max_blocks_per_seq,
    int max_ctx_len
) {
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;
    int ctx_len = context_lengths[seq_id];
    int total_elements = ctx_len * head_dim;

    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        int token_pos = idx / head_dim;
        int d = idx % head_dim;

        int logical_block = token_pos / block_size;
        int offset_in_block = token_pos % block_size;

        int physical_block = block_table[seq_id * max_blocks_per_seq + logical_block];

        int cache_idx = physical_block * (num_heads * block_size * head_dim)
                      + head_id * (block_size * head_dim)
                      + offset_in_block * head_dim
                      + d;
        float val = kv_cache[cache_idx];

        int out_idx = seq_id * (num_heads * max_ctx_len * head_dim)
                    + head_id * (max_ctx_len * head_dim)
                    + token_pos * head_dim
                    + d;
        output[out_idx] = val;
    }
}

/*
 * Naive Attention Kernel
 *
 * Computes single-query attention on contiguous (already gathered) K/V.
 * query:  [num_seqs, num_heads, head_dim]
 * k:      [num_seqs, num_heads, max_ctx_len, head_dim]
 * v:      [num_seqs, num_heads, max_ctx_len, head_dim]
 * output: [num_seqs, num_heads, head_dim]
 *
 * Each CUDA block handles one (seq, head) pair.
 * Step 1: compute scores = q . k[t] * scale for all t
 * Step 2: softmax over scores
 * Step 3: output = sum(scores[t] * v[t])
 */
__global__ void naive_attention_kernel(
    const float* __restrict__ query,   // [num_seqs, num_heads, head_dim]
    const float* __restrict__ k,       // [num_seqs, num_heads, max_ctx_len, head_dim]
    const float* __restrict__ v,       // [num_seqs, num_heads, max_ctx_len, head_dim]
    float* __restrict__ output,        // [num_seqs, num_heads, head_dim]
    const int* __restrict__ context_lengths,
    int num_heads,
    int head_dim,
    int max_ctx_len,
    float scale
) {
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int ctx_len = context_lengths[seq_id];

    // Pointers for this (seq, head)
    const float* q_ptr = query + seq_id * (num_heads * head_dim)
                               + head_id * head_dim;
    const float* k_ptr = k + seq_id * (num_heads * max_ctx_len * head_dim)
                           + head_id * (max_ctx_len * head_dim);
    const float* v_ptr = v + seq_id * (num_heads * max_ctx_len * head_dim)
                           + head_id * (max_ctx_len * head_dim);
    float* out_ptr = output + seq_id * (num_heads * head_dim)
                            + head_id * head_dim;

    // --- Step 1: Compute attention scores ---
    // Use shared memory for scores (one per token position)
    extern __shared__ float shared[];
    float* scores = shared;  // [ctx_len]

    for (int t = tid; t < ctx_len; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[t * head_dim + d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // --- Step 2: Softmax ---
    // Find max (parallel reduction)
    float local_max = -1e20f;
    for (int t = tid; t < ctx_len; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    // Warp-level reduction for max
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    // Block-level reduction: use first element of each warp
    __shared__ float warp_maxes[32];  // max 32 warps per block (1024 threads)
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_maxes[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float block_max = -1e20f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            block_max = fmaxf(block_max, warp_maxes[w]);
        }
        warp_maxes[0] = block_max;  // store global max in warp_maxes[0]
    }
    __syncthreads();
    float global_max = warp_maxes[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < ctx_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - global_max);
        local_sum += scores[t];
    }
    // Warp-level reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    __shared__ float warp_sums[32];
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float block_sum = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            block_sum += warp_sums[w];
        }
        warp_sums[0] = block_sum;
    }
    __syncthreads();
    float global_sum = warp_sums[0];

    // Normalize scores
    for (int t = tid; t < ctx_len; t += blockDim.x) {
        scores[t] /= global_sum;
    }
    __syncthreads();

    // --- Step 3: Weighted sum of V ---
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            acc += scores[t] * v_ptr[t * head_dim + d];
        }
        out_ptr[d] = acc;
    }
}

// C interface for attention kernel launch
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
) {
    dim3 grid(num_seqs, num_heads);
    int threads = 256;
    // Shared memory: scores array sized to max_ctx_len
    int shared_mem_size = max_ctx_len * sizeof(float);
    naive_attention_kernel<<<grid, threads, shared_mem_size>>>(
        query, k, v, output, context_lengths,
        num_heads, head_dim, max_ctx_len, scale
    );
}

/*
 * V2: Fused Paged Attention Kernel (Warp-Parallel)
 *
 * Reads K/V directly from paged cache (no intermediate buffer).
 * Uses online softmax in a single pass.
 *
 * Key design: each WARP independently processes a subset of tokens.
 *   - Within a warp, 32 threads split head_dim (each handles head_dim/32 dims)
 *   - Warp-level reduction for dot products (no __syncthreads needed!)
 *   - Each warp maintains its own online softmax state
 *   - At the end, merge all warps' results with one block-level sync
 *
 * This reduces sync count from O(ctx_len) to O(1).
 */
__global__ void fused_paged_attention_kernel(
    const float* __restrict__ query,      // [num_seqs, num_heads, head_dim]
    const float* __restrict__ k_cache,    // [num_blocks, num_heads, block_size, head_dim]
    const float* __restrict__ v_cache,    // [num_blocks, num_heads, block_size, head_dim]
    float* __restrict__ output,           // [num_seqs, num_heads, head_dim]
    const int* __restrict__ block_table,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lengths,
    int num_heads,
    int block_size,
    int head_dim,
    int max_blocks_per_seq,
    float scale
) {
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int ctx_len = context_lengths[seq_id];

    // Shared memory layout:
    //   [0 .. head_dim-1]: query
    //   [head_dim .. head_dim + num_warps-1]: warp max values (for merge)
    //   [head_dim + num_warps .. head_dim + 2*num_warps-1]: warp sum values
    //   [head_dim + 2*num_warps .. head_dim + 2*num_warps + num_warps*head_dim-1]: warp output accumulators
    extern __shared__ float smem[];
    float* q_shared = smem;  // [head_dim]

    // Load query into shared memory
    const float* q_ptr = query + seq_id * (num_heads * head_dim) + head_id * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = q_ptr[d];
    }
    __syncthreads();

    // Each thread handles head_dim/32 output dimensions (within its warp)
    // For head_dim=128: each thread handles 4 dimensions
    // For head_dim=64: each thread handles 2 dimensions
    const int DIMS_PER_THREAD = 8;  // max, supports head_dim up to 256
    float out_acc[DIMS_PER_THREAD];
    int num_d = 0;
    for (int d = lane_id; d < head_dim; d += 32) {
        out_acc[num_d] = 0.0f;
        num_d++;
    }

    float warp_max = -1e20f;
    float warp_sum = 0.0f;

    // Flatten all tokens into a linear index and distribute across warps
    // Warp w processes tokens: w, w+num_warps, w+2*num_warps, ...
    int total_tokens = ctx_len;

    for (int token_idx = warp_id; token_idx < total_tokens; token_idx += num_warps) {
        // Map token_idx to physical block location
        int logical_block = token_idx / block_size;
        int offset_in_block = token_idx % block_size;
        int physical_block = block_table[seq_id * max_blocks_per_seq + logical_block];
        int block_base = physical_block * (num_heads * block_size * head_dim)
                       + head_id * (block_size * head_dim);

        // Compute dot(q, k[token]) within this warp
        // Each lane handles head_dim/32 dimensions
        const float* k_ptr = k_cache + block_base + offset_in_block * head_dim;
        float partial_dot = 0.0f;
        for (int d = lane_id; d < head_dim; d += 32) {
            partial_dot += q_shared[d] * k_ptr[d];
        }
        // Warp-level reduction (no __syncthreads needed!)
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial_dot += __shfl_down_sync(0xffffffff, partial_dot, offset);
        }
        // Lane 0 has the full dot product; broadcast to all lanes
        float score = __shfl_sync(0xffffffff, partial_dot, 0) * scale;

        // Online softmax update
        float new_max = fmaxf(warp_max, score);
        float correction = expf(warp_max - new_max);
        float exp_score = expf(score - new_max);
        warp_sum = warp_sum * correction + exp_score;
        warp_max = new_max;

        // Update output accumulators with V
        const float* v_ptr = v_cache + block_base + offset_in_block * head_dim;
        int di = 0;
        for (int d = lane_id; d < head_dim; d += 32) {
            out_acc[di] = out_acc[di] * correction + exp_score * v_ptr[d];
            di++;
        }
    }

    // === Merge warp results ===
    // Each warp has: (warp_max, warp_sum, out_acc[head_dim/32 per lane])
    // We need to combine them using the online softmax merge formula.

    // Store per-warp max and sum to shared memory
    float* warp_maxes = smem + head_dim;                    // [num_warps]
    float* warp_sums = smem + head_dim + num_warps;         // [num_warps]
    float* warp_outputs = smem + head_dim + 2 * num_warps;  // [num_warps * head_dim]

    // Each lane writes its output dimensions to shared memory
    int di = 0;
    for (int d = lane_id; d < head_dim; d += 32) {
        warp_outputs[warp_id * head_dim + d] = out_acc[di];
        di++;
    }
    if (lane_id == 0) {
        warp_maxes[warp_id] = warp_max;
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Now merge: use all threads to compute final output
    // First find global max
    float global_max = -1e20f;
    for (int w = 0; w < num_warps; w++) {
        global_max = fmaxf(global_max, warp_maxes[w]);
    }

    // Compute rescaled sum and output
    float* out_ptr = output + seq_id * (num_heads * head_dim) + head_id * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        float total_sum = 0.0f;
        for (int w = 0; w < num_warps; w++) {
            float correction = expf(warp_maxes[w] - global_max);
            acc += warp_outputs[w * head_dim + d] * correction;
            // Only count sum once per dimension (use first dim thread)
            if (d == tid)  // always true for first iteration
                total_sum += warp_sums[w] * correction;
        }
        // We need total_sum computed once; do it separately
        out_ptr[d] = acc;  // will divide below
    }
    __syncthreads();

    // Compute total_sum once
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int w = 0; w < num_warps; w++) {
            total_sum += warp_sums[w] * expf(warp_maxes[w] - global_max);
        }
        smem[0] = total_sum;  // store in smem[0]
    }
    __syncthreads();
    float total_sum = smem[0];

    // Final normalization
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_ptr[d] /= total_sum;
    }
}

// C interface for V2 fused paged attention
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
) {
    dim3 grid(num_seqs, num_heads);
    int threads = 256;
    int num_warps = threads / 32;
    // Shared memory: query[head_dim] + warp_maxes[num_warps] + warp_sums[num_warps]
    //              + warp_outputs[num_warps * head_dim]
    int shared_mem_size = (head_dim + 2 * num_warps + num_warps * head_dim) * sizeof(float);
    fused_paged_attention_kernel<<<grid, threads, shared_mem_size>>>(
        query, k_cache, v_cache, output, block_table, context_lengths,
        num_heads, block_size, head_dim, max_blocks_per_seq, scale
    );
}

// C interface for the gather kernel launch
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
) {
    dim3 grid(num_seqs, num_heads);
    int threads = 256;
    paged_gather_kernel<<<grid, threads>>>(
        kv_cache, output, block_table, context_lengths,
        num_heads, block_size, head_dim, max_blocks_per_seq, max_ctx_len
    );
}
