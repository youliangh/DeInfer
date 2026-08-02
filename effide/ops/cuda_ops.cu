// DISABLE FLAGS PASSED BY TORCH.EXTENSION
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif // __CUDA_NO_HALF_OPERATORS__

#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif // __CUDA_NO_HALF_CONVERSIONS__

#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif // __CUDA_NO_BFLOAT16_CONVERSIONS__

#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif // __CUDA_NO_HALF2_OPERATORS__

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <wmma_extension/utils.hpp>
#include <wmma_extension/wmma_extension.hpp>
#include <wmma_extension/wmma_mma.hpp>

#include <torch/extension.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"


/////////////customized_rotary_embedding_kernel/////////////

namespace impl {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void general_apply_rotary_embedding(
    scalar_t* __restrict__ token, const int n,
    const scalar_t* cache_ptr, const int head_size, const int num_heads,
    const int embed_dim, const int rot_dim,
    const int token_idx, const int64_t stride
) {
    const scalar_t* cos_ptr = cache_ptr;
    const scalar_t* sin_ptr = cache_ptr + embed_dim;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int head_idx = i / embed_dim;
        const int64_t token_head = token_idx * stride + head_idx * head_size;
        const int rot_offset = i % embed_dim;
        apply_token_rotary_embedding<scalar_t, IS_NEOX>(
            token + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
      }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void customized_rotary_embedding_kernel(
    const int64_t* __restrict__ positions,          // [batch_size]
    scalar_t* __restrict__ query,                   // [batch_size, num_heads, head_size]
    const int64_t* __restrict__ positions_tables,   // [batch_size, max_blocks_per_seq * block_size]
    scalar_t* __restrict__ key_buffer,              // [num_blocks, block_size, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,     // [max_position, 2, rot_dim // 2]
    const int num_new_tokens, const int num_key_tokens,
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  int token_idx = blockIdx.x;

  // Dispatch
  int64_t pos;
  scalar_t* token_ptr;
  int n;
  int token_num_head;
  int64_t stride;
  const int embed_dim = rot_dim / 2;

  if (num_new_tokens - token_idx >= 1) {
    pos = positions[token_idx];
    token_ptr = query;
    n = num_heads * embed_dim;
    token_num_head = num_heads;
    stride = query_stride;
  } else {
    token_idx -= num_new_tokens;
    pos = positions_tables[token_idx];
    token_ptr = key_buffer;
    n = num_kv_heads * embed_dim;
    token_num_head = num_kv_heads;
    stride = key_stride;
  }
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  general_apply_rotary_embedding<scalar_t, IS_NEOX>(
    token_ptr, n, cache_ptr, head_size, token_num_head,
    embed_dim, rot_dim,
    token_idx, stride);
}

template <typename scalar_t, bool IS_NEOX, int BLOCK_SIZE=16>
__global__ void key_rotary_embedding_kernel(
    const int* __restrict__ remapping_tables,       // [batch_size, max_blocks_per_seq]
    const int* __restrict__ seq_lens_ptr,           // [batch_size]
    scalar_t* __restrict__ key_buffer,              // [num_blocks, block_size, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,     // [max_position, 2, rot_dim // 2]
    const int max_blocks_per_seq,
    const int rot_dim, const int key_stride,
    const int num_kv_heads, const int head_size) {
    // Each thread block is responsible for one sequence.
    int seq_idx = blockIdx.x;
    int worker_idx = blockIdx.y;

    const int embed_dim = rot_dim / 2;
    const int key_size = num_kv_heads * head_size;;

    int seq_len = seq_lens_ptr[seq_idx];

    const int* remapping_tables_base = remapping_tables + seq_idx * max_blocks_per_seq;
    for (int pos_start = 0; pos_start < seq_len; pos_start += gridDim.y) {
        int pos = pos_start + worker_idx;
        if (pos >= seq_len)
            return;

        const int block_idx = remapping_tables_base[pos / BLOCK_SIZE];
        const int token_idx = pos % BLOCK_SIZE;

        scalar_t* token_ptr = key_buffer + block_idx * key_stride + token_idx * key_size;
        const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

        const scalar_t* cos_ptr = cache_ptr;
        const scalar_t* sin_ptr = cache_ptr + embed_dim;

        for (int i = threadIdx.x; i < num_kv_heads * embed_dim; i += blockDim.x) {
            const int head_idx = i / embed_dim;
            const int token_head = head_idx * head_size;
            const int rot_offset = i % embed_dim;
            apply_token_rotary_embedding<scalar_t, IS_NEOX>(
                token_ptr + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim
            );
        }

    }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void query_rotary_embedding_kernel(
    const int64_t* __restrict__ positions,          // [batch_size]
    scalar_t* __restrict__ query,                   // [batch_size, num_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,     // [max_position, 2, rot_dim // 2]
    const int rot_dim, const int query_stride,
    const int num_heads, const int head_size) {
    // Each thread block is responsible for one query token and one sequence.
    int token_idx = blockIdx.x;
    const int embed_dim = rot_dim / 2;

    general_apply_rotary_embedding<scalar_t, IS_NEOX>(
        query, num_heads * embed_dim, cos_sin_cache + positions[token_idx] * rot_dim,
        head_size, num_heads,
        embed_dim, rot_dim, token_idx, query_stride
    );
}

}  // namespace impl

void customized_rotary_embedding(
    torch::Tensor& positions,           // [batch_size]
    torch::Tensor& query,               // [batch_size, num_heads, head_size]
    torch::Tensor& remapping_tables,    // [batch_size, max_blocks_per_seq]
    torch::Tensor& seq_lens_tensor,     // [batch_size]
    torch::Tensor& key_buffer,          // [num_block, block_size, num_kv_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,       // [max_position, rot_dim]
    bool is_neox
) {
    int batch_size = positions.size(0);

    // Make sure the batch_size is consistent
    TORCH_CHECK(
        batch_size == query.size(0) &&
        batch_size == remapping_tables.size(0) &&
        batch_size == seq_lens_tensor.size(0),
        "Inconsistent batch size among positions, query, remapping_tables, and seq_lens_tensor"
    );

    TORCH_CHECK(
        query.dim() == 3,
        "The shape of query should be [batch_size, num_heads, head_size]"
    );

    TORCH_CHECK(
        key_buffer.dim() == 4,
        "The shape of key_buffer should be [num_block, block_size, num_kv_heads, head_size]"
    );

    // Make sure head_size is valid for query and key
    int query_size = query.size(2);
    int key_size = key_buffer.size(3);
    TORCH_CHECK(query_size % head_size == 0);
    TORCH_CHECK(key_size % head_size == 0);

    // Make sure query and key have consistent number of heads
    int num_heads = query.size(1);
    int num_kv_heads = key_buffer.size(2);
    TORCH_CHECK(num_heads % num_kv_heads == 0);

    int rot_dim = cos_sin_cache.size(1);
    int embed_dim = rot_dim / 2;
    int query_stride = query.stride(0);
    int key_stride = key_buffer.stride(0);

    const int worker_per_seq = 16;
    dim3 grid_query(batch_size);
    dim3 grid_key(batch_size, worker_per_seq);
    dim3 block_query(std::min<int>(num_heads * embed_dim, 512));
    dim3 block_key(std::min<int>(num_kv_heads * embed_dim, 512));
    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
        if (is_neox) {
            impl::query_rotary_embedding_kernel<scalar_t, true><<<grid_query, block_query, 0, stream>>>(
                positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                cos_sin_cache.data_ptr<scalar_t>(),
                rot_dim, query_stride, num_heads, head_size
            );

            impl::key_rotary_embedding_kernel<scalar_t, true><<<grid_key, block_key, 0, stream>>>(
                remapping_tables.data_ptr<int>(), seq_lens_tensor.data_ptr<int>(),
                key_buffer.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                remapping_tables.size(1),
                rot_dim, key_stride, num_kv_heads, head_size
            );
        } else {
            impl::query_rotary_embedding_kernel<scalar_t, false><<<grid_query, block_query, 0, stream>>>(
                positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                cos_sin_cache.data_ptr<scalar_t>(),
                rot_dim, query_stride, num_heads, head_size
            );

            impl::key_rotary_embedding_kernel<scalar_t, false><<<grid_key, block_key, 0, stream>>>(
                remapping_tables.data_ptr<int>(), seq_lens_tensor.data_ptr<int>(),
                key_buffer.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                remapping_tables.size(1),
                rot_dim, key_stride, num_kv_heads, head_size
            );
        }
    });
}

/////////////cache_copy/////////////

using namespace nvcuda;

constexpr int kWarpSize = 32;

template<typename T>
struct TorchTypeTrais { using cuda_type = T; };

template<> struct TorchTypeTrais<at::Half> { using cuda_type = half; };
template<> struct TorchTypeTrais<at::BFloat16> { using cuda_type = nv_bfloat16; };


template<typename scalar_t>
__device__ inline void
load_matrix_sync(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16, scalar_t, nvcuda::wmma::row_major> &frag,
                 const scalar_t *const ptr, const unsigned ldm,
                 const bool sync = true) {
  // length of leading dimension of the input fragment
  constexpr unsigned old_ldm = 16;
  mtk::wmma::mma::foreach<decltype(frag)>(
      [&](const unsigned *frag_index_list, const unsigned fragment_index_count,
          const unsigned mem_index) {
        const unsigned offset = (mem_index / old_ldm) * ldm + mem_index % old_ldm;
        for (unsigned i = 0; i < fragment_index_count; i++) {
          const unsigned frag_index = frag_index_list[i];
          frag.x[frag_index] = ptr[offset];
        }
      });
  if (sync)
    __syncwarp();
}

template<typename scalar_t>
__device__ inline void
store_matrie_sync(scalar_t *const ptr,
                  const mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16, scalar_t, nvcuda::wmma::row_major> &frag,
                  const unsigned ldm, const bool sync = true) {
  // length of leading dimension of the input fragment
  const unsigned old_ldm = 16;
  mtk::wmma::mma::foreach<decltype(frag)>(
      [&](const unsigned *frag_index_list, const unsigned fragment_index_count,
          const unsigned mem_index) {
        const unsigned offset = (mem_index / old_ldm) * ldm + mem_index % old_ldm;
        for (unsigned i = 0; i < fragment_index_count; i++) {
          const unsigned frag_index = frag_index_list[i];
          ptr[offset] = frag.x[frag_index];
        }
      });
  if (sync)
    __syncwarp();
}

template <typename scalar_t, int WRAPS_PER_BLOCK=4, int WMMA_M=16, int WMMA_N=8, int WMMA_K=16>
__global__ void paged_copy_kernel(
    scalar_t* __restrict__          cache_ptr,              // [all_blocks, block_size, kv_cache_size]
    scalar_t* __restrict__          output_ptr,             // [num_block, block_size, kv_cache_size]
    const int* __restrict__         block_table_ptr,        // [batch_size, max_blocks_per_seq (padded)]
    const int* __restrict__         block_per_seq_ptr,      // [batch_size] i.e., ceil(seq_lens_tensor / 16)
    const int* __restrict__         remapping_table_ptr,    // [batch_size, max_blocks_per_seq (padded)]
    const int kv_cache_size,
    const int cache_stride, const int batch_size,
    const int max_blocks_per_seq, const int block_size
) {
    int seq_idx = blockIdx.x;
    int num_blocks = block_per_seq_ptr[seq_idx];

    int tid = threadIdx.x;
    int wid = tid / kWarpSize;  // 0 ~ WRAPS_PER_BLOCK - 1

    const int block_table_offset = seq_idx * max_blocks_per_seq;
    const int* block_table_base = block_table_ptr + block_table_offset;
    const int* remapping_table_base = remapping_table_ptr + block_table_offset;
    const int chunk_size = kv_cache_size / WRAPS_PER_BLOCK;
    const int split_k_iters = chunk_size / WMMA_K;

    // Ignore WMMA_N, just used for template instantiate
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, scalar_t, nvcuda::wmma::row_major> frag_a;
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, scalar_t, nvcuda::wmma::row_major> frag_b;

    for (int i = 0; i < num_blocks; ++i) {
        const scalar_t* src_ptr = cache_ptr + block_table_base[i] * cache_stride;
        scalar_t* dst_ptr = output_ptr + remapping_table_base[i] * cache_stride;

        const int start_idx = wid * chunk_size;
        for (int j = 0; j < split_k_iters; j += 2) {
            load_matrix_sync<scalar_t>(frag_a, src_ptr + start_idx + j * WMMA_K, kv_cache_size, false /* sync = false*/);
            load_matrix_sync<scalar_t>(frag_b, src_ptr + start_idx + (j + 1) * WMMA_K, kv_cache_size, true /* sync = true*/);
            store_matrie_sync<scalar_t>(dst_ptr + start_idx + j * WMMA_K, frag_a, kv_cache_size, false /* sync = false*/);
            store_matrie_sync<scalar_t>(dst_ptr + start_idx + (j + 1) * WMMA_K, frag_b, kv_cache_size, true /* sync = true*/);
        }
    }
}


void paged_copy(
    torch::Tensor& paged_cache,         // [all_blocks, block_size, cache_size]
    torch::Tensor& block_table,         // [batch_size, max_blocks_per_seq (padded)]
    torch::Tensor& block_per_seq,       // [batch_size] i.e., ceil(seq_lens_tensor / 16)
    torch::Tensor& remapping_table,     // [batch_size, max_blocks_per_seq (padded)]
    torch::Tensor& output               // [num_block, block_size, cache_size]
) {
    int batch_size = block_table.size(0);

    TORCH_CHECK(
        paged_cache.dim() == 3,
        "positions must have shape [all_blocks, block_size, cache_size]"
    );

    TORCH_CHECK(
        output.dim() == 3,
        "output must have shape [num_block, block_size, cache_size]"
    );

    TORCH_CHECK(
        paged_cache.size(2) == output.size(2),
        "Inconsistent input dimensions between paged_cache and weight"
    );

    TORCH_CHECK(
        paged_cache.size(1) == output.size(1),
        "Inconsistent block size between paged_cache and output"
    );

    TORCH_CHECK(
        block_table.size(1) == remapping_table.size(1),
        "Inconsistent block size between paged_cache and output"
    );

    TORCH_CHECK(
        batch_size == remapping_table.size(0) &&
        batch_size == block_per_seq.size(0),
        "Inconsistent batch_size between block_table, block_per_seq, and remapping_table"
    );

    int max_blocks_per_seq = block_table.size(1);

    int block_size = paged_cache.size(1);
    int cache_stride = paged_cache.size(1) * paged_cache.size(2); // block_size * kv_cache_size

    const int WRAPS_PER_BLOCK = 4;

    dim3 grid(batch_size);
    dim3 block(kWarpSize * WRAPS_PER_BLOCK);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(paged_cache));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VLLM_DISPATCH_HALF_PRECISION_FLOATING_TYPES(paged_cache.scalar_type(), "paged_copy", [&] {
        using cuda_type = TorchTypeTrais<scalar_t>::cuda_type;
        paged_copy_kernel<cuda_type, WRAPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            reinterpret_cast<cuda_type*>(paged_cache.data_ptr()),
            reinterpret_cast<cuda_type*>(output.data_ptr()),
            block_table.data_ptr<int>(),
            block_per_seq.data_ptr<int>(),
            remapping_table.data_ptr<int>(),
            paged_cache.size(2),  // cache size
            cache_stride, batch_size, max_blocks_per_seq,
            block_size
        );
    });
}

template<typename scalar_t>
__global__ void cache_copy_kernel(scalar_t* key_dst, scalar_t* key_src,
                                  scalar_t* val_dst, scalar_t* val_src,
                                  int* size, int bytes_per_block) {
    int tid = threadIdx.x;
    int num_bytes = (*size) * bytes_per_block;
    if (tid == 0)
        cudaMemcpyAsync(key_dst, key_src, num_bytes, cudaMemcpyDeviceToDevice);
    if (tid == 1)
        cudaMemcpyAsync(val_dst, val_src, num_bytes, cudaMemcpyDeviceToDevice);

    __syncthreads();
}

void cache_copy(
    torch::Tensor& paged_cache,         // [2, all_blocks, block_size, cache_size]
    torch::Tensor& output,              // [2, num_block, block_size, cache_size])
    torch::Tensor& contiguous_blocks    // [1]
) {
    TORCH_CHECK(
        paged_cache.dim() == 4,
        "paged_cache must have shape [2, all_blocks, block_size, cache_size]"
    );

    TORCH_CHECK(
        output.dim() == 4,
        "output must have shape [2, num_block, block_size, cache_size]"
    );

    TORCH_CHECK(
        paged_cache.size(3) == output.size(3),
        "Inconsistent input dimensions between paged_cache and weight"
    );

    TORCH_CHECK(
        paged_cache.size(2) == output.size(2),
        "Inconsistent block size between paged_cache and output"
    );

    const at::cuda::OptionalCUDAGuard device_guard(device_of(paged_cache));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VLLM_DISPATCH_HALF_PRECISION_FLOATING_TYPES(paged_cache.scalar_type(), "cache_copy", [&] {
        scalar_t* dst = reinterpret_cast<scalar_t*>(output.data_ptr());
        scalar_t* src = reinterpret_cast<scalar_t*>(paged_cache.data_ptr());
        int bytes_per_block = paged_cache.size(2) * paged_cache.size(3) * sizeof(scalar_t);

        cache_copy_kernel<scalar_t><<<1, 2, 0, stream>>>(
            dst, src,
            dst + output.stride(0), src + paged_cache.stride(0),
            contiguous_blocks.data_ptr<int>(), bytes_per_block
        );
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("paged_apply_rotary_embeds", &customized_rotary_embedding);
    m.def("paged_copy", &paged_copy);
    m.def("cache_copy", &cache_copy);
}