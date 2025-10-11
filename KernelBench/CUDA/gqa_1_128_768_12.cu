#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define H_Q 2
#define H_KV 1
#define M 256
#define N 4096
#define D 64

__device__ __forceinline__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void gqa_forward_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
          half* __restrict__ O,
          half* __restrict__ workspace
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int head = blockIdx.z;

    if (head >= H_Q) return;

    const int kv_head = (head < H_KV) ? head : (H_KV - 1);
    const half* k_base = K + kv_head * N * D;
    const half* v_base = V + kv_head * N * D;
    const half* q_base = Q + head * M * D;
          half* o_base = O + head * M * D;

    const int tile_m = bid / (N / WMMA_N);
    const int tile_n = bid % (N / WMMA_N);

    if (tile_m >= M / WMMA_M) return;

    const int m_start = tile_m * WMMA_M;
    const int n_start = tile_n * WMMA_N;

    half* s_base = workspace + head * M * N;
    half* s_ptr = s_base + m_start * N + n_start;

    const half* q_ptr = q_base + m_start * D + (tid / WMMA_K) * 16 + (tid % WMMA_K);
    const half* k_ptr = k_base + n_start * D + (tid / WMMA_K) * 16 + (tid % WMMA_K);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> q_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> s_frag;

    wmma::fill_fragment(s_frag, __float2half_rn(0.0f));
    wmma::load_matrix_sync(q_frag, q_ptr, D);
    wmma::load_matrix_sync(k_frag, k_ptr, D);
    wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
    wmma::store_matrix_sync(s_ptr, s_frag, N, wmma::mem_col_major);
}


extern "C" void gqa_forward(
    half* Q,  // [2, 256, 64]
    half* K,  // [1, 4096, 64]
    half* V,  // [1, 4096, 64]
    half* out // [2, 256, 64]
) {
    const int bytes_per_half = sizeof(half);

    // Step 1: 广播 K 和 V 到 2 个 head
    half *d_K_exp, *d_V_exp;
    cudaMalloc(&d_K_exp, H_Q * SEQ_KV * HEAD_DIM * bytes_per_half);
    cudaMalloc(&d_V_exp, H_Q * SEQ_KV * HEAD_DIM * bytes_per_half);

    // 复制两次（H_KV=1 -> H_Q=2）
    for (int i = 0; i < H_Q; i++) {
        cudaMemcpy(d_K_exp + i * SEQ_KV * HEAD_DIM, K, SEQ_KV * HEAD_DIM * bytes_per_half, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_V_exp + i * SEQ_KV * HEAD_DIM, V, SEQ_KV * HEAD_DIM * bytes_per_half, cudaMemcpyDeviceToDevice);
    }

    // 分配注意力分数 S: [2, 256, 4096]
    half *d_S;
    cudaMalloc(&d_S, H_Q * SEQ_Q * SEQ_KV * bytes_per_half);

    dim3 grid_QK(SEQ_Q / WMMA_M, SEQ_KV / WMMA_N, H_Q);
    dim3 block(16, 16);
    wmma_gemm<<<grid_QK, block>>>(Q, d_K_exp, d_S, SEQ_Q, SEQ_KV, HEAD_DIM);

    // Softmax: S -> softmax(S)
    dim3 grid_softmax(SEQ_Q * H_Q);  // 每行一个 block
    dim3 block_softmax(256, 16);     // 256 threads per row, 16 for reduce
    softmax_kernel<<<grid_softmax, block_softmax, 16 * sizeof(float)>>>(d_S, d_S, H_Q * SEQ_Q, SEQ_KV);

    // Step 3: O = S @ V, O: [2, 256, 64]
    // S: [2, 256, 4096], V: [2, 4096, 64] -> O: [2, 256, 64]
    wmma_gemm2<<<dim3(SEQ_Q / WMMA_M, HEAD_DIM / WMMA_N, H_Q), block>>>(d_S, d_V_exp, out, SEQ_Q, HEAD_DIM, SEQ_KV);

    // Cleanup
    cudaFree(d_K_exp);
    cudaFree(d_V_exp);
    cudaFree(d_S);
}
