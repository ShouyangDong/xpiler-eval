

#define BLOCK_M 16
#define BLOCK_N 64  // K-dim block
#define BLOCK_K 32  // N-dim block (key seq)

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset, 32);
    return val;
}

__global__ void fused_gqa(
    const float* __restrict__ Q,  // [B, H, M, K]
    const float* __restrict__ K,  // [B, H, K, N]
    const float* __restrict__ V,  // [B, H, N, K]
    float* __restrict__ O,        // [B, H, M, K]
    int B, int H, int M, int N, int K_dim
) {
    int bid = blockIdx.z;  // batch * heads
    int head = bid % H;
    int b = bid / H;

    int tx = threadIdx.x;  // 0~63 (BLOCK_N)
    int ty = threadIdx.y;  // 0~15 (BLOCK_M)
    int bx = blockIdx.x;   // M block
    int by = blockIdx.y;   // N block

    int m_start = bx * BLOCK_M + ty;
    int n_start = by * BLOCK_K;

    // Shared memory
    __shared__ float q_smem[BLOCK_M][BLOCK_N];  // Q: [BLOCK_M, K]
    __shared__ float k_smem[BLOCK_N][BLOCK_K];  // K: [K, BLOCK_K]
    __shared__ float v_smem[BLOCK_K][BLOCK_N];  // V: [BLOCK_K, K_dim]

    // Output register for O[m, :]
    float acc[4] = {0};  // up to 4 warps per block, each handles 1 row

    if (ty < 4) acc[ty] = 0.0f;  // initialize

    // Online Softmax state
    float max_val = -INFINITY;
    float exp_sum = 0.0f;

    // Loop over K_dim in tiles
    for (int k = 0; k < K_dim; k += BLOCK_N) {
        // Load Q: [BLOCK_M, BLOCK_N]
        if (m_start < M && k + tx < K_dim) {
            q_smem[ty][tx] = Q[b * H * M * K_dim + head * M * K_dim + m_start * K_dim + k + tx];
        } else {
            q_smem[ty][tx] = 0.0f;
        }

        // Load K: [BLOCK_N, BLOCK_K]
        if (k + ty < K_dim && n_start + tx < N) {
            k_smem[ty][tx] = K[b * H * K_dim * N + head * K_dim * N + (k + ty) * N + n_start + tx];
        } else {
            k_smem[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute S[m, n:n+BLOCK_K] = Q[m, k:k+BLOCK_N] @ K[k:k+BLOCK_N, n:n+BLOCK_K]
        float s_val = 0.0f;
        for (int i = 0; i < BLOCK_N; ++i) {
            s_val += q_smem[ty][i] * k_smem[i][tx];
        }

        // Load V: [BLOCK_K, K_dim]
        if (n_start + ty < N && tx < K_dim) {
            v_smem[ty][tx] = V[b * H * N * K_dim + head * N * K_dim + (n_start + ty) * K_dim + tx];
        } else {
            v_smem[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Online Softmax: update max and exp_sum
        float old_max = max_val;
        max_val = fmaxf(max_val, s_val);
        if (old_max != -INFINITY) {
            exp_sum = exp_sum * expf(old_max - max_val);
        }
        float exp_s = expf(s_val - max_val);
        exp_sum += exp_s;

        // Accumulate: O[m] += exp(S) * V
        // Each thread handles one row (ty), accumulate over V columns
        if (ty < 4 && m_start < M) {
            for (int i = 0; i < BLOCK_K && (n_start + i) < N; ++i) {
                acc[ty] += exp_s * v_smem[i][tx];  // only if tx < K_dim
            }
        }

        __syncthreads();
    }

    // Finalize softmax: O[m] /= exp_sum
    if (ty < 4 && m_start < M && exp_sum > 0) {
        for (int i = 0; i < BLOCK_N && i < K_dim; ++i) {
            if (tx == i) {
                acc[ty] /= exp_sum;
                int out_idx = b * H * M * K_dim + head * M * K_dim + m_start * K_dim + i;
                atomicAdd(&O[out_idx], acc[ty]);
            }
        }
    }
}

extern "C" void gqa_kernel(float* Q, float* K, float* V, float* O,
                           int batch, int heads, int M, int N, int K_dim) {
    float *d_Q, *d_K, *d_V, *d_O;
    size_t q_sz = batch * heads * M * K_dim * sizeof(float);
    size_t k_sz = batch * heads * K_dim * N * sizeof(float);
    size_t v_sz = batch * heads * N * K_dim * sizeof(float);
    size_t o_sz = batch * heads * M * K_dim * sizeof(float);

    hipMalloc(&d_Q, q_sz);
    hipMalloc(&d_K, k_sz);
    hipMalloc(&d_V, v_sz);
    hipMalloc(&d_O, o_sz);

    hipMemcpy(d_Q, Q, q_sz, hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, k_sz, hipMemcpyHostToDevice);
    hipMemcpy(d_V, V, v_sz, hipMemcpyHostToDevice);
    hipMemset(d_O, 0, o_sz);  // zero out

    // Grid: M_blocks x N_blocks x (batch * heads)
    dim3 grid(
        (M + BLOCK_M - 1) / BLOCK_M,
        (N + BLOCK_K - 1) / BLOCK_K,
        batch * heads
    );
    dim3 block(64, 16);  // tx=64, ty=16

    fused_gqa<<<grid, block>>>(
        d_Q, d_K, d_V, d_O,
        batch, heads, M, N, K_dim
    );

    hipMemcpy(O, d_O, o_sz, hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_O);
}