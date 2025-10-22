#define WARP_SIZE 32

#define M 16
#define N 16
#define K 16

#define M_TOTAL 32
#define N_TOTAL 512
#define K_TOTAL 512

__global__ void dense(half *A, half *B, float *C, float *D) {
  int ix = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int iy = (blockIdx.y * blockDim.y + threadIdx.y);

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> ab_frag;

  wmma::fill_fragment(ab_frag, 0.0f);

  int a_row = ix * M;
  int b_row = iy * N;
  for (int k = 0; k < K_TOTAL; k += K) {
    if (a_row < M_TOTAL && k < K_TOTAL && b_row < K_TOTAL && k < N_TOTAL) {

      wmma::load_matrix_sync(a_frag, A + k + a_row * M_TOTAL, M_TOTAL);
      wmma::load_matrix_sync(b_frag, B + k + k * K_TOTAL, K_TOTAL);

      wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
    }
  }

  for (int i = 0; i < ab_frag.num_elements; i++) {
    int row_in_tile = i / N;
    int col_in_tile = i % N;
    int global_col = iy * N + col_in_tile;

    if (global_col < N_TOTAL) {
      ab_frag.x[i] += C[global_col];
    }
  }
  wmma::store_matrix_sync(D + a_row * N_TOTAL + b_row, ab_frag, N_TOTAL,
                          wmma::mem_row_major);
}

extern "C" void dense_kernel(half *A, half *B, float *C, float *D, int m, int k,
                             int n) {
  half *d_A;
  half *d_B;
  float *d_C;
  float *d_D;

  cudaMalloc(&d_A, m * k * sizeof(half));
  cudaMalloc(&d_B, k * n * sizeof(half));
  cudaMalloc(&d_C, n * sizeof(float));
  cudaMalloc(&d_D, m * n * sizeof(float));

  cudaMemcpy(d_A, A, m * k * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, k * n * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 gridDim, blockDim;
  blockDim.x = 4 * WARP_SIZE;
  blockDim.y = 4;

  gridDim.x =
      (m + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
  gridDim.y = (n + N * blockDim.y - 1) / (N * blockDim.y);
  dense<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_D);

  cudaMemcpy(D, d_D, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);
}
