

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void gatemlp(const half *X, const half *A, const half *B, float *O) {

  int blockRow = blockIdx.y * 16;
  int blockCol = blockIdx.x * 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag_A;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag_B;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o1_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o2_frag;

  wmma::fill_fragment(o1_frag, 0.0f);
  wmma::fill_fragment(o2_frag, 0.0f);

  for (int k = 0; k < 4096; k += 16) {
    const half *X_tile = X + blockRow * 4096 + k;
    const half *A_tile = A + k * 4096 + blockCol;
    const half *B_tile = B + k * 4096 + blockCol;

    wmma::load_matrix_sync(a_frag, X_tile, 4096);
    wmma::load_matrix_sync(b_frag_A, A_tile, 4096);
    wmma::load_matrix_sync(b_frag_B, B_tile, 4096);

    wmma::mma_sync(o1_frag, a_frag, b_frag_A, o1_frag);
    wmma::mma_sync(o2_frag, a_frag, b_frag_B, o2_frag);
  }

  for (int i = 0; i < o1_frag.num_elements; ++i) {
    float s = silu(o1_frag.x[i]);
    o1_frag.x[i] = s * o2_frag.x[i];
  }

  if (blockRow < 16 && blockCol < 4096) {
    wmma::store_matrix_sync(O + blockRow * 4096 + blockCol, o1_frag, 4096,
                            wmma::mem_row_major);
  }
}

extern "C" void gatemlp_kernel(const half *h_X, const half *h_A,
                               const half *h_B, float *h_O, int batch, int K,
                               int N) {
  // sizes
  size_t sizeX = (size_t)batch * (size_t)K * sizeof(half);
  size_t sizeA = (size_t)K * (size_t)N * sizeof(half);
  size_t sizeB = sizeA;
  size_t sizeO = (size_t)batch * (size_t)N * sizeof(float);

  half *d_X;
  half *d_A;
  half *d_B;
  float *d_O;

  cudaMalloc((void **)&d_X, sizeX);
  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_O, sizeO);

  cudaMemcpy(d_X, h_X, sizeX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  dim3 block(32, 1, 1);
  int grid_x = (N + 16 - 1) / 16;
  int grid_y = (batch + 16 - 1) / 16;
  dim3 grid(grid_x, grid_y, 1);

  gatemlp<<<grid, block>>>(d_X, d_A, d_B, d_O);

  cudaMemcpy(h_O, d_O, sizeO, cudaMemcpyDeviceToHost);

  // free
  cudaFree(d_X);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_O);
}
