__global__ void gemm(half *A, half *B, float *C) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  int blockRow = blockIdx.y * 16;
  int blockCol = blockIdx.x * 16;

  if (blockRow < 32 && blockCol < 1024) {

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < 128; k += 16) {

      wmma::load_matrix_sync(a_frag, A + blockRow * 128 + k, 128);
      wmma::load_matrix_sync(b_frag, B + k * 1024 + blockCol, 1024);

      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + blockRow * 1024 + blockCol, c_frag, 1024,
                            wmma::mem_row_major);
  }
}

extern "C" void gemm_kernel(half *A, half *B, float *C, int m, int k, int n) {
  half *d_A;
  half *d_B;
  float *d_C;

  cudaMalloc(&d_A, m * k * sizeof(half));
  cudaMalloc(&d_B, k * n * sizeof(half));
  cudaMalloc(&d_C, m * n * sizeof(float));

  cudaMemcpy(d_A, A, m * k * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, k * n * sizeof(half), cudaMemcpyHostToDevice);

  dim3 blockSize(32);
  dim3 numBlocks((n + 16 - 1) / 16, (m + 16 - 1) / 16);
  gemm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
