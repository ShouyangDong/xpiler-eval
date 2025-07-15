// Forward declaration of the device function
__device__ float geluf(float x);

__device__ float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

__global__ void __launch_bounds__(60)
    gelu(float *__restrict__ A, float *__restrict__ compute) {
  compute[((int)threadIdx.x)] = geluf(A[((int)threadIdx.x)]);
}

extern "C" void gelu_kernel(float *A, float *C, int size) {
  float *d_A;
  float *d_C;

  cudaMalloc(&d_A, size * sizeof(float));
  cudaMalloc(&d_C, size * sizeof(float));

  cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(60);
  dim3 numBlocks((size + 60 - 1) / 60);

  gelu<<<numBlocks, blockSize>>>(d_A, d_C);


  cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
