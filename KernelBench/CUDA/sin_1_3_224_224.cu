

__global__ void __launch_bounds__(1024)
    sin(float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 150528) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel(float *h_A, float *h_C, int n, int h, int w, int c) {
  float *d_A, *d_C;
  const int total = n * w * h * c;

  cudaMalloc(&d_A, total * sizeof(float));
  cudaMalloc(&d_C, total * sizeof(float));

  cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(294);
  dim3 numBlocks((total + 293) / 294);

  sin<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}

__global__ void __launch_bounds__(294)
    sin_32x64(float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 2048) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel_32x64(float *h_A, float *h_C) {
  float *d_A, *d_C;
  const int total = 32 * 64;

  cudaMalloc(&d_A, total * sizeof(float));
  cudaMalloc(&d_C, total * sizeof(float));

  cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(294);
  dim3 numBlocks((total + 293) / 294);

  sin_32x64<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}

__global__ void __launch_bounds__(294)
    sin_8x16x32x32(float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 131072) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel_8x16x32x32(float *h_A, float *h_C) {
  float *d_A, *d_C;
  const int total = 8 * 16 * 32 * 32;

  cudaMalloc(&d_A, total * sizeof(float));
  cudaMalloc(&d_C, total * sizeof(float));

  cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(294);
  dim3 numBlocks((total + 293) / 294);

  sin_8x16x32x32<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}

__global__ void __launch_bounds__(294)
    sin_1x512(float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = threadIdx.x;
  if (idx < 512) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel_1x512(float *h_A, float *h_C) {
  float *d_A, *d_C;
  const int total = 512;

  cudaMalloc(&d_A, total * sizeof(float));
  cudaMalloc(&d_C, total * sizeof(float));

  cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(294);
  dim3 numBlocks(1);

  sin_1x512<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}

__global__ void __launch_bounds__(294)
    sin_64x64x64(float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 262144) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel_64x64x64(float *h_A, float *h_C) {
  float *d_A, *d_C;
  const int total = 64 * 64 * 64;

  cudaMalloc(&d_A, total * sizeof(float));
  cudaMalloc(&d_C, total * sizeof(float));

  cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(294);
  dim3 numBlocks((total + 293) / 294);

  sin_64x64x64<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}

__global__ void __launch_bounds__(294)
    sin_2x1x1024(float *__restrict__ A, float *__restrict__ T_sin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 2048) {
    T_sin[idx] = sinf(A[idx]);
  }
}

extern "C" void sin_kernel_2x1x1024(float *h_A, float *h_C) {
  float *d_A, *d_C;
  const int total = 2 * 1 * 1024;

  cudaMalloc(&d_A, total * sizeof(float));
  cudaMalloc(&d_C, total * sizeof(float));

  cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(294);
  dim3 numBlocks((total + 293) / 294);

  sin_2x1x1024<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}

__global__ void __launch_bounds__(294)
    sin_scalar(float *__restrict__ A, float *__restrict__ T_sin) {
  if (threadIdx.x == 0) {
    T_sin[0] = sinf(A[0]);
  }
}

extern "C" void sin_kernel_scalar(float *h_A, float *h_C) {
  float *d_A, *d_C;
  const int total = 1;

  cudaMalloc(&d_A, sizeof(float));
  cudaMalloc(&d_C, sizeof(float));

  cudaMemcpy(d_A, h_A, sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(294);
  dim3 numBlocks(1);

  sin_scalar<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}