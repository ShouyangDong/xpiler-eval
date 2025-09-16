// =============================================================================
// 2. Shape: [1, 3, 224, 224] → Total: 150,528 elements (Image input)
// =============================================================================
__global__ void __launch_bounds__(960)
sub_1x3x224x224(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 150528) {
        C[idx] = A[idx] - B[idx];
    }
}

extern "C" void sub_kernel_1x3x224x224(float *h_A, float *h_B, float *h_C) {
    float *d_A, *d_B, *d_C;
    const int total = 1 * 3 * 224 * 224;

    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_B, total * sizeof(float));
    cudaMalloc(&d_C, total * sizeof(float));

    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(960);
    dim3 numBlocks((total + 959) / 960);

    sub_1x3x224x224<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
