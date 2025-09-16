// =============================================================================
// 5. Shape: [1, 512] → Total: 512 elements (embedding or small vector)
// =============================================================================
__global__ void __launch_bounds__(960)
sub_1x512(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C) {
    int idx = threadIdx.x;
    if (idx < 512) {
        C[idx] = A[idx] - B[idx];
    }
}

extern "C" void sub_kernel_1x512(float *h_A, float *h_B, float *h_C) {
    float *d_A, *d_B, *d_C;
    const int total = 512;

    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_B, total * sizeof(float));
    cudaMalloc(&d_C, total * sizeof(float));

    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(960);
    dim3 numBlocks(1);

    sub_1x512<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
