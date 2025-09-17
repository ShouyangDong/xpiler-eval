// =============================================================================
// 1. Shape: [7, 1, 6, 7] → Total: 294 elements (matches original)
// =============================================================================
__global__ void __launch_bounds__(294)
sin_7x1x6x7(float *__restrict__ A, float *__restrict__ T_sin) {
    int idx = threadIdx.x;
    if(idx < 294){
        T_sin[idx] = sinf(A[idx]);
    }
}

extern "C" void sin_kernel_7x1x6x7(float *h_A, float *h_C) {
    float *d_A, *d_C;
    const int total = 7 * 1 * 6 * 7;

    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_C, total * sizeof(float));

    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(294);
    dim3 numBlocks(1);

    sin_7x1x6x7<<<numBlocks, blockSize>>>(d_A, d_C);

    cudaMemcpy(h_C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_C);
}
