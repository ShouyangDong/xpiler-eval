// =============================================================================
// Modified CUDA Kernel for shape [2, 1, 1024]
// =============================================================================

__global__ void __launch_bounds__(1024) // Adjusted launch bounds to match the maximum number of threads per block.
sin(const float *__restrict__ A, float *__restrict__ T_sin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements){
        T_sin[idx] = sinf(A[idx]);
    }
}

extern "C" void sin_kernel(float *h_A, float *h_C, int n, int elements_per_batch) {
    float *d_A, *d_C;
    const int total_elements = n * elements_per_batch;

    cudaMalloc(&d_A, total_elements * sizeof(float));
    cudaMalloc(&d_C, total_elements * sizeof(float));

    cudaMemcpy(d_A, h_A, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes based on the total number of elements.
    dim3 blockSize(256); // You can adjust this value based on your GPU's capabilities. Common values are 128, 256, etc.
    dim3 numBlocks((total_elements + blockSize.x - 1) / blockSize.x); // Calculate required blocks to cover all elements.

    // Launch the kernel
    sin<<<numBlocks, blockSize>>>(d_A, d_C);
    cudaDeviceSynchronize(); // Ensure all operations are completed before copying data back to host.

    cudaMemcpy(h_C, d_C, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_C);
}