// Kernel: reduce along axis=0 for input [64] -> output scalar
// Only one thread needed, but we use one block for simplicity
__global__ void min_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    float min_val = FLT_MAX;  // Initialize to +infinity

    for (int i = 0; i < 64; i++) {
        min_val = fminf(min_val, input[i]);
    }

    *output = min_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C"
    void min_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 64;
        const int output_size = 1;  // scalar

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        // Only need 1 thread, but use a small block
        dim3 blockSize(1);
        dim3 numBlocks(1);

        min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);


        // Copy result back to host (h_output points to a single float)
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_input);
        cudaFree(d_output);

}
