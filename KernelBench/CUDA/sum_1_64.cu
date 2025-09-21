// Kernel: reduce along axis=1 for input [1, 64] -> output [1]
// Only one thread is needed since output is scalar-like
__global__ void sum_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int tid = threadIdx.x;
    if (tid != 0) return;  // Only use thread 0

    float sum = 0.0f;
    for (int col = 0; col < 64; col++) {
        int idx = 0 * 64 + col;  // input[0][col]
        sum += input[idx];
    }
    output[0] = sum;  // Sum of 64 elements
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C"void mean_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 1 * 64;   // 64
        const int output_size = 1;       // 1

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(1);    // Only 1 thread needed
        dim3 numBlocks(1);    // One block

        sum_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

}