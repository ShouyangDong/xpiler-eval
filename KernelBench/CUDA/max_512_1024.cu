// Kernel: reduce along axis=0 for input [512, 1024] -> output [1024]
// Each thread handles one column (w)
__global__ void max_dev(const float* __restrict__ input, float* __restrict__ output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= 1024) return;  // only 1024 columns

    float max_val = -FLT_MAX;
    for (int row = 0; row < 512; row++) {
        int idx = row * 1024 + col;  // input[row][col]
        max_val = fmaxf(max_val, input[idx]);
    }
    output[col] = max_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void max_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 512 * 1024;   // 524288
        const int output_size = 1024;        // 1024

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(256);
        dim3 numBlocks((1024 + 255) / 256);  // (1024 + 255) / 256 = 5 blocks

        max_dev<<<numBlocks, blockSize>>>(d_input, d_output);
        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}
