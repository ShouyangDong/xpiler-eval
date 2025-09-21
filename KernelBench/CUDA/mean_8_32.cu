// Kernel: reduce along axis=0 for input [8, 32] -> output [32]
// Each thread handles one column
__global__ void mean_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= 32) return;  // Only 32 columns

    float sum = 0.0f;
    for (int row = 0; row < 8; row++) {
        int idx = row * 32 + col;  // input[row][col]
        sum += input[idx];
    }
    output[col] = sum / 8.0f;  // Divide by reduction size
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void mean_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 8 * 32;   // 256
        const int output_size = 32;      // 32

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(32);
        dim3 numBlocks(1);  // 32 threads → one block is enough

        mean_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

}