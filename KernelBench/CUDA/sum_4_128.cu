// Kernel: reduce along axis=1 for input [4, 128] -> output [4]
// Each thread handles one row
__global__ void sum_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 4) return;  // Only 4 rows

    float sum = 0.0f;
    for (int col = 0; col < 128; col++) {
        int idx = row * 128 + col;  // input[row][col]
        sum += input[idx];
    }
    output[row] = sum;  // No division for sum
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void sum_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 4 * 128;   // 512
        const int output_size = 4;        // 4

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(4);
        dim3 numBlocks(1);  // 4 threads → one block is enough

        sum_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}
