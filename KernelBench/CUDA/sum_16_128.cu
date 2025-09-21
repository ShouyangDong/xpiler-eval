// Kernel: reduce along axis=1 for input [16, 128] -> output [16]
// Each thread handles one row
__global__ void sum_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 16) return;  // Only 16 rows

    float sum = 0.0f;
    for (int col = 0; col < 128; col++) {
        int idx = row * 128 + col;  // input[row][col]
        sum += input[idx];
    }
    output[row] = sum;  // Sum of 128 elements in the row
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C"
    void mean_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 16 * 128;   // 2048
        const int output_size = 16;        // 16

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(16);
        dim3 numBlocks(1);  // 16 threads → one block is enough

        sum_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}