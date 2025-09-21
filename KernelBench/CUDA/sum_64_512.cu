// Kernel: reduce along axis=1 for input [64, 512] -> output [64]
// Each thread handles one row
__global__ void sum_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 64) return;  // Only 64 rows

    float sum = 0.0f;
    for (int col = 0; col < 512; col++) {
        int idx = row * 512 + col;  // input[row][col]
        sum += input[idx];
    }
    output[row] = sum;  // No division for sum
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void sum_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 64 * 512;   // 32768
        const int output_size = 64;        // 64

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(64);
        dim3 numBlocks(1);  // 64 threads → one block is enough

        sum_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);


        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}
