// Kernel: reduce along axis=1 for input [4, 1] -> output [4]
// Each thread handles one row
__global__ void mean_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 4) return;  // Only 4 rows

    // Since axis=1 has size 1, mean = the only element
    int idx = row * 1 + 0;  // input[row][0]
    output[row] = input[idx];  // No need to divide: sum / 1 = sum
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void mean_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 4 * 1;   // 4
        const int output_size = 4;      // 4

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(4);
        dim3 numBlocks(1);  // 4 threads → one block is enough

        mean_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}
