// Kernel: reduce along axis=1 for input [128, 256] -> output [128]
// Each thread handles one row (n)
__global__ void max_dev(const float* __restrict__ input, float* __restrict__ output) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 128) return;  // only 128 rows

    float max_val = -FLT_MAX;
    for (int col = 0; col < 256; col++) {
        int idx = row * 256 + col;  // input[row][col]
        max_val = fmaxf(max_val, input[idx]);
    }
    output[row] = max_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void max_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 128 * 256;   // 32768
        const int output_size = 128;        // 128

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(128);
        dim3 numBlocks(1);  // 128 threads → one block is enough

        max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}