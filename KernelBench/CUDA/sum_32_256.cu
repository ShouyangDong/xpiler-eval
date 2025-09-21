// Kernel: reduce along axis=0 for input [32, 256] -> output [256]
// Each thread handles one column
__global__ void sum_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= 256) return;  // Only 256 columns

    float sum = 0.0f;
    for (int row = 0; row < 32; row++) {
        int idx = row * 256 + col;  // input[row][col]
        sum += input[idx];
    }
    output[col] = sum;  // Sum of 32 elements in the column
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void sum_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 32 * 256;   // 8192
        const int output_size = 256;       // 256

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(256);
        dim3 numBlocks(1);  // 256 threads → one block is enough

        sum_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);


        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

}
