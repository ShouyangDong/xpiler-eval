// Kernel: reduce along axis=1 for input [3, 32, 3, 32] -> output [3, 3, 32]
// Each thread handles one (n, h, w) position
__global__ void min_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // linear index in [0, 287]
    const int output_size = 3 * 3 * 32;  // 288
    if (idx >= output_size) return;

    // Decode (n, h, w) from linear index
    int n = idx / (3 * 32);                    // batch index: 0~2
    int rem = idx % (3 * 32);
    int h = rem / 32;                          // inner height index: 0~2
    int w = rem % 32;                          // width index: 0~31

    float min_val = FLT_MAX;  // initialize to +inf
    for (int c = 0; c < 32; c++) {  // axis=1 has size 32
        // input[n][c][h][w]
        int in_idx = n * (32 * 3 * 32) + c * (3 * 32) + h * 32 + w;
        min_val = fminf(min_val, input[in_idx]);
    }
    output[idx] = min_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void min_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 3 * 32 * 3 * 32;   // 9216
        const int output_size = 3 * 3 * 32;       // 288

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(288);
        dim3 numBlocks(1);  // 288 threads → one block is enough

        min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_input);
        cudaFree(d_output);

}