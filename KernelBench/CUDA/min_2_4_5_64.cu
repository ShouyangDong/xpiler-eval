// Kernel: reduce along axis=1 for input [2, 4, 5, 64] -> output [2, 5, 64]
// Each thread handles one (n, h, w) position
__global__ void min_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // linear index in [0, 639]
    const int output_size = 2 * 5 * 64;  // 640
    if (idx >= output_size) return;

    // Decode (n, h, w) from linear index
    int n = idx / (5 * 64);                    // batch index: 0~1
    int rem = idx % (5 * 64);
    int h = rem / 64;                          // height index: 0~4
    int w = rem % 64;                          // width index: 0~63

    float min_val = FLT_MAX;  // initialize to +inf
    for (int c = 0; c < 4; c++) {  // axis=1 has size 4
        // input[n][c][h][w]
        int in_idx = n * (4 * 5 * 64) + c * (5 * 64) + h * 64 + w;
        min_val = fminf(min_val, input[in_idx]);
    }
    output[idx] = min_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void min_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 2 * 4 * 5 * 64;   // 2560
        const int output_size = 2 * 5 * 64;      // 640

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(640);
        dim3 numBlocks(1);  // 640 threads → one block is enough

        min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);
        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

}