// Kernel: reduce along axis=1 for input [8, 16, 8] -> output [8, 8]
// Each thread handles one (n, w) position
__global__ void min_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // linear index in [0, 63]
    const int output_size = 8 * 8;  // 64
    if (idx >= output_size) return;

    int n = idx / 8;  // batch index: 0~7
    int w = idx % 8;  // width index: 0~7

    float min_val = FLT_MAX;  // initialize to +inf
    for (int h = 0; h < 16; h++) {
        int in_idx = n * (16 * 8) + h * 8 + w;  // input[n][h][w]
        min_val = fminf(min_val, input[in_idx]);
    }
    output[idx] = min_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C"
    void min_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 8 * 16 * 8;   // 1024
        const int output_size = 8 * 8;       // 64

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(64);
        dim3 numBlocks(1);  // 64 threads → one block is enough

        min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}