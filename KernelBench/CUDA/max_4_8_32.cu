// Kernel: reduce along axis=0 for input [4, 8, 32] -> output [8, 32]
// Each thread handles one spatial location (h, w)
__global__ void max_dev(const float* __restrict__ input, float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // linear index in [0, 255]
    if (idx >= 8 * 32) return;  // total output elements: 256

    int h = idx / 32;  // height index: 0~7
    int w = idx % 32;  // width index: 0~31

    float max_val = -FLT_MAX;
    for (int n = 0; n < 4; n++) {
        int in_idx = n * (8 * 32) + h * 32 + w;  // input[n][h][w]
        max_val = fmaxf(max_val, input[in_idx]);
    }
    output[idx] = max_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void max_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 4 * 8 * 32;   // 1024
        const int output_size = 8 * 32;      // 256

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(256);
        dim3 numBlocks((output_size + 255) / 256);  // (256 + 255) / 256 = 1

        max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

}
