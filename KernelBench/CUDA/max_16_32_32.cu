// Kernel: reduce along axis=1 for input [16, 32, 32] -> output [16, 32]
// Each thread handles one (n, w) position
__global__ void max_dev(const float* __restrict__ input, float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_size = 16 * 32;  // 512
    if (idx >= output_size) return;

    int n = idx / 32;  // batch index: 0~15
    int w = idx % 32;  // width index: 0~31

    float max_val = -FLT_MAX;
    for (int h = 0; h < 32; h++) {
        int in_idx = n * (32 * 32) + h * 32 + w;  // input[n][h][w]
        max_val = fmaxf(max_val, input[in_idx]);
    }
    output[idx] = max_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void max_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 16 * 32 * 32;  // 16384
        const int output_size = 16 * 32;      // 512

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(256);
        dim3 numBlocks((output_size + 255) / 256);  // (512 + 255) / 256 = 3

        max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}