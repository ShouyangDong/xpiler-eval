// Kernel: reduce along axis=1 for input [4, 64, 64] -> output [4, 64]
// Each thread handles one (n, w) position
__global__ void min_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // linear index in [0, 255]
    const int output_size = 4 * 64;  // 256
    if (idx >= output_size) return;

    int n = idx / 64;  // batch index: 0~3
    int w = idx % 64;  // width index: 0~63

    float min_val = FLT_MAX;  // initialize to +inf
    for (int h = 0; h < 64; h++) {
        // input[n][h][w]
        int in_idx = n * (64 * 64) + h * 64 + w;
        min_val = fminf(min_val, input[in_idx]);
    }
    output[idx] = min_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void min_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 4 * 64 * 64;   // 16384
        const int output_size = 4 * 64;       // 256

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(256);
        dim3 numBlocks(1);  // 256 threads → one block is enough

        min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);


        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}