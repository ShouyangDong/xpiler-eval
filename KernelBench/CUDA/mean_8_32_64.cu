// Kernel: reduce along axis=1 for input [8, 32, 64] -> output [8, 64]
// Each thread handles one (n, w) position
__global__ void mean_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // linear index in [0, 511]
    const int output_size = 8 * 64;  // 512
    if (idx >= output_size) return;

    int n = idx / 64;  // batch index: 0~7
    int w = idx % 64;  // width index: 0~63

    float sum = 0.0f;
    for (int h = 0; h < 32; h++) {
        // input[n][h][w]
        int in_idx = n * (32 * 64) + h * 64 + w;
        sum += input[in_idx];
    }
    output[idx] = sum / 32.0f;  // Divide by reduction size (axis=1 has 32 elements)
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void mean_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 8 * 32 * 64;   // 16384
        const int output_size = 8 * 64;       // 512

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(512);
        dim3 numBlocks(1);  // 512 threads → one block is enough

        mean_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);


        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}