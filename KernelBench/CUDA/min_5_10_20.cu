// Kernel: reduce along axis=1 for input [5, 10, 20] -> output [5, 20]
// Each thread handles one (n, w) position
__global__ void min_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // linear index in [0, 99]
    const int output_size = 5 * 20;  // 100
    if (idx >= output_size) return;

    int n = idx / 20;  // batch index: 0~4
    int w = idx % 20;  // width index: 0~19

    float min_val = FLT_MAX;  // initialize to +inf
    for (int h = 0; h < 10; h++) {
        int in_idx = n * (10 * 20) + h * 20 + w;  // input[n][h][w]
        min_val = fminf(min_val, input[in_idx]);
    }
    output[idx] = min_val;
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void min_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 5 * 10 * 20;   // 1000
        const int output_size = 5 * 20;       // 100

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(100);
        dim3 numBlocks(1);  // 100 threads → one block is enough

        min_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);


        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

}