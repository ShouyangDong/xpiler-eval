// Kernel: reduce along axis=0 for input [16, 128, 128] -> output [128, 128]
// Each thread handles one (row, col) element
__global__ void sum_kernel_dev(const float* __restrict__ input, float* __restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 128 * 128) return;  // Only 128x128 = 16384 elements

    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        int input_idx = i * (128 * 128) + idx;  // input[i][j][k], where j*128 + k = idx
        sum += input[input_idx];
    }
    output[idx] = sum;  // No division for sum
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" 
    void mean_kernel(const float* h_input, float* h_output) {
        float *d_input, *d_output;
        const int input_size = 16 * 128 * 128;   // 262144
        const int output_size = 128 * 128;       // 16384

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        int total_elements = 128 * 128;
        int blockSize = 256;
        int numBlocks = (total_elements + blockSize - 1) / blockSize;  // ceil(16384 / 256) = 64

        sum_kernel_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back to host
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
    
}