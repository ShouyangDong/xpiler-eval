// Generated: mean along last dimension for input [3x3x3x3x3x3] -> [3x3x3x3x3]
// Total input: 729, Reduce size: 3, Output count: 243

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void __launch_bounds__(256)
mean(const float *__restrict__ input, float *__restrict__ output) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= 243) return;

    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        int in_idx = out_idx * 3 + i;
        sum += input[in_idx];
    }
    output[out_idx] = sum / 3;  // mean = sum / N
}

extern "C" void mean_kernel_3_3_3_3_3_3(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int input_size = 729;
    const int output_size = 243;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((output_size + 255) / 256);

    mean<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
