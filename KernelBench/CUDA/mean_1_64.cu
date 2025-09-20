// Generated: mean along last dimension for input [1x1x1x1x1024] -> [1x1x1x1]
// Total input: 1024, Reduce size: 1024, Output count: 1

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void __launch_bounds__(256)
mean(const float *__restrict__ input, float *__restrict__ output) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= 1) return;

    float sum = 0.0f;
    for (int i = 0; i < 1024; i++) {
        int in_idx = out_idx * 1024 + i;
        sum += input[in_idx];
    }
    output[out_idx] = sum / 1024;  // mean = sum / N
}

extern "C" void mean_kernel_1_1_1_1_1024(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int input_size = 1024;
    const int output_size = 1;

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
