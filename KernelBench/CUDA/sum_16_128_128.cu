// Generated: sum along last dimension for input [2x1x1024] -> [2x1]
// Total input: 2048, Reduce size: 1024, Output count: 2

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void __launch_bounds__(256)
sum(const float *__restrict__ input, float *__restrict__ output) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= 2) return;

    float sum = 0.0f;
    for (int i = 0; i < 1024; i++) {
        int in_idx = out_idx * 1024 + i;
        sum += input[in_idx];
    }
    output[out_idx] = sum;
}

extern "C" void sum_kernel(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int input_size = 2048;
    const int output_size = 2;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((output_size + 255) / 256);

    sum<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
