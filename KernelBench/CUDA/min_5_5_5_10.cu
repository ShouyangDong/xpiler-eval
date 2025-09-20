// Generated: min along last dimension for input [5x5x5x10] -> [5x5x5]
// Total input: 1250, Reduce size: 10, Output count: 125

#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

__global__ void __launch_bounds__(256)
min(const float *__restrict__ input, float *__restrict__ output) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= 125) return;

    float min_val = FLT_MAX;
    for (int i = 0; i < 10; i++) {
        int in_idx = out_idx * 10 + i;
        float val = input[in_idx];
        min_val = fminf(min_val, val);
    }
    output[out_idx] = min_val;
}

extern "C" void min_kernel(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int input_size = 1250;
    const int output_size = 125;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((output_size + 255) / 256);

    min<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
