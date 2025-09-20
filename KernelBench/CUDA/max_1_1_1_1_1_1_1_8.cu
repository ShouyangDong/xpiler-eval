// Generated: max along last dimension for input [1x1x1x1x1x1x1x8] -> [1x1x1x1x1x1x1]
// Total input: 8, Reduce size: 8, Output count: 1

#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

__global__ void __launch_bounds__(256)
max_last_dim(const float *__restrict__ input, float *__restrict__ output) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= 1) return;

    float max_val = -FLT_MAX;
    for (int i = 0; i < 8; i++) {
        int in_idx = out_idx * 8 + i;
        float val = input[in_idx];
        max_val = fmaxf(max_val, val);
    }
    output[out_idx] = max_val;
}

extern "C" void max(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int input_size = 8;
    const int output_size = 1;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((output_size + 255) / 256);

    max_last_dim<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
