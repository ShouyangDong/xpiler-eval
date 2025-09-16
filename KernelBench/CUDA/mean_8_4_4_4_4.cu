// Generated: mean along last dimension for input [8x4x4x4x4] -> [8x4x4x4]
// Total input: 2048, Reduce size: 4, Output count: 512

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void __launch_bounds__(256)
mean_last_dim(const float *__restrict__ input, float *__restrict__ output) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= 512) return;

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        int in_idx = out_idx * 4 + i;
        sum += input[in_idx];
    }
    output[out_idx] = sum / 4;  // mean = sum / N
}

extern "C" void mean_kernel_8_4_4_4_4(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int input_size = 2048;
    const int output_size = 512;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((output_size + 255) / 256);

    mean_last_dim<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
