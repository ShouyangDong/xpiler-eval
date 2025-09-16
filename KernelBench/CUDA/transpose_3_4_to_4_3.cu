// Generated: Transpose from [3x4] to [4x3]
// Axes: (1 0), Total elements: 12

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void __launch_bounds__(256)
transpose_kernel(const float *__restrict__ input, float *__restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 12) return;

    // Step 1: Flatten index -> multi-dimensional indices (input shape)
    int in_indices[2];
    int tmp = idx;
    in_indices[1] = tmp % 4;
    tmp /= 4;
    in_indices[0] = tmp % 3;
    tmp /= 3;

    // Step 2: Permute indices according to axes
    int out_indices[2];
    out_indices[0] = in_indices[1];
    out_indices[1] = in_indices[0];

    // Step 3: Multi-dimensional indices -> linear index (output shape)
    int out_idx = 0;
    out_idx += out_indices[1];
    out_idx += out_indices[0] * 3;

    // Write to output
    output[out_idx] = input[idx];
}

extern "C" void transpose_kernel_3_4_to_4_3(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int total = 12;

    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((total + 255) / 256);

    transpose_kernel<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
