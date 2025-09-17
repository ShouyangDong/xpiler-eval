// Generated: Transpose from [2x3x4] to [4x2x3]
// Axes: (2 0 1), Total elements: 24

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void __launch_bounds__(256)
transpose_kernel(const float *__restrict__ input, float *__restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 24) return;

    // Step 1: Flatten index -> multi-dimensional indices (input shape)
    int in_indices[3];
    int tmp = idx;
    in_indices[2] = tmp % 4;
    tmp /= 4;
    in_indices[1] = tmp % 3;
    tmp /= 3;
    in_indices[0] = tmp % 2;
    tmp /= 2;

    // Step 2: Permute indices according to axes
    int out_indices[3];
    out_indices[0] = in_indices[2];
    out_indices[1] = in_indices[0];
    out_indices[2] = in_indices[1];

    // Step 3: Multi-dimensional indices -> linear index (output shape)
    int out_idx = 0;
    out_idx += out_indices[2];
    out_idx += out_indices[1] * 3;
    out_idx += out_indices[0] * 6;

    // Write to output
    output[out_idx] = input[idx];
}

extern "C" void transpose_kernel_2_3_4_to_4_2_3(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int total = 24;

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
