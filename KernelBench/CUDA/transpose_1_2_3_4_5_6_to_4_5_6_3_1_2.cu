// Generated: Transpose from [1x2x3x4x5x6] to [4x5x6x3x1x2]
// Axes: (3 4 5 2 0 1), Total elements: 720

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void __launch_bounds__(256)
transpose_kernel(const float *__restrict__ input, float *__restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 720) return;

    // Step 1: Flatten index -> multi-dimensional indices (input shape)
    int in_indices[6];
    int tmp = idx;
    in_indices[5] = tmp % 6;
    tmp /= 6;
    in_indices[4] = tmp % 5;
    tmp /= 5;
    in_indices[3] = tmp % 4;
    tmp /= 4;
    in_indices[2] = tmp % 3;
    tmp /= 3;
    in_indices[1] = tmp % 2;
    tmp /= 2;
    in_indices[0] = tmp % 1;
    tmp /= 1;

    // Step 2: Permute indices according to axes
    int out_indices[6];
    out_indices[0] = in_indices[3];
    out_indices[1] = in_indices[4];
    out_indices[2] = in_indices[5];
    out_indices[3] = in_indices[2];
    out_indices[4] = in_indices[0];
    out_indices[5] = in_indices[1];

    // Step 3: Multi-dimensional indices -> linear index (output shape)
    int out_idx = 0;
    out_idx += out_indices[5];
    out_idx += out_indices[4] * 2;
    out_idx += out_indices[3] * 2;
    out_idx += out_indices[2] * 6;
    out_idx += out_indices[1] * 36;
    out_idx += out_indices[0] * 180;

    // Write to output
    output[out_idx] = input[idx];
}

extern "C" void transpose_kernel_1_2_3_4_5_6_to_4_5_6_3_1_2(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int total = 720;

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
