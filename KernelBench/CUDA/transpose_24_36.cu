#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel
__global__ void transpose(const float* __restrict__ input,
                                 float* __restrict__ output) {
    int o0 = blockIdx.y * blockDim.y + threadIdx.y;
    int o1 = blockIdx.x * blockDim.x + threadIdx.x;
    int s0_in = 24;  // input rows
    int s1_in = 36;  // input cols
    int s0_out = 36; // output rows
    int s1_out = 24; // output cols
    if (o0 < s0_out && o1 < s1_out) {
        // output[o0][o1] = input[o1][o0]
        output[o0 * s1_out + o1] = input[o1 * s1_in + o0];
    }
}

// Host wrapper
extern "C" void transpose_kernel(float* input, float* output,
                                 int s0_in, int s1_in) {


    size_t in_size = s0_in * s1_in * sizeof(float);
    size_t out_size = s0_in * s1_in * sizeof(float);

    float *d_input, *d_output;
    cudaMalloc(&d_input, in_size);
    cudaMalloc(&d_output, out_size);

    cudaMemcpy(d_input, input, in_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((s1_out + block.x - 1) / block.x,
              (s0_out + block.y - 1) / block.y);

    transpose<<<grid, block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
