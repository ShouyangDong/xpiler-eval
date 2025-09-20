#include <cuda_runtime.h>

// CUDA kernel: permute(1,0,2) for shape [42,36,55] -> [36,42,55]
__global__ void transpose(
    const float* __restrict__ input,
          float* __restrict__ output
) {
    // Calculate the global thread indices
    int i0 = blockIdx.x * blockDim.x + threadIdx.x; // maps to input dim0 (42)
    int i1 = blockIdx.y * blockDim.y + threadIdx.y; // maps to input dim1 (36)
    int i2 = blockIdx.z * blockDim.z + threadIdx.z; // maps to input dim2 (55)
    const int d0 = 42;  // input dim0
    const int d1 = 36;  // input dim1
    const int d2 = 55;  // input dim2
    // Ensure we do not go out of bounds
    if (i0 < d0 && i1 < d1 && i2 < d2) {
        // Compute the input and output indices based on the permutation rule
        int in_idx  = i0 * (d1 * d2) + i1 * d2 + i2;
        int out_idx = i1 * (d0 * d2) + i0 * d2 + i2;

        // Perform the transposition
        output[out_idx] = input[in_idx];
    }
}

// Host wrapper function to call the CUDA kernel
extern "C" void transpose_kernel(float* host_input, float* host_output,
    int d0, int d1, int d2) {


    // Calculate the sizes in bytes
    size_t in_bytes  = static_cast<size_t>(d0) * d1 * d2 * sizeof(float);
    size_t out_bytes = static_cast<size_t>(d1) * d0 * d2 * sizeof(float);

    // Allocate device memory
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_output, out_bytes);

    // Copy input data from host to device
    cudaMemcpy(d_input, host_input, in_bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(8, 8, 8); // Adjust these values as necessary
    dim3 grid(
        (d0 + block.x - 1) / block.x,
        (d1 + block.y - 1) / block.y,
        (d2 + block.z - 1) / block.z
    );

    // Launch the kernel
    transpose<<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();

    // Copy the result back from device to host
    cudaMemcpy(host_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}