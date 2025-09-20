// CUDA kernel: permute(1,0,2) for shape [36,24,16] -> [24,36,16]
__global__ void transpose(
    const float* __restrict__ input,
          float* __restrict__ output
) {
    const int d0 = 36;  // input dim0
    const int d1 = 24;  // input dim1
    const int d2 = 16;  // input dim2
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;  // maps to input dim0 (36)
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;  // maps to input dim1 (24)
    int i2 = blockIdx.z * blockDim.z + threadIdx.z;  // maps to input dim2 (16)

    if (i0 < d0 && i1 < d1 && i2 < d2) {
        int in_idx  = i0 * (d1 * d2) + i1 * d2 + i2;
        int out_idx = i1 * (d0 * d2) + i0 * d2 + i2;  // output[i1][i0][i2]

        output[out_idx] = input[in_idx];
    }
}

// Host wrapper function
extern "C" void transpose_kernel(float* host_input, float* host_output,
    int d0, int d1, int d2) {
    const size_t in_bytes  = (size_t)d0 * d1 * d2 * sizeof(float);
    const size_t out_bytes = (size_t)d1 * d0 * d2 * sizeof(float);

    float *d_input = nullptr;
    float *d_output = nullptr;

    cudaMalloc(&d_input, in_bytes);
    cudaMalloc(&d_output, out_bytes);

    cudaMemcpy(d_input, host_input, in_bytes, cudaMemcpyHostToDevice);

    // 3D grid and block
    dim3 block(8, 8, 8);  // 512 threads per block (max 1024)
    dim3 grid(
        (d0 + block.x - 1) / block.x,
        (d1 + block.y - 1) / block.y,
        (d2 + block.z - 1) / block.z
    );

    transpose<<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(host_output, d_output, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}