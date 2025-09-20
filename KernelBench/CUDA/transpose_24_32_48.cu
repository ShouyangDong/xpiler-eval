extern "C" __global__ void transpose(const float* __restrict__ input,
                                            float* __restrict__ output) {
    const int d0 = 24;
    const int d1 = 32;
    const int d2 = 48;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2;
    if (idx >= total) return;

    // 解出三维坐标
    int i2 = idx % d2;
    int i1 = (idx / d2) % d1;
    int i0 = idx / (d1 * d2);

    // output[i2][i0][i1] = input[i0][i1][i2]
    output[((i2 * d0) + i0) * d1 + i1] =
        input[((i0 * d1) + i1) * d2 + i2];
}

extern "C" void transpose_kernel(const float* input, float* output, int d0, int d1, int d2) {
    int total = d0 * d1 * d2;

    float *d_input, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, input, total * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose<<<blocks, threads>>>(d_input, d_output);

    cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

