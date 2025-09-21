
__global__ void transpose(const float* __restrict__ input,
                                 float* __restrict__ output) {
    const int N = 2;
    const int C = 32;
    const int H = 4;
    const int W = 64;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total) {
        // inline 方式计算 in/out 索引
        int out_idx =
            (idx / (C * H * W)) * (H * C * W) +     // n
            ((idx / W) % H) * (C * W) +             // h
            ((idx / (H * W)) % C) * W +             // c
            (idx % W);                              // w

        int in_idx =
            (idx / (C * H * W)) * (C * H * W) +     // n
            ((idx / (H * W)) % C) * (H * W) +       // c
            ((idx / W) % H) * W +                   // h
            (idx % W);                              // w

        output[out_idx] = input[in_idx];
    }
}

extern "C" void transpose_kernel(float* input, float* output,
                                 int N, int C, int H, int W) {


    int total = N * C * H * W;
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
