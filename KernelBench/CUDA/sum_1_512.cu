__global__ void __launch_bounds__(256)
sum(const float *__restrict__ input, float *__restrict__ output) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= 1) return;

    float sum = 0.0f;
    for (int i = 0; i < 512; i++) {
        int in_idx = out_idx * 512 + i;
        sum += input[in_idx];
    }
    output[out_idx] = sum;
}

extern "C" void sum_kernel(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int input_size = 512;
    const int output_size = 1;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((output_size + 255) / 256);

    sum<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
