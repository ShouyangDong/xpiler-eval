__global__ void __launch_bounds__(256)
copy_reshape(const float *__restrict__ input, float *__restrict__ output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = 512;

    for (int i = idx; i < total; i += stride) {
        output[i] = input[i];
    }
}

extern "C" void reshape_kernel_8_8_8_to_512_1_1(const float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int total_elements = 512;

    cudaMalloc(&d_input, total_elements * sizeof(float));
    cudaMalloc(&d_output, total_elements * sizeof(float));

    cudaMemcpy(d_input, h_input, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((total_elements + 255) / 256);

    copy_reshape<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
