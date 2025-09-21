__global__ void max_dev(const float* __restrict__ input, float* __restrict__ output) {
    if (threadIdx.x == 0) {  // 只用一个线程
        float max_val = -FLT_MAX;
        for (int i = 0; i < 128; i++) {
            max_val = fmaxf(max_val, input[i]);
        }
        output[0] = max_val;
    }
}

// Host wrapper - DO NOT CHANGE FUNCTION NAME
extern "C" void max_kernel(const float* h_input, float* h_output, int a) {
        float *d_input, *d_output;
        const int input_size = a;
        const int output_size = 1;

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input from host to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel: 1 block, 128 threads, but only threadIdx.x == 0 does work
        dim3 blockSize(128);
        dim3 numBlocks(1);
        max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

        // Copy result back
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
}
