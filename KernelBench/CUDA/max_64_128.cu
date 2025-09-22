// Fixed: max along axis=0 for input [64x128] -> [128]
// Total input: 8192, Reduce size: 64, Output count: 128

__global__ void __launch_bounds__(256)
max_dev(const float *__restrict__ input, float *__restrict__ output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    if (col >= 128) return;

    float max_val = -FLT_MAX;
    for (int row = 0; row < 64; row++) {
        int idx = row * 128 + col;  // row-major indexing
        float val = input[idx];
        max_val = fmaxf(max_val, val);
    }
    output[col] = max_val;
}

extern "C" void max_kernel(float *h_input, float *h_output) {
    float *d_input, *d_output;
    const int input_size = 8192;     // 64 * 128
    const int output_size = 128;     // one per column

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((128 + 255) / 256);  // ceil(128 / 256) = 1

    max_dev<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
