extern "C" void transpose(float* input, float* output) {
    const int N = 1;
    const int C = 3;
    const int H = 224;
    const int W = 224;

    // NCHW -> NHWC
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    int out_idx = n * (H * W * C) + h * (W * C) + w * C + c;
                    int in_idx  = n * (C * H * W) + c * (H * W) + h * W + w;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}
