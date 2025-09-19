extern "C" void permute_4d_0213(float* input, float* output) {
    const int N = 2;
    const int C = 3;
    const int H = 4;
    const int W = 5;

    // NCHW -> NHWC 的中间形态：[N, C, H, W] -> [N, H, C, W]
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int c = 0; c < C; ++c) {
                for (int w = 0; w < W; ++w) {
                    int out_idx = n * (H * C * W) + h * (C * W) + c * W + w;
                    int in_idx  = n * (C * H * W) + c * (H * W) + h * W + w;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}