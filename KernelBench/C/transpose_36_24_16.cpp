extern "C" void transpose(float* input, float* output) {
    const int d0 = 36;
    const int d1 = 24;
    const int d2 = 16;

    // output[d1][d0][d2] = input[d0][d1][d2]
    for (int i1 = 0; i1 < d1; ++i1) {
        for (int i0 = 0; i0 < d0; ++i0) {
            for (int i2 = 0; i2 < d2; ++i2) {
                int out_idx = i1 * (d0 * d2) + i0 * d2 + i2;
                int in_idx  = i0 * (d1 * d2) + i1 * d2 + i2;
                output[out_idx] = input[in_idx];
            }
        }
    }
}
