extern "C" void permute_3d_021(float* input, float* output) {
    const int d0 = 3;
    const int d1 = 4;
    const int d2 = 5;

    // output[d0][d2][d1] = input[d0][d1][d2]
    for (int i0 = 0; i0 < d0; ++i0) {
        for (int i2 = 0; i2 < d2; ++i2) {
            for (int i1 = 0; i1 < d1; ++i1) {
                int out_idx = i0 * (d2 * d1) + i2 * d1 + i1;
                int in_idx  = i0 * (d1 * d2) + i1 * d2 + i2;
                output[out_idx] = input[in_idx];
            }
        }
    }
}