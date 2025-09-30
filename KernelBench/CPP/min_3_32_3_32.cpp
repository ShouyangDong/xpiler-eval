extern "C" void min(float *input, float *output) {
    const int s0 = 3;
    const int s1 = 32;  
    const int s2 = 3;
    const int s3 = 32;

    for (int i0 = 0; i0 < s0; ++i0) {
        for (int i2 = 0; i2 < s2; ++i2) {
            for (int i3 = 0; i3 < s3; ++i3) {
                output[i0 * s2 * s3 + i2 * s3 + i3] = INFINITY;
            }
        }
    }

    for (int i0 = 0; i0 < s0; ++i0) {
        for (int i1 = 0; i1 < s1; ++i1) {
            for (int i2 = 0; i2 < s2; ++i2) {
                for (int i3 = 0; i3 < s3; ++i3) {
                    size_t in_idx = (size_t)i0 * s1 * s2 * s3
                                  + (size_t)i1 * s2 * s3
                                  + (size_t)i2 * s3
                                  + (size_t)i3;
                    size_t out_idx = (size_t)i0 * s2 * s3
                                   + (size_t)i2 * s3
                                   + (size_t)i3;
                    float val = input[in_idx];
                    if (val < output[out_idx]) {
                        output[out_idx] = val;
                    }
                }
            }
        }
    }
}
