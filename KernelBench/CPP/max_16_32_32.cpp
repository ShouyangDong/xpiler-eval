extern "C" void max(float *input, float *output) {
    for (int i = 0; i < 16; ++i) {
        for (int k = 0; k < 32; ++k) {
            output[i * 32 + k] = -INFINITY;
        }
    }

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 32; ++j) {
            int  base =i * 32 * 32 +j * 32;
            int  out_base =i * 32;
            for (int k = 0; k < 32; ++k) {
                float val = input[base + k];
                if (val > output[out_base + k]) {
                    output[out_base + k] = val;
                }
            }
        }
    }
}
