extern "C" void transpose(float *input, float *output) {
    int s0_in = 2;  // input rows
    int s1_in = 3;  // input cols

    int s0_out = 3;  // output rows (was s0_in)
    int s1_out = 2;  // output cols (was s1_in)

    for (int o0 = 0; o0 < s0_out; o0++) {
        for (int o1 = 0; o1 < s1_out; o1++) {
            // output[o0][o1] = input[o1][o0]
            output[o0 * s1_out + o1] = input[o1 * s1_in + o0];
        }
    }
}