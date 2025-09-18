extern "C" void transpose(float* input, float* output) {
    int s0_in = 3;
    int s1_in = 1;  // singleton dimension
    int s2_in = 4;  // you probably forgot to define this!
    // [3, 1, 4] -> [3, 4, 1]
    int s0_out = s2_in;  // output dim0 comes from input dim2
    int s1_out = s0_in;  // output dim1 comes from input dim0

    // Perform: output[d2][d0] = input[d0][0][d2]
    for (int o0 = 0; o0 < s0_out; ++o0) {        // o0: d2 index
        for (int o1 = 0; o1 < s1_out; ++o1) {    // o1: d0 index
            // input[d0=o1][d1=0][d2=o0]
            output[o0 * s1_out + o1] = input[o1 * (s1_in * s2_in) + 0 * s2_in + o0];
        }
    }
}
