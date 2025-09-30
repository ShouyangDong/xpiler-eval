extern "C" void min(float *input, float *output) {
    int s0 = 64;

    output[0] = INFINITY;

    for (int i0 = 0; i0 < s0; i0++) {
        if (input[i0] < output[0]) {
            output[0] = input[i0];
        }
    }
}
