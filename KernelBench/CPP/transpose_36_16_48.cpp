extern "C" void transpose(float* input, float* output) {
    const int D0 = 36; 
    const int D1 = 16;
    const int D2 = 48;

    for (int d0 = 0; d0 < D0; d0++) {
        for (int d1 = 0; d1 < D1; d1++) {
            for (int d2 = 0; d2 < D2; d2++) {
                int input_idx = d0 * (D1 * D2) + d1 * D2 + d2;
                int output_idx = d0 * (D2 * D1) + d2 * D1 + d1;

                output[output_idx] = input[input_idx];
            }
        }
    }
}
