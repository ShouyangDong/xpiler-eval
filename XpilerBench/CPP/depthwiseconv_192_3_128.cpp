extern "C" void depthwiseconv(float *input, float *kernel, float *output) {
  for (int c = 0; c < 128; ++c) {
    for (int i = 0; i < 190; ++i) {
      for (int j = 0; j < 190; ++j) {
        output[i * 190 * 128 + j * 128 + c] = 0.0;
        for (int fi = 0; fi < 3; ++fi) {
          for (int fj = 0; fj < 3; ++fj) {
            output[i * 190 * 128 + j * 128 + c] +=
                input[(i + fi) * 128 * 192 + (j + fj) * 128 + c] *
                kernel[fi * 3 * 128 + fj * 128 + c];
            ;
          }
        }
      }
    }
  }
}