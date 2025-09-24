extern "C" void depthwiseconv(float *input, float *kernel, float *output) {
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 254; ++i) {
      for (int j = 0; j < 254; ++j) {
        output[i * 254 * 3 + j * 3 + c] = 0.0;
        for (int fi = 0; fi < 3; ++fi) {
          for (int fj = 0; fj < 3; ++fj) {
            output[i * 254 * 3 + j * 3 + c] +=
                input[(i + fi) * 3 * 256 + (j + fj) * 3 + c] *
                kernel[fi * 3 * 3 + fj * 3 + c];
          }
        }
      }
    }
  }
}