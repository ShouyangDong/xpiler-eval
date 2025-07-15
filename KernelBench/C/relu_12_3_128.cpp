
float reluf(float input) { return input > 0 ? input : 0; }
extern "C" void relu(float *input, float *output) {
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 128; k++) {
        output[i * 3 * 128 + j * 128 + k] =
            reluf(input[i * 3 * 128 + j * 128 + k]);
      }
    }
  }
}