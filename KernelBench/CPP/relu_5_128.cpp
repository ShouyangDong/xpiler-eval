
float reluf(float input) { return input > 0 ? input : 0; }
extern "C" void relu(float *input, float *output) {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 128; j++) {
      output[i * 128 + j] = reluf(input[i * 128 + j]);
    }
  }
}