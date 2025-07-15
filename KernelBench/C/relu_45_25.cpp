
float reluf(float input) { return input > 0 ? input : 0; }
extern "C" void relu(float *input, float *output) {
  for (int i = 0; i < 45; i++) {
    for (int j = 0; j < 25; j++) {
      output[i * 25 + j] = reluf(input[i * 25 + j]);
    }
  }
}