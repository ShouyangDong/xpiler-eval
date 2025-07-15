
float signf(float input) {
  if (input > 0) {
    return 1;
  } else if (input < 0) {
    return -1;
  } else {
    return 0;
  }
}
extern "C" void sign(float *input, float *output) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 10; j++) {
      for (int k = 0; k < 64; k++) {
        output[i * 10 * 64 + j * 64 + k] =
            signf(input[i * 10 * 64 + j * 64 + k]);
      }
    }
  }
}