
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
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 5; k++) {
        output[i * 4 * 5 + j * 5 + k] = signf(input[i * 4 * 5 + j * 5 + k]);
      }
    }
  }
}