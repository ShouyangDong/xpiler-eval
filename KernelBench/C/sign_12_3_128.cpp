
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
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 128; k++) {
        output[i * 3 * 128 + j * 128 + k] =
            signf(input[i * 3 * 128 + j * 128 + k]);
      }
    }
  }
}