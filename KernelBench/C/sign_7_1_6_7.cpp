
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
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 6; k++) {
        for (int l = 0; l < 7; l++) {
          output[i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l] =
              signf(input[i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l]);
        }
      }
    }
  }
}