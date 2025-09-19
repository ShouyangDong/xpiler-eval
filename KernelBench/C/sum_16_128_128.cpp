extern "C" void sum(float *input, float *output) {
  int d0 = 16;
  int d1 = 128;
  int d2 = 128;

  // Output size: d1 * d2
  int output_size = d1 * d2;

  // Initialize output to 0
  for (int i = 0; i < output_size; i++) {
    output[i] = 0.0f;
  }

  // Sum over axis 0 (d0)
  for (int k = 0; k < d1; k++) {      // d1 index
    for (int l = 0; l < d2; l++) {    // d2 index
      for (int i = 0; i < d0; i++) {  // sum over d0
        // input[i][k][l] -> offset = i*d1*d2 + k*d2 + l
        output[k * d2 + l] += input[i * d1 * d2 + k * d2 + l];
      }
    }
  }
}
