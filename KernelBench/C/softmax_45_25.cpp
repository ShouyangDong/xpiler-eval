extern "C" void softmax(float *x, float *output) {
  float max_val = -INFINITY;
  for (int i = 0; i < 45; i++) {
    for (int j = 0; j < 25; j++) {
      if (x[i * 25 + j] > max_val) {
        max_val = x[i * 25 + j];
      }
    }

    float sum_exp = 0.0;
    for (int j = 0; j < 25; j++) {
      int index = i * 25 + j;
      float exp_val = expf(x[index] - max_val);
      output[index] = exp_val;
      sum_exp += exp_val;
    }

    for (int j = 0; j < 25; j++) {
      output[i * 25 + j] /= sum_exp;
    }
  }
}
