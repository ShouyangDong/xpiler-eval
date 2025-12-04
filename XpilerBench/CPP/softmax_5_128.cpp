extern "C" void softmax(float *x, float *output) {

  float max_val = -INFINITY;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 128; j++) {
      if (x[i * 128 + j] > max_val) {
        max_val = x[i * 128 + j];
      }
    }

    float sum_exp = 0.0;
    for (int j = 0; j < 128; j++) {
      int index = i * 128 + j;
      float exp_val = expf(x[index] - max_val);
      output[index] = exp_val;
      sum_exp += exp_val;
    }

    for (int j = 0; j < 128; j++) {
      output[i * 128 + j] /= sum_exp;
    }
  }
}