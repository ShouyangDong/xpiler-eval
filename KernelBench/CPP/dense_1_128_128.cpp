extern "C" void dense(float *input, float *weight, float *output) {
  int N = 1;
  int in_features = 128;
  int out_features = 128;

  for (int n = 0; n < N; n++) {
    for (int o = 0; o < out_features; o++) {
      float sum = 0.0f;
      for (int i = 0; i < in_features; i++) {
        int input_idx = n * in_features + i;
        int weight_idx = o * in_features + i;
        sum += input[input_idx] * weight[weight_idx];
      }
      int output_idx = n * out_features + o;
      output[output_idx] = sum;
    }
  }
}
