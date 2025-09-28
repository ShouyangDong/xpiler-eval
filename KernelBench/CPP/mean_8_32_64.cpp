extern "C" void mean(float *input, float *output) {
  const int M = 8;
  const int N = 32;
  const int K = 64;

  const int output_size = M * K;

  for (int i = 0; i < output_size; i++) {
    output[i] = 0.0f;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        int in_idx = i * (N * K) + j * K + k;
        int out_idx = i * K + k;
        output[out_idx] += input[in_idx];
      }
    }
  }

  for (int i = 0; i < output_size; i++) {
    output[i] /= N;
  }
}
