extern "C" void mean(float* input, float* output) {
  const int M = 8;   // dim0
  const int N = 32;  // dim1 (reduction axis)
  const int K = 64;  // dim2

  // Output shape: [M, K] = [8, 64]
  const int output_size = M * K;

  // Initialize output to 0
  for (int i = 0; i < output_size; i++) {
    output[i] = 0.0f;
  }

  // Sum over dimension 1 (N-axis, size=32)
  // Input index: i * N * K + j * K + k
  // Output index: i * K + k
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        int in_idx = i * (N * K) + j * K + k;
        int out_idx = i * K + k;
        output[out_idx] += input[in_idx];
      }
    }
  }

  // Divide by N to get mean
  for (int i = 0; i < output_size; i++) {
    output[i] /= N;  // divide by 32
  }
}
