extern "C" void add(float *input1, float *input2, float *output) {
  int dim1 = 3;
  int dim2 = 3;
  int dim3 = 256;

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      for (int k = 0; k < dim3; k++) {
        int index = i * dim2 * dim3 + j * dim3 + k;
        output[index] = input1[index] + input2[index];
      }
    }
  }
}
