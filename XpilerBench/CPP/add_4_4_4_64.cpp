extern "C" void add(float *input1, float *input2, float *output) {
  int dim1 = 4;
  int dim2 = 4;
  int dim3 = 4;
  int dim4 = 64;

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      for (int k = 0; k < dim3; k++) {
        for (int l = 0; l < dim4; l++) {
          int index = i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l;
          output[index] = input1[index] + input2[index];
        }
      }
    }
  }
}
