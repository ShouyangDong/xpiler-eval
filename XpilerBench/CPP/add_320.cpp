extern "C" void add(float *input1, float *input2, float *output) {
  int size = 320;

  for (int i = 0; i < size; i++) {
    output[i] = input1[i] + input2[i];
  }
}
