extern "C" void reshape(float *input, float *output) {
  // Flatten and reorder: assumes row-major (C-style) layout
  for (int idx = 0; idx < 9; idx++) {
    // Step 1: Read from input using multi-dimensional indexing
    float val;
    {
      int i0 = idx;
      i0 /= 1;
      val = input[idx];  // input is flat in memory
    }

    // Step 2: Write to output at multi-dimensional position
    {
      int flat_out = 0;
      int o0 = idx % 9;
      int out_idx = o0;
      output[out_idx] = val;
    }
  }
}
