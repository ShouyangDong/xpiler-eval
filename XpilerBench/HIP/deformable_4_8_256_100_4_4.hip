__global__ void __launch_bounds__(64)
    deformable(float *__restrict__ attention_weights_,
               float *__restrict__ output_,
               float *__restrict__ sampling_locations_,
               float *__restrict__ value_,
               int *__restrict__ value_level_start_index_,
               int *__restrict__ value_spatial_shapes_) {
  float attention_sum[4];
  int height_width[2];
  float xy[2];
  float xy_grid[2];
  int xy_rounded[4];
  float corner_values[16];
  for (int ii_d = 0; ii_d < 4; ++ii_d) {
    attention_sum[ii_d] = 0.000000e+00f;
  }
  for (int i = 0; i < 4; ++i) {
    height_width[0] = value_spatial_shapes_[(i * 2)];
    height_width[1] = value_spatial_shapes_[((i * 2) + 1)];
    for (int k = 0; k < 4; ++k) {
      xy[1] = sampling_locations_[(
          ((((((int)blockIdx.y) * 25600) + (((int)blockIdx.x) * 256)) +
            (((int)blockIdx.z) * 32)) +
           (i * 8)) +
          (k * 2))];
      xy[0] = sampling_locations_[(
          (((((((int)blockIdx.y) * 25600) + (((int)blockIdx.x) * 256)) +
             (((int)blockIdx.z) * 32)) +
            (i * 8)) +
           (k * 2)) +
          1)];
      xy_grid[0] = ((xy[0] * ((float)height_width[0])) - 5.000000e-01f);
      xy_grid[1] = ((xy[1] * ((float)height_width[1])) - 5.000000e-01f);
      xy_rounded[0] = ((int)floorf(xy_grid[0]));
      xy_rounded[1] = (xy_rounded[0] + 1);
      xy_rounded[2] = ((int)floorf(xy_grid[1]));
      xy_rounded[3] = (xy_rounded[2] + 1);
      if ((((xy_rounded[0] < 0) || (height_width[0] <= xy_rounded[0])) ||
           (xy_rounded[2] < 0)) ||
          (height_width[1] <= xy_rounded[2])) {
        for (int ii_d_1 = 0; ii_d_1 < 4; ++ii_d_1) {
          corner_values[ii_d_1] = 0.000000e+00f;
        }
      } else {
        for (int ii_d_2 = 0; ii_d_2 < 4; ++ii_d_2) {
          corner_values[ii_d_2] =
              value_[(((((((((int)blockIdx.y) * 26830848) +
                           (value_level_start_index_[i] * 2048)) +
                          ((xy_rounded[0] * height_width[1]) * 2048)) +
                         (xy_rounded[2] * 2048)) +
                        (((int)blockIdx.z) * 256)) +
                       (((int)threadIdx.x) * 4)) +
                      ii_d_2)];
        }
      }
      if ((((xy_rounded[0] < 0) || (height_width[0] <= xy_rounded[0])) ||
           (xy_rounded[3] < 0)) ||
          (height_width[1] <= xy_rounded[3])) {
        for (int ii_d_3 = 0; ii_d_3 < 4; ++ii_d_3) {
          corner_values[(ii_d_3 + 4)] = 0.000000e+00f;
        }
      } else {
        for (int ii_d_4 = 0; ii_d_4 < 4; ++ii_d_4) {
          corner_values[(ii_d_4 + 4)] =
              value_[(((((((((int)blockIdx.y) * 26830848) +
                           (value_level_start_index_[i] * 2048)) +
                          ((xy_rounded[0] * height_width[1]) * 2048)) +
                         (xy_rounded[3] * 2048)) +
                        (((int)blockIdx.z) * 256)) +
                       (((int)threadIdx.x) * 4)) +
                      ii_d_4)];
        }
      }
      if ((((xy_rounded[1] < 0) || (height_width[0] <= xy_rounded[1])) ||
           (xy_rounded[2] < 0)) ||
          (height_width[1] <= xy_rounded[2])) {
        for (int ii_d_5 = 0; ii_d_5 < 4; ++ii_d_5) {
          corner_values[(ii_d_5 + 8)] = 0.000000e+00f;
        }
      } else {
        for (int ii_d_6 = 0; ii_d_6 < 4; ++ii_d_6) {
          corner_values[(ii_d_6 + 8)] =
              value_[(((((((((int)blockIdx.y) * 26830848) +
                           (value_level_start_index_[i] * 2048)) +
                          ((xy_rounded[1] * height_width[1]) * 2048)) +
                         (xy_rounded[2] * 2048)) +
                        (((int)blockIdx.z) * 256)) +
                       (((int)threadIdx.x) * 4)) +
                      ii_d_6)];
        }
      }
      if ((((xy_rounded[1] < 0) || (height_width[0] <= xy_rounded[1])) ||
           (xy_rounded[3] < 0)) ||
          (height_width[1] <= xy_rounded[3])) {
        for (int ii_d_7 = 0; ii_d_7 < 4; ++ii_d_7) {
          corner_values[(ii_d_7 + 12)] = 0.000000e+00f;
        }
      } else {
        for (int ii_d_8 = 0; ii_d_8 < 4; ++ii_d_8) {
          corner_values[(ii_d_8 + 12)] =
              value_[(((((((((int)blockIdx.y) * 26830848) +
                           (value_level_start_index_[i] * 2048)) +
                          ((xy_rounded[1] * height_width[1]) * 2048)) +
                         (xy_rounded[3] * 2048)) +
                        (((int)blockIdx.z) * 256)) +
                       (((int)threadIdx.x) * 4)) +
                      ii_d_8)];
        }
      }
      for (int ii_d_9 = 0; ii_d_9 < 4; ++ii_d_9) {
        attention_sum[ii_d_9] =
            (attention_sum[ii_d_9] +
             ((((((corner_values[ii_d_9] *
                   (((float)xy_rounded[1]) - xy_grid[0])) *
                  (((float)xy_rounded[3]) - xy_grid[1])) +
                 ((corner_values[(ii_d_9 + 8)] *
                   (xy_grid[0] - ((float)xy_rounded[0]))) *
                  (((float)xy_rounded[3]) - xy_grid[1]))) +
                ((corner_values[(ii_d_9 + 4)] *
                  (((float)xy_rounded[1]) - xy_grid[0])) *
                 (xy_grid[1] - ((float)xy_rounded[2])))) +
               ((corner_values[(ii_d_9 + 12)] *
                 (xy_grid[0] - ((float)xy_rounded[0]))) *
                (xy_grid[1] - ((float)xy_rounded[2])))) *
              attention_weights_[(
                  ((((((int)blockIdx.y) * 12800) + (((int)blockIdx.x) * 128)) +
                    (((int)blockIdx.z) * 16)) +
                   (i * 4)) +
                  k)]));
      }
    }
  }
  for (int ii_d_10 = 0; ii_d_10 < 4; ++ii_d_10) {
    output_[(((((((int)blockIdx.y) * 204800) + (((int)blockIdx.x) * 2048)) +
               (((int)blockIdx.z) * 256)) +
              (((int)threadIdx.x) * 4)) +
             ii_d_10)] = attention_sum[ii_d_10];
  }
}

extern "C" void deformable_kernel(float *value, int *value_spatial_shapes,
                                  int *level_start_index,
                                  float *sampling_locations,
                                  float *attention_weights, float *output) {

  float *d_value, *d_sampling_locations, *d_attention_weights, *d_output;
  int *d_value_spatial_shapes, *d_value_level_start_index;
  int n = 4;
  int l = 4;
  int lq = 100;
  int s = 13101;
  int m = 8;
  int d = 256;
  int p = 4;

  hipMalloc(&d_value, n * s * m * d * sizeof(float));
  hipMalloc(&d_value_spatial_shapes, l * 2 * sizeof(int));
  hipMalloc(&d_value_level_start_index, l * sizeof(int));
  hipMalloc(&d_sampling_locations, n * lq * m * l * p * 2 * sizeof(float));
  hipMalloc(&d_attention_weights, n * lq * m * l * p * sizeof(float));
  hipMalloc(&d_output, n * lq * m * d * sizeof(float));

  hipMemcpy(d_value, value, n * s * m * d * sizeof(float),
            hipMemcpyHostToDevice);
  hipMemcpy(d_value_spatial_shapes, value_spatial_shapes, l * 2 * sizeof(int),
            hipMemcpyHostToDevice);
  hipMemcpy(d_value_level_start_index, level_start_index, l * sizeof(int),
            hipMemcpyHostToDevice);
  hipMemcpy(d_sampling_locations, sampling_locations,
            n * lq * m * l * p * 2 * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_attention_weights, attention_weights,
            n * lq * m * l * p * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(d / 4);
  dim3 numBlocks(lq, n, m);

  deformable<<<numBlocks, blockSize>>>(
      d_attention_weights, d_output, d_sampling_locations, d_value,
      d_value_level_start_index, d_value_spatial_shapes);

  hipMemcpy(output, d_output, n * lq * m * d * sizeof(float),
            hipMemcpyDeviceToHost);

  hipFree(d_value);
  hipFree(d_value_spatial_shapes);
  hipFree(d_value_level_start_index);
  hipFree(d_sampling_locations);
  hipFree(d_attention_weights);
  hipFree(d_output);
}