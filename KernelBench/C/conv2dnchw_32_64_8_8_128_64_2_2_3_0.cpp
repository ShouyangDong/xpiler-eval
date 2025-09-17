extern "C" void conv2dnchw(float *input, float *kernel, float *output) {
  int batch_size = 32;
  int input_height = 8;
  int input_width = 8;
  int input_channels = 64;
  int kernel_height = 2;
  int kernel_width = 2;
  int output_channels = 128;
  int stride = 3;
  int padding = 0;
  int output_height = (input_height - kernel_height) / stride + 1;
  int output_width = (input_width - kernel_width) / stride + 1;

  // 遍历批次、输出通道、高度和宽度
  for (int bs = 0; bs < batch_size; bs++) {
    for (int oc = 0; oc < output_channels; oc++) {
      for (int oh = 0; oh < output_height; oh++) {
        for (int ow = 0; ow < output_width; ow++) {
          int32_t sum = 0; // 使用int32_t来存储累加结果

          // 每次遍历64个通道，以便使用AVX-512的向量化能力
          for (int ic = 0; ic < input_channels; ic += 64) {
            int8_t input_block[64];
            int8_t kernel_block[64];

            // 遍历卷积核的高度和宽度
            for (int kh = 0; kh < kernel_height; kh++) {
              for (int kw = 0; kw < kernel_width; kw++) {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;

                // 边界检查，确保卷积不会越界
                if (ih >= input_height || iw >= input_width) {
                  continue;
                }

                // 填充数据块
                for (int i = 0; i < 64; i++) {
                  if (ic + i < input_channels) {
                    input_block[i] = static_cast<int8_t>(roundf(
                        input[bs * input_channels * input_height * input_width +
                              (ic + i) * input_height * input_width +
                              ih * input_width + iw]));
                    kernel_block[i] = static_cast<int8_t>(
                        roundf(kernel[oc * kernel_height * kernel_width *
                                          input_channels +
                                      (ic + i) * kernel_height * kernel_width +
                                      kh * kernel_width + kw]));
                  } else {
                    input_block[i] = 0; // 若超过通道数，用0填充
                    kernel_block[i] = 0;
                  }
                }

                // 使用AVX-512进行点积计算
                __m512i _input = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i *>(input_block));
                __m512i _kernel = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i *>(kernel_block));
                __m512i acc = _mm512_setzero_si512();
                acc = _mm512_dpbusd_epi32(acc, _input, _kernel); // 执行乘加操作

                // 将累加结果存储到sum中
                int32_t acc_result[16]; // AVX-512每个acc包含16个int32的值
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(acc_result),
                                    acc);
                for (int i = 0; i < 16; i++) {
                  sum += acc_result[i];
                }
              }
            }
          }

          // 将计算的结果存储到输出中
          output[bs * output_channels * output_height * output_width +
                 oc * output_height * output_width + oh * output_width + ow] =
              static_cast<float>(sum);
        }
      }
    }
  }
}
