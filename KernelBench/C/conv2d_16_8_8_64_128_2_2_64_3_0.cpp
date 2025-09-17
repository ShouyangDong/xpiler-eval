extern "C" void conv2d(float *input, float *kernel, float *output) {
  int batch_size = 16;
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

  // 遍历批次、输出特征图的高度和宽度
  for (int bs = 0; bs < batch_size; bs++) {
    for (int oh = 0; oh < output_height; oh++) {
      for (int ow = 0; ow < output_width; ow++) {
        for (int oc = 0; oc < output_channels; oc++) {
          int32_t sum = 0;

          // 定义临时数组来存储64个元素，用于AVX-512指令
          int8_t input_block[64];
          int8_t kernel_block[64];

          // 遍历卷积核的高度和宽度
          for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
              for (int ic = 0; ic < input_channels;
                   ic += 64) { // 每次处理64个通道
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;

                // 边界检查，确保卷积不会越界
                if (ih >= input_height || iw >= input_width) {
                  continue;
                }

                // 直接将浮点数转换为int8并填充到块中
                for (int i = 0; i < 64; i++) {
                  if (ic + i < input_channels) {
                    input_block[i] = static_cast<int8_t>(
                        input[bs * input_height * input_width * input_channels +
                              ih * input_width * input_channels +
                              iw * input_channels + (ic + i)]);
                    kernel_block[i] = static_cast<int8_t>(
                        kernel[oc * kernel_height * kernel_width *
                                   input_channels +
                               kh * kernel_width * input_channels +
                               kw * input_channels + (ic + i)]);
                  } else {
                    input_block[i] = 0; // 如果超过通道数，用0填充
                    kernel_block[i] = 0;
                  }
                }

                // 使用AVX-512指令进行点积计算
                __m512i _input = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i *>(input_block));
                __m512i _kernel = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i *>(kernel_block));
                __m512i acc = _mm512_setzero_si512();
                acc = _mm512_dpbusd_epi32(acc, _input, _kernel); // 执行乘加操作

                // 将累加结果存储到sum中
                int32_t
                    acc_result[16]; // AVX-512点积累加的结果会有16个int32的值
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(acc_result),
                                    acc);
                for (int i = 0; i < 16; i++) {
                  sum += acc_result[i];
                }
              }
            }
          }

          // 将计算的结果存储到输出中
          output[bs * output_height * output_width * output_channels +
                 oh * output_width * output_channels + ow * output_channels +
                 oc] = static_cast<float>(sum);
        }
      }
    }
  }
}
