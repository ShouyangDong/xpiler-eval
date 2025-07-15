
extern "C" void layernorm(float *input,  // shape=[1, 4, 8]
                                 float *gamma,  // shape=[8]
                                 float *beta,   // shape=[8]
                                 float *output) // shape=[1, 4, 8]
{
  for (int i_seq = 0; i_seq < 4; i_seq++) {
    float mean = 0.0;
    float variance = 0.0;
    float diff[8];
    // Calculate mean
    for (int i_mean = 0; i_mean < 8; i_mean++) {
      mean += input[i_seq * 8 + i_mean];
    }
    mean /= 8;
    // Calculate variance
    for (int i_diff = 0; i_diff < 8; i_diff++) {
      diff[i_diff] = input[i_seq * 8 + i_diff] - mean;
    }

    for (int i_pow = 0; i_pow < 8; i_pow++) {
      diff[i_pow] = diff[i_pow] * diff[i_pow];
    }
    for (int i_var = 0; i_var < 8; i_var++) {
      variance += diff[i_var];
    }
    variance = sqrt(variance / 8);

    // Normalize input
    for (int i_norm = 0; i_norm < 8; i_norm++) {
      diff[i_norm] = (input[i_seq * 8 + i_norm] - mean);
    }

    for (int i_mul = 0; i_mul < 8; i_mul++) {
      diff[i_mul] = diff[i_mul] * gamma[i_mul];
    }

    for (int i_div = 0; i_div < 8; i_div++) {
      diff[i_div] = diff[i_div] / (variance + 1e-5f);
    }

    for (int i_bet = 0; i_bet < 8; i_bet++) {
      output[i_seq * 8 + i_bet] = diff[i_bet] + beta[i_bet];
    }
  }
}