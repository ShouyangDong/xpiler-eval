
extern "C" void mha(float *Q, float *K, float *V, float *output) {
  float score[6 * 6];
  // The dimension 1, 4096, 6, 256
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 4096; j++) {
      for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
          score[m * 6 + n] = 0.0;
          for (int p = 0; p < 256; p++) {
            score[m * 6 + n] +=
                Q[i * 4096 * 6 * 256 + j * 6 * 256 + m * 256 + p] *
                K[i * 4096 * 6 * 256 + j * 6 * 256 + n * 256 + p];
          }
        }
      }

      // score
      for (int m_sc = 0; m_sc < 6; m_sc++) {
        for (int n_sc = 0; n_sc < 6; n_sc++) {
          score[m_sc * 6 + n_sc] = score[m_sc * 6 + n_sc] / sqrt(256);
        }
      }

      // The Softmax code:
      for (int j_sf = 0; j_sf < 6; ++j_sf) {
        float sum = 0;

        for (int i_ex = 0; i_ex < 6; ++i_ex) {
          score[j_sf * 6 + i_ex] = expf(score[j_sf * 6 + i_ex]);
        }
        for (int i_sf = 0; i_sf < 6; ++i_sf) {
          sum += score[j_sf * 6 + i_sf];
        }
        for (int k_sf = 0; k_sf < 6; ++k_sf) {
          score[j_sf * 6 + k_sf] = score[j_sf * 6 + k_sf] / sum;
        }
      }

      // The final Matmul
      for (int m_fl = 0; m_fl < 6; ++m_fl) {
        for (int n_fl = 0; n_fl < 256; ++n_fl) {
          output[i * 4096 * 6 * 256 + j * 6 * 256 + m_fl * 256 + n_fl] = 0.0;
          for (int k_fl = 0; k_fl < 6; ++k_fl) {
            output[i * 4096 * 6 * 256 + j * 6 * 256 + m_fl * 256 + n_fl] +=
                score[m_fl * 6 + k_fl] *
                V[i * 4096 * 6 * 256 + j * 6 * 256 + k_fl * 256 + n_fl];
          }
        }
      }
    }
  }
}