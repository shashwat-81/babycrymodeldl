#include <Arduino.h>         // for millis(), etc.
#include "spo2_algorithm.h"
#include <math.h>

void maxim_heart_rate_and_oxygen_saturation(
    uint32_t *pun_ir_buffer,
    int n_ir_buffer_length,
    uint32_t *pun_red_buffer,
    float *pn_spo2,
    int8_t *pch_spo2_valid,
    int32_t *pn_heart_rate,
    int8_t *pch_hr_valid
) {
    // --- crude placeholder algorithm ---
    // Compute DC (average) and AC (max-min) for IR and Red
    uint32_t ir_min = pun_ir_buffer[0], ir_max = pun_ir_buffer[0];
    uint32_t red_min = pun_red_buffer[0], red_max = pun_red_buffer[0];
    uint64_t ir_sum = 0, red_sum = 0;
    for (int i = 0; i < n_ir_buffer_length; i++) {
      uint32_t ir = pun_ir_buffer[i];
      uint32_t red = pun_red_buffer[i];
      if (ir < ir_min) ir_min = ir;
      if (ir > ir_max) ir_max = ir;
      if (red < red_min) red_min = red;
      if (red > red_max) red_max = red;
      ir_sum += ir;
      red_sum += red;
    }
    float ir_dc = (float)ir_sum / n_ir_buffer_length;
    float red_dc = (float)red_sum / n_ir_buffer_length;
    float ir_ac = (float)(ir_max - ir_min);
    float red_ac = (float)(red_max - red_min);

    // Heart rate dummy (not implemented here)
    *pn_heart_rate = 0;
    *pch_hr_valid = 0;

    // SpO2: ratio-of-ratios approximate formula
    if (ir_dc > 0 && red_dc > 0 && ir_ac > 0 && red_ac > 0) {
      float ratio = (red_ac / red_dc) / (ir_ac / ir_dc);
      float spo2 = 110.0f - 25.0f * ratio; // crude linear calibration
      if (spo2 < 0) spo2 = 0;
      if (spo2 > 100) spo2 = 100;
      *pn_spo2 = spo2;
      *pch_spo2_valid = 1;
    } else {
      *pn_spo2 = 0.0f;
      *pch_spo2_valid = 0;
    }
}
