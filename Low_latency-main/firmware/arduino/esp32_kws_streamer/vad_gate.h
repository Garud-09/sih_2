#ifndef VAD_GATE_H_
#define VAD_GATE_H_

#include <Arduino.h>
#include "config.h"

class VADGate {
public:
    static void init(float threshold = RMS_SILENCE_THRESHOLD);

    // Fast RMS calculation (< 0.2ms runtime)
    static float calculateRMS(const int16_t* pcm_data, size_t num_samples);

    // Returns true if frame energy is above ambient threshold (speech present)
    static bool isVoiceActive(const int16_t* pcm_data, size_t num_samples, float* out_rms = nullptr);

    // Dynamically recalibrate baseline noise floor during quiet periods
    static void updateAmbientFloor(float current_rms);

    static float getThreshold();
    static void setThreshold(float new_threshold);

private:
    static float s_threshold;
    static float s_ambient_floor;
};

#endif // VAD_GATE_H_
