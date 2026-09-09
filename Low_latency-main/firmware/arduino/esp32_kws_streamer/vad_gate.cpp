#include "vad_gate.h"
#include <cmath>

float VADGate::s_threshold = RMS_SILENCE_THRESHOLD;
float VADGate::s_ambient_floor = 100.0f;

void VADGate::init(float threshold) {
    s_threshold = threshold;
    s_ambient_floor = 100.0f;
    Serial.printf("[VAD] Energy gate initialized with RMS threshold: %.1f\n", s_threshold);
}

float VADGate::calculateRMS(const int16_t* pcm_data, size_t num_samples) {
    if (num_samples == 0) return 0.0f;

    // Fast 64-bit integer energy accumulator to avoid floating point overhead in the hot loop
    int64_t sum_squares = 0;
    for (size_t i = 0; i < num_samples; i++) {
        int32_t val = (int32_t)pcm_data[i];
        sum_squares += (val * val);
    }

    double mean_square = (double)sum_squares / (double)num_samples;
    return (float)sqrt(mean_square);
}

bool VADGate::isVoiceActive(const int16_t* pcm_data, size_t num_samples, float* out_rms) {
    float rms = calculateRMS(pcm_data, num_samples);
    if (out_rms) {
        *out_rms = rms;
    }

    if (rms < s_threshold) {
        // Adapt ambient noise floor slowly during silence
        updateAmbientFloor(rms);
        return false;
    }
    return true;
}

void VADGate::updateAmbientFloor(float current_rms) {
    // Exponential moving average for baseline ambient floor tracking
    s_ambient_floor = 0.95f * s_ambient_floor + 0.05f * current_rms;
}

float VADGate::getThreshold() {
    return s_threshold;
}

void VADGate::setThreshold(float new_threshold) {
    s_threshold = new_threshold;
}
