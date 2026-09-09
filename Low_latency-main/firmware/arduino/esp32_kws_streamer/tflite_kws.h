#ifndef TFLITE_KWS_H_
#define TFLITE_KWS_H_

#include <Arduino.h>
#include <cstdint>
#include "config.h"

class TFLiteKWS {
public:
    static bool init();

    // Runs Mel-spectrogram feature extraction on 1.0s (16,000 samples) PCM buffer
    // and executes INT8 neural network inference.
    // Returns confidence probability for the custom keyword "ISRO" (0.0f to 1.0f).
    static float runInference(const int16_t* pcm_1s);

    // Fast Mel-spectrogram generator on 1.0s audio window
    static void extractFeatures(const int16_t* pcm_1s, int8_t* out_spectrogram);

    static bool isInitialized();

private:
    static bool s_initialized;
    // Tensor arena statically allocated in internal SRAM (<256 KB constraint)
    static uint8_t s_tensor_arena[TFLITE_ARENA_SIZE];
    static float s_input_scale;
    static int32_t s_input_zero_point;
};

#endif // TFLITE_KWS_H_
