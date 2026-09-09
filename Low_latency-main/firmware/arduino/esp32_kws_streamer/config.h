#ifndef CONFIG_H_
#define CONFIG_H_

#include <Arduino.h>
#include <driver/i2s.h>
#include <driver/gpio.h>

// =========================================================================
// 1. NETWORK CREDENTIALS & SERVER ADDRESS
// =========================================================================
// Replace these with your local Wi-Fi router credentials and host server IP
#define WIFI_SSID           "Jeevan"
#define WIFI_PASSWORD        "PASSWORD"
#define SERVER_HOST         "10.118.168.1"   // IP of laptop running asr_server.py
#define SERVER_PORT         8765       // WebSocket port
#define SERVER_PATH         "/ws/audio"

// =========================================================================
// 2. HARDWARE I2S PIN ASSIGNMENTS (INMP441 / SPH0645 -> ESP32-S3)
// =========================================================================
#define I2S_PORT            I2S_NUM_0
#define I2S_SCK_PIN         GPIO_NUM_6   // Continuous Serial Clock (BCLK)
#define I2S_WS_PIN          GPIO_NUM_5   // Word Select / L/R Clock (WS / LRCLK)
#define I2S_SD_PIN          GPIO_NUM_4   // Serial Data In (SD)

// =========================================================================
// 3. AUDIO PIPELINE SPECIFICATIONS
// =========================================================================
#define AUDIO_SAMPLE_RATE   16000        // 16 kHz sampling rate
#define DMA_BUFFER_SAMPLES  256          // 16ms DMA block (256 samples)
#define DMA_BUFFER_BYTES    (DMA_BUFFER_SAMPLES * sizeof(int32_t)) // Raw 24-bit in 32-bit slot

// 1.0 Second Pre-Roll Circular Buffer (16,000 samples @ 16-bit = 32,000 bytes)
#define PRE_ROLL_SAMPLES    16000
#define PRE_ROLL_BYTES      (PRE_ROLL_SAMPLES * sizeof(int16_t))

// Frame specs for Feature Extraction
#define FRAME_LEN_SAMPLES   480          // 30ms window
#define FRAME_STRIDE_SAMPLES 320         // 20ms stride
#define NUM_MEL_BANDS       40           // 40 Mel filterbank channels
#define SPECTROGRAM_FRAMES  49           // 49 frames per 1.0s

// =========================================================================
// 4. ENERGY GATING (VAD) & TRIGGER THRESHOLDS
// =========================================================================
#define RMS_SILENCE_THRESHOLD       120.0f  // Below this RMS, skip TinyML inference
#define KWS_CONFIDENCE_THRESHOLD    0.20f   // Wake-word confidence required to trigger (>80%)
#define SILENCE_TIMEOUT_MS          800     // 800ms sustained silence terminates streaming

// =========================================================================
// 5. STATIC MEMORY BOUNDS (ISRO PS 26172 COMPLIANCE)
// =========================================================================
#define TFLITE_ARENA_SIZE   (48 * 1024)     // 48 KB Tensor Arena in internal SRAM
#define TASK_STACK_CORE0    (8 * 1024)      // 6 KB stack for Core 0 Sentinel
#define TASK_STACK_CORE1    (6 * 1024)      // 6 KB stack for Core 1 Streamer

#endif // CONFIG_H_
