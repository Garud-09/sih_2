#include "audio_dma.h"
#include <cstring>
#include <algorithm>

// Statically allocated 32 KB array in ESP32-S3 internal SRAM (< 256 KB constraint)
int16_t AudioDMA::s_ring_buffer[PRE_ROLL_SAMPLES] = {0};
volatile uint32_t AudioDMA::s_write_head = 0;

esp_err_t AudioDMA::init() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = AUDIO_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = DMA_BUFFER_SAMPLES,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PIN
    };

    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[DMA] Failed to install I2S driver: %d\n", err);
        return err;
    }

    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[DMA] Failed to set I2S pins: %d\n", err);
        return err;
    }

#if !defined(ESP_IDF_VERSION_MAJOR) || (ESP_IDF_VERSION_MAJOR < 5)
    i2s_zero_dma_buffer(I2S_PORT);
#endif
    Serial.println("[DMA] Hardware I2S DMA initialized successfully (16 kHz, 24-bit PCM).");
    return ESP_OK;
}

size_t AudioDMA::readRawDMA(int32_t* out_buffer, size_t num_samples, TickType_t wait_ticks) {
    size_t bytes_read = 0;
    esp_err_t err = i2s_read(I2S_PORT, (void*)out_buffer, num_samples * sizeof(int32_t), &bytes_read, wait_ticks);
    if (err != ESP_OK) {
        return 0;
    }
    return bytes_read / sizeof(int32_t);
}

void AudioDMA::conditionAudio(const int32_t* raw_in, int16_t* pcm_out, size_t num_samples) {
    // Edge Case 3 Mitigation: Soft clamp + scaling to eliminate distortion harmonics
    for (size_t i = 0; i < num_samples; i++) {
        int32_t sample = raw_in[i] >> 14; // Right-shift 24-bit left-justified data to 16-bit
        if (sample > 32767) {
            sample = 32767;
        } else if (sample < -32768) {
            sample = -32768;
        }
        pcm_out[i] = (int16_t)sample;
    }
}

void AudioDMA::pushToRingBuffer(const int16_t* pcm_in, size_t num_samples) {
    uint32_t head = s_write_head;
    for (size_t i = 0; i < num_samples; i++) {
        s_ring_buffer[head] = pcm_in[i];
        head++;
        if (head >= PRE_ROLL_SAMPLES) {
            head = 0;
        }
    }
    s_write_head = head;
}

uint32_t AudioDMA::freezeSnapshot() {
    // Edge Case 1 Mitigation: Lockless instantaneous snapshot of the write pointer
    return s_write_head;
}

void AudioDMA::copyHistoricalSnapshot(uint32_t snapshot_head, int16_t* destination) {
    // Reconstruct continuous chronological 1.0s window:
    // [snapshot_head ... end] followed by [start ... snapshot_head]
    size_t tail_count = PRE_ROLL_SAMPLES - snapshot_head;
    memcpy(destination, &s_ring_buffer[snapshot_head], tail_count * sizeof(int16_t));
    if (snapshot_head > 0) {
        memcpy(destination + tail_count, &s_ring_buffer[0], snapshot_head * sizeof(int16_t));
    }
}

const int16_t* AudioDMA::getRawRingBuffer() {
    return s_ring_buffer;
}
