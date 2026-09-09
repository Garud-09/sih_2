#ifndef AUDIO_DMA_H_
#define AUDIO_DMA_H_

#include <Arduino.h>
#include <driver/i2s.h>
#include "config.h"

class AudioDMA {
public:
    static esp_err_t init();
    
    // Reads raw 32-bit words from I2S hardware DMA FIFO
    static size_t readRawDMA(int32_t* out_buffer, size_t num_samples, TickType_t wait_ticks = portMAX_DELAY);
    
    // Signal conditioning: >> 14 bit-shift + soft limiting to prevent harmonic clipping distortion
    static void conditionAudio(const int32_t* raw_in, int16_t* pcm_out, size_t num_samples);
    
    // Writes conditioned 16-bit PCM frames into static 32 KB circular array
    static void pushToRingBuffer(const int16_t* pcm_in, size_t num_samples);
    
    // Edge-case 1 mitigation: Atomically snapshots current write index at trigger moment
    static uint32_t freezeSnapshot();
    
    // Reads ordered 1.0s historical audio ending at frozen snapshot index
    static void copyHistoricalSnapshot(uint32_t snapshot_head, int16_t* destination);

    // Direct access to the static ring buffer memory
    static const int16_t* getRawRingBuffer();

private:
    // Statically allocated in internal SRAM (32,000 bytes = 16,000 samples)
    static int16_t s_ring_buffer[PRE_ROLL_SAMPLES];
    static volatile uint32_t s_write_head;
};

#endif // AUDIO_DMA_H_
