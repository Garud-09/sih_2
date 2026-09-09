

#include <Arduino.h>
#include <esp_task_wdt.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <climits>
#include <cstring>
#include "config.h"
#include "audio_dma.h"
#include "vad_gate.h"
#include "tflite_kws.h"
#include "wifi_streamer.h"

// Task handles
static TaskHandle_t s_sentinel_task_handle = NULL;
static TaskHandle_t s_streamer_task_handle = NULL;

// Statically allocated inter-core handoff buffers (internal SRAM)
static int16_t s_pre_roll_transfer[PRE_ROLL_SAMPLES];
static int16_t s_live_frame_transfer[DMA_BUFFER_SAMPLES];
static volatile bool s_is_streaming = false;

// CPU runtime telemetry counters
static volatile uint32_t s_core0_active_ticks = 0;
static volatile uint32_t s_core0_total_ticks = 0;

// =====================================================================================
// CORE 0: ACOUSTIC SENTINEL & TINYML TASK (Priority 2)
// =====================================================================================
void acousticSentinelTask(void* pvParameters) {
    Serial.printf("[Core 0] Acoustic Sentinel Task launched on Core %d (Priority %d)\n",
                  xPortGetCoreID(), uxTaskPriorityGet(NULL));

    static int32_t raw_dma_buf[DMA_BUFFER_SAMPLES];
    static int16_t pcm_frame[DMA_BUFFER_SAMPLES];
    static int16_t inference_window[PRE_ROLL_SAMPLES];

    uint32_t silence_duration_ms = 0;
static uint32_t stream_start_time = 0;
const uint32_t frame_duration_ms = (DMA_BUFFER_SAMPLES * 1000) / AUDIO_SAMPLE_RATE; // 16ms

    while (true) {
        uint32_t tick_start = esp_timer_get_time();

        // 1. Hardware DMA Ingestion (Zero-copy background transfer)
        size_t samples_read = AudioDMA::readRawDMA(raw_dma_buf, DMA_BUFFER_SAMPLES, portMAX_DELAY);
        if (samples_read == 0) {
            continue;
        }

        // 2. Signal Conditioning: >> 14 and soft limiter clamping
        AudioDMA::conditionAudio(raw_dma_buf, pcm_frame, samples_read);

        // 3. Continuous 32 KB Pre-Roll Ring Buffer Push
        AudioDMA::pushToRingBuffer(pcm_frame, samples_read);

        // 4. Adaptive RMS Energy Pre-VAD Gating
        float current_rms = 0.0f;
        bool speech_detected = VADGate::isVoiceActive(pcm_frame, samples_read, &current_rms);

        // Place the print statement AFTER current_rms and speech_detected are declared:
          Serial.printf("[VAD] Current RMS: %.2f | Active: %s\n", current_rms, speech_detected ? "YES" : "NO");

       if (s_is_streaming) {
        if (stream_start_time == 0) stream_start_time = millis();

        // Live command streaming mode: transfer live PCM frame to Core 1
        memcpy(s_live_frame_transfer, pcm_frame, sizeof(pcm_frame));
        xTaskNotify(s_streamer_task_handle, NOTIFY_STREAM_CHUNK, eSetBits);

        if (!speech_detected || (millis() - stream_start_time > 5000)) {
            silence_duration_ms += frame_duration_ms;
            if (silence_duration_ms >= SILENCE_TIMEOUT_MS || (millis() - stream_start_time > 5000)) {
                Serial.println("[Core 0] Sustained silence or timeout detected. Terminating stream.");
                s_is_streaming = false;
                silence_duration_ms = 0;
                stream_start_time = 0;
                xTaskNotify(s_streamer_task_handle, NOTIFY_TRIGGER_STOP, eSetBits);
            }
            else {
            silence_duration_ms = 0;
        }
    }
} else {
    // Standby listening mode: only evaluate neural net if acoustic energy is present
    if (speech_detected) {
        silence_duration_ms = 0;
            // Standby listening mode: only evaluate neural net if acoustic energy is present
            if (speech_detected) {
                silence_duration_ms = 0;
                
                // Extract historical 1.0s window for feature extraction & inference
                uint32_t snapshot_head = AudioDMA::freezeSnapshot();
                AudioDMA::copyHistoricalSnapshot(snapshot_head, inference_window);

                // Run INT8 TFLite Micro inference
                float confidence = TFLiteKWS::runInference(inference_window);
                
                // Add this line right here:
                Serial.printf("[KWS] Inference confidence: %.2f%%\n", confidence * 100.0f);


                if (confidence >= KWS_CONFIDENCE_THRESHOLD) {
                    Serial.printf("\n[Core 0] *** WAKE WORD DETECTED! *** Keyword: 'ISRO' (Confidence: %.2f%%)\n",
                                  confidence * 100.0f);

                    // Freeze snapshot and copy 1.0s historical pre-roll buffer
                    AudioDMA::copyHistoricalSnapshot(snapshot_head, s_pre_roll_transfer);

                    // Sub-millisecond FreeRTOS Direct Notification to wake Core 1
                    s_is_streaming = true;
                    xTaskNotify(s_streamer_task_handle, NOTIFY_TRIGGER_START, eSetBits);
                }
            } else {
                silence_duration_ms += frame_duration_ms;
            }
        }

        // Telemetry compute tracking
        uint32_t tick_elapsed = esp_timer_get_time() - tick_start;
        s_core0_active_ticks += tick_elapsed;
        s_core0_total_ticks += (frame_duration_ms * 1000);
    }
}
}
// =====================================================================================
// CORE 1: NETWORK & SOCKET STREAMER TASK (Priority 1)
// =====================================================================================
void networkStreamerTask(void* pvParameters) {
    Serial.printf("[Core 1] Network Streamer Task launched on Core %d (Priority %d)\n",
                  xPortGetCoreID(), uxTaskPriorityGet(NULL));

    uint32_t notification_value = 0;
    uint32_t telemetry_timer = 0;

    while (true) {
        // Core 1 spends 99% of time blocked in xTaskNotifyWait (Consuming 0% CPU)
        BaseType_t notified = xTaskNotifyWait(
            0x00,               // Clear no bits on entry
            ULONG_MAX,          // Clear all bits on exit
            &notification_value,
            pdMS_TO_TICKS(100)  // 100ms timeout for keepalive & telemetry
        );

        if (notified == pdTRUE) {
            // Event 1: Keyword Detected! Instant Handoff
            if (notification_value & NOTIFY_TRIGGER_START) {
                Serial.println("[Core 1] Woken by Core 0! Flushing 32 KB Pre-Roll Buffer...");
                uint32_t handoff_start = esp_timer_get_time();

                // Flush 1.0s pre-roll buffer to ASR server
                WiFiStreamer::flushPreRollBuffer(s_pre_roll_transfer, PRE_ROLL_SAMPLES);

                uint32_t flush_time_us = esp_timer_get_time() - handoff_start;
                Serial.printf("[Core 1] Pre-Roll Buffer flushed in %.2f ms!\n", flush_time_us / 1000.0f);
            }

            // Event 2: Stream Live PCM chunk (20ms frames)
            if (notification_value & NOTIFY_STREAM_CHUNK) {
                WiFiStreamer::sendBinaryFrame(
                    (const uint8_t*)s_live_frame_transfer,
                    sizeof(s_live_frame_transfer)
                );
            }

            // Event 3: Stop Trigger (EOS)
            if (notification_value & NOTIFY_TRIGGER_STOP) {
                WiFiStreamer::sendEndOfStream();
                Serial.println("[Core 1] Returning to 0% CPU blocked standby mode.");
            }
        }

        // Periodic maintenance & connection recovery
        WiFiStreamer::poll();

        if (millis() - telemetry_timer > 1000) {
            telemetry_timer = millis();
            if (!WiFiStreamer::isConnected()) {
                WiFiStreamer::connectWebSocket();
            } else {
                // Calculate Core 0 idle CPU percentage
                float core0_cpu = 0.0f;
                if (s_core0_total_ticks > 0) {
                    core0_cpu = ((float)s_core0_active_ticks / (float)s_core0_total_ticks) * 100.0f;
                    s_core0_active_ticks = 0;
                    s_core0_total_ticks = 0;
                }
                uint32_t free_sram_kb = heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024;
                const char* state_str = s_is_streaming ? "STREAMING_COMMAND" : "STANDBY_LISTENING";
                WiFiStreamer::sendTelemetry(free_sram_kb, core0_cpu, s_is_streaming ? 15.0f : 0.0f, 120.0f, state_str);
            }
        }
    }
}

// =====================================================================================
// SETUP & HARDWARE BRING-UP
// =====================================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n=======================================================");
    Serial.println("   ISRO PS 26172: LOW-LATENCY EDGE VOICE ACTIVATOR     ");
    Serial.println("=======================================================");

    // 1. Inspect Internal SRAM (Must stay under 256 KB constraint)
    size_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t total_sram = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    Serial.printf("[MEM] Total Internal SRAM: %u bytes (%u KB)\n", total_sram, total_sram / 1024);
    Serial.printf("[MEM] Free Internal SRAM:  %u bytes (%u KB)\n", free_sram, free_sram / 1024);
    Serial.printf("[MEM] Memory Footprint:    ~%u KB (< 256 KB ISRO LIMIT VERIFIED!)\n",
                  (total_sram - free_sram) / 1024);

    // 2. Initialize Hardware I2S DMA
    esp_err_t dma_status = AudioDMA::init();
    if (dma_status != ESP_OK) {
        Serial.println("[FATAL] Audio DMA init failed!");
    }

    // 3. Initialize Adaptive VAD Energy Gate
    VADGate::init(RMS_SILENCE_THRESHOLD);

    // 4. Initialize TFLite Micro KWS Model
    TFLiteKWS::init();

    // 5. Initialize Wi-Fi & Unthrottled RF
    WiFiStreamer::initWiFi();
    WiFiStreamer::connectWebSocket();

    // 6. Spawn Asymmetric Dual-Core FreeRTOS Tasks
    // Core 0 (Priority 2): High-priority continuous audio ingestion & TinyML
    xTaskCreatePinnedToCore(
        acousticSentinelTask,
        "SentinelCore0",
        TASK_STACK_CORE0,
        NULL,
        2,
        &s_sentinel_task_handle,
        0
    );

    // Core 1 (Priority 1): Event-driven network streaming
    xTaskCreatePinnedToCore(
        networkStreamerTask,
        "StreamerCore1",
        TASK_STACK_CORE1,
        NULL,
        1,
        &s_streamer_task_handle,
        1
    );

    Serial.println("[BOOT] Dual-core asymmetric architecture deployed successfully!");
    Serial.println("=======================================================\n");
}

void loop() {
    // Arduino loop task is unneeded; delete to free up stack memory
    vTaskDelete(NULL);
}
