#ifndef WIFI_STREAMER_H_
#define WIFI_STREAMER_H_

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include "config.h"

// Core 1 notification flags
#define NOTIFY_TRIGGER_START   (1 << 0)
#define NOTIFY_STREAM_CHUNK    (1 << 1)
#define NOTIFY_TRIGGER_STOP    (1 << 2)

class WiFiStreamer {
public:
    static bool initWiFi();
    static bool connectWebSocket();
    static bool isConnected();

    // Sends binary audio frame over WebSocket
    static bool sendBinaryFrame(const uint8_t* data, size_t length);

    // Sends JSON telemetry packet to host server
    static bool sendTelemetry(uint32_t free_sram_kb, float core0_cpu, float core1_cpu, float current_rms, const char* state);

    // Flushes the entire 32 KB historical pre-roll buffer in chunks
    static void flushPreRollBuffer(const int16_t* unrolled_pcm, size_t total_samples);

    // Service WebSocket event loop
    static void poll();

    // Send End-of-Stream packet
    static void sendEndOfStream();

private:
    static bool s_connected;
    static WiFiClient s_tcp_client;
};

#endif // WIFI_STREAMER_H_
