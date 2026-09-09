#include "wifi_streamer.h"
#include <cstring>
#include <algorithm>
#include "esp_heap_caps.h"

#define SAFE_MIN(a, b) ((a) < (b) ? (a) : (b))

bool WiFiStreamer::s_connected = false;
WiFiClient WiFiStreamer::s_tcp_client;

bool WiFiStreamer::initWiFi() {
    Serial.printf("[WiFi] Connecting to %s...\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 25) {
        delay(300);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\n[WiFi] Connection timed out. Running in offline sentinel mode.");
        return false;
    }

    // =========================================================================
    // Edge Case 2 Mitigation: Disable Wi-Fi modem sleep!
    // Prevents 50ms - 120ms radio wake delays, keeping socket latency < 15ms
    // =========================================================================
    WiFi.setSleep(false);
    esp_wifi_set_ps(WIFI_PS_NONE);

    Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.println("[WiFi] Wi-Fi Modem-Sleep DISABLED (Locked to zero-latency RF state).");
    return true;
}

bool WiFiStreamer::connectWebSocket() {
    if (!WiFi.isConnected()) {
        s_connected = false;
        return false;
    }

    if (s_tcp_client.connected()) {
        s_connected = true;
        return true;
    }

    Serial.printf("[Socket] Connecting to ASR Server at %s:%d...\n", SERVER_HOST, SERVER_PORT);
    if (!s_tcp_client.connect(SERVER_HOST, SERVER_PORT)) {
        Serial.println("[Socket] TCP connection failed. Will retry on next cycle.");
        s_connected = false;
        return false;
    }

    // RFC 6455 WebSocket Client Handshake
    String handshake = String("GET ") + SERVER_PATH + " HTTP/1.1\r\n" +
                       "Host: " + SERVER_HOST + ":" + String(SERVER_PORT) + "\r\n" +
                       "Upgrade: websocket\r\n" +
                       "Connection: Upgrade\r\n" +
                       "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" +
                       "Sec-WebSocket-Version: 13\r\n\r\n";

    s_tcp_client.print(handshake);

    // Read handshake response
    unsigned long timeout = millis() + 2000;
    bool success = false;
    String response = "";

    while (millis() < timeout) {
        while (s_tcp_client.available()) {
            char c = s_tcp_client.read();
            response += c;
            if (response.indexOf("101") >= 0) {
                success = true;
            }
            if (response.endsWith("\r\n\r\n")) {
                goto handshake_done;
            }
        }
        delay(5);
    }
handshake_done:

    if (success) {
        s_connected = true;
        Serial.println("[Socket] WebSocket handshake verified! Socket is pre-warmed and ready.");
        // Send initial telemetry packet
        sendTelemetry(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024, 5.0f, 0.0f, 150.0f, "STANDBY_LISTENING");
    } else {
        Serial.println("[Socket] WebSocket handshake failed.");
        s_tcp_client.stop();
        s_connected = false;
    }

    return s_connected;
}

bool WiFiStreamer::isConnected() {
    return s_connected && s_tcp_client.connected();
}

bool WiFiStreamer::sendBinaryFrame(const uint8_t* data, size_t length) {
    if (!isConnected()) {
        return false;
    }

    // RFC 6455: Client-to-server frames must be masked
    // Byte 0: 0x82 (FIN=1, Opcode=2 for Binary)
    uint8_t header[10];
    size_t header_len = 0;

    header[0] = 0x82;

    if (length < 126) {
        header[1] = 0x80 | (uint8_t)length; // Mask bit set
        header_len = 2;
    } else if (length <= 65535) {
        header[1] = 0x80 | 126;
        header[2] = (length >> 8) & 0xFF;
        header[3] = length & 0xFF;
        header_len = 4;
    } else {
        return false;
    }

    // 4-byte masking key
    uint8_t mask[4] = {0x12, 0x34, 0x56, 0x78};
    memcpy(&header[header_len], mask, 4);
    header_len += 4;

    // Send header
    s_tcp_client.write(header, header_len);

    // Apply mask and send payload in chunks
    uint8_t buffer[512];
    size_t sent = 0;
    while (sent < length) {
        size_t chunk = SAFE_MIN((size_t)sizeof(buffer), length - sent);
        for (size_t i = 0; i < chunk; i++) {
            buffer[i] = data[sent + i] ^ mask[(sent + i) % 4];
        }
        s_tcp_client.write(buffer, chunk);
        sent += chunk;
    }

    return true;
}

bool WiFiStreamer::sendTelemetry(uint32_t free_sram_kb, float core0_cpu, float core1_cpu, float current_rms, const char* state) {
    if (!isConnected()) return false;

    char json_buf[256];
    snprintf(json_buf, sizeof(json_buf),
             "{\"type\":\"telemetry\",\"free_sram_kb\":%u,\"core0_cpu\":%.1f,\"core1_cpu\":%.1f,\"rms\":%.1f,\"state\":\"%s\"}",
             free_sram_kb, core0_cpu, core1_cpu, current_rms, state);

    size_t len = strlen(json_buf);
    uint8_t header[6];
    header[0] = 0x81; // FIN=1, Opcode=1 for Text
    header[1] = 0x80 | (uint8_t)len;
    uint8_t mask[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    memcpy(&header[2], mask, 4);

    s_tcp_client.write(header, 6);

    uint8_t masked_text[256];
    for (size_t i = 0; i < len; i++) {
        masked_text[i] = json_buf[i] ^ mask[i % 4];
    }
    s_tcp_client.write(masked_text, len);
    return true;
}

void WiFiStreamer::flushPreRollBuffer(const int16_t* unrolled_pcm, size_t total_samples) {
    if (!isConnected()) return;

    Serial.printf("[Socket] Flushing 1.0s Pre-Roll Buffer (%d bytes) over WebSocket...\n",
                  (int)(total_samples * sizeof(int16_t)));

    // Stream 1.0s history in 1024-byte (512 samples) chunks
    const uint8_t* byte_ptr = (const uint8_t*)unrolled_pcm;
    size_t remaining_bytes = total_samples * sizeof(int16_t);

    while (remaining_bytes > 0) {
        size_t chunk_size = SAFE_MIN((size_t)1024, remaining_bytes);
        sendBinaryFrame(byte_ptr, chunk_size);
        byte_ptr += chunk_size;
        remaining_bytes -= chunk_size;
    }
}

void WiFiStreamer::sendEndOfStream() {
    if (!isConnected()) return;

    const char* eos_msg = "{\"type\":\"eos\",\"reason\":\"silence_timeout\"}";
    size_t len = strlen(eos_msg);
    uint8_t header[6];
    header[0] = 0x81;
    header[1] = 0x80 | (uint8_t)len;
    uint8_t mask[4] = {0x55, 0x66, 0x77, 0x88};
    memcpy(&header[2], mask, 4);

    s_tcp_client.write(header, 6);

    uint8_t masked[128];
    for (size_t i = 0; i < len; i++) {
        masked[i] = eos_msg[i] ^ mask[i % 4];
    }
    s_tcp_client.write(masked, len);
    Serial.println("[Socket] End-of-Stream (EOS) packet dispatched to server.");
}

void WiFiStreamer::poll() {
    // Flush any incoming server control frames / ping-pongs
    while (s_tcp_client.available()) {
        s_tcp_client.read();
    }
}
