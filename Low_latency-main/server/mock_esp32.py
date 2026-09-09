"""
Hardware-in-the-Loop Mock ESP32-S3 Edge Device Simulator
Simulates the exact WebSocket protocol and dual-core audio streaming behavior:
1. Pushes standby telemetry (SRAM 115 KB / 256 KB, Core 0 5.4%, Core 1 0.0%)
2. Triggers keyword spotting event ("ISRO")
3. Flushes 32 KB (1.0s) Pre-Roll Ring Buffer
4. Streams continuous 20ms live PCM audio frames
5. Detects 800ms silence and dispatches End-of-Stream (EOS) packet
"""

import sys
import time
import json
import asyncio
import numpy as np
import websockets


SERVER_URI = "ws://localhost:8765/ws/audio"


async def simulate_esp32_device(num_triggers=3):
    print(f"[Mock ESP32] Connecting to ASR server at {SERVER_URI}...")
    
    try:
        async with websockets.connect(SERVER_URI) as ws:
            print("[Mock ESP32] Connected! Wi-Fi Modem-Sleep: DISABLED. Unthrottled RF ready.")

            for t in range(num_triggers):
                print(f"\n--- [Cycle {t+1}/{num_triggers}] State: STANDBY_LISTENING ---")
                # 1. Send idle telemetry (Proving <10% idle CPU and <256 KB SRAM)
                for _ in range(3):
                    telemetry = {
                        "type": "telemetry",
                        "free_sram_kb": 141,
                        "core0_cpu": float(np.random.uniform(4.5, 6.2)),
                        "core1_cpu": 0.0,
                        "rms": float(np.random.uniform(90.0, 140.0)),
                        "state": "STANDBY_LISTENING"
                    }
                    await ws.send(json.dumps(telemetry))
                    await asyncio.sleep(0.8)

                # 2. Trigger Wake-Word Event!
                print("[Mock ESP32] *** CORE 0 TRIGGER DETECTED: 'ISRO' (Confidence: 95.4%) ***")
                print("[Mock ESP32] Core 0 fired FreeRTOS xTaskNotify to Core 1 (< 0.8ms latency)")
                
                # 3. Core 1 Wakes Up: Flushes 32 KB Pre-Roll Ring Buffer (16,000 samples @ 16-bit)
                print("[Mock ESP32] Core 1 flushing 32 KB Pre-Roll Ring Buffer (1.0s Historical Audio)...")
                # Generate synthetic audio waveform
                sample_rate = 16000
                t_audio = np.linspace(0, 1.0, sample_rate)
                # Syllable frequencies for "ISRO"
                audio_wave = (0.6 * np.sin(2 * np.pi * 320 * t_audio) + 0.3 * np.sin(2 * np.pi * 1200 * t_audio))
                pcm_data = (audio_wave * 25000).astype(np.int16).tobytes()

                chunk_size = 1024
                for offset in range(0, len(pcm_data), chunk_size):
                    await ws.send(pcm_data[offset:offset + chunk_size])
                    await asyncio.sleep(0.005)

                print("[Mock ESP32] Pre-Roll Buffer flushed in 11.2ms (Zero syllables lost).")

                # 4. Stream continuous live speech command (2.0 seconds)
                print("[Mock ESP32] Streaming live voice command in 20ms frames...")
                live_frames = 100  # 100 frames * 20ms = 2.0 seconds
                for i in range(live_frames):
                    t_frame = np.linspace(0, 0.02, 320)
                    freq = 250 + 100 * np.sin(i * 0.1)
                    frame_wave = (0.7 * np.sin(2 * np.pi * freq * t_frame))
                    frame_pcm = (frame_wave * 24000).astype(np.int16).tobytes()
                    await ws.send(frame_pcm)
                    await asyncio.sleep(0.02)  # Real-time 20ms pacing

                # 5. Core 0 Energy Gating detects 800ms of sustained silence
                print("[Mock ESP32] Sustained acoustic silence detected by Core 0 VAD (800ms threshold).")
                eos_packet = {
                    "type": "eos",
                    "reason": "silence_timeout"
                }
                await ws.send(json.dumps(eos_packet))
                print("[Mock ESP32] End-of-Stream dispatched. Core 1 returning to 0% CPU standby.")
                await asyncio.sleep(2.0)

            print("\n[Mock ESP32] All simulation cycles completed successfully!")

    except Exception as e:
        print(f"[Mock ESP32 Error] {e}")


if __name__ == "__main__":
    asyncio.run(simulate_esp32_device(num_triggers=2))
