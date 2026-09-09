"""
Audio feature extraction module for Keyword Spotting (KWS).
Calculates 40-channel Mel-filterbank spectrogram matching the fixed-point DSP
implementation on the ESP32-S3 (esp-dsp Mel filterbank).

Specifications:
- Audio Sample Rate: 16000 Hz
- Frame Length: 30ms (480 samples)
- Frame Stride: 20ms (320 samples)
- FFT Size: 512
- Mel Bands: 40 (20 Hz to 8000 Hz)
- Frames per 1.0s: 49 frames -> Spectrogram Shape: (49, 40)
"""

import numpy as np


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(num_filters=40, fft_size=512, sample_rate=16000, low_freq=20.0, high_freq=8000.0):
    """Generates triangular Mel filterbank matrix of shape (num_filters, fft_size // 2 + 1)."""
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    num_bins = fft_size // 2 + 1
    filterbank = np.zeros((num_filters, num_bins), dtype=np.float32)

    for i in range(1, num_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        for j in range(left, center):
            if j < num_bins:
                filterbank[i - 1, j] = (j - left) / max(1, center - left)
        for j in range(center, right):
            if j < num_bins:
                filterbank[i - 1, j] = (right - j) / max(1, right - center)

    # Normalize filterbank weights
    enorm = 2.0 / (hz_points[2:num_filters + 2] - hz_points[:num_filters])
    filterbank *= enorm[:, np.newaxis]
    return filterbank


# Precomputed global filterbank for fast reuse
MEL_FILTERBANK = create_mel_filterbank()
HANNING_WINDOW = np.hanning(480).astype(np.float32)


def extract_mel_spectrogram(audio, sample_rate=16000, frame_len=480, frame_stride=320, fft_size=512, num_filters=40):
    """
    Extracts 40-band log-Mel spectrogram from 1D PCM audio array.
    Expects 1.0 second of audio (16,000 samples).
    Returns numpy array of shape (49, 40) float32.
    """
    if len(audio) < sample_rate:
        # Pad with zeros if shorter than 1.0s
        audio = np.pad(audio, (0, sample_rate - len(audio)), mode='constant')
    elif len(audio) > sample_rate:
        # Trim to 1.0s
        audio = audio[:sample_rate]

    audio = audio.astype(np.float32)
    num_frames = (len(audio) - frame_len) // frame_stride + 1
    spectrogram = np.zeros((num_frames, num_filters), dtype=np.float32)

    for i in range(num_frames):
        start = i * frame_stride
        frame = audio[start:start + frame_len] * HANNING_WINDOW
        
        # 512-point FFT
        fft_result = np.fft.rfft(frame, n=fft_size)
        power_spec = (np.abs(fft_result) ** 2) / fft_size

        # Apply Mel Filterbank
        mel_energy = np.dot(MEL_FILTERBANK, power_spec)
        
        # Log compression with floor to prevent log(0)
        mel_energy = np.maximum(mel_energy, 1e-6)
        spectrogram[i, :] = np.log(mel_energy)

    return spectrogram


def compute_rms(audio):
    """Computes Root-Mean-Square (RMS) energy of audio frame."""
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
