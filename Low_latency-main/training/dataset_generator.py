"""
Synthetic Dataset Generator with Multi-Factor Acoustic Data Augmentation
Designed for ISRO PS 26172 Wake-Word Spotting ("ISRO")

Features:
- Formant-based acoustic phoneme synthesis (standalone, zero external API keys required)
- Target wake-word: "isro"
- Confuser words: "vayu", "vroom", "viomesh", "vipin", "system", "offline"
- Ambient background noises: Gaussian white, pink, brown, and room reverberation
- Augmentations:
    * Pitch shifting (+-15%)
    * Speed / tempo perturbation (0.85x to 1.15x)
    * Time-shifting within 1.0s (16,000 samples) window
    * Room Impulse Response (RIR) acoustic reverberation
    * Signal-to-Noise Ratio (SNR) mixing from 0 dB to 20 dB
"""

import os
import random
import numpy as np
from scipy import signal
from features import extract_mel_spectrogram


# Phoneme formant table (F1, F2, F3 in Hz)
PHONEME_FORMANTS = {
    'v': [(250, 60), (1000, 100), (2200, 150)],
    'y': [(280, 50), (2250, 100), (3000, 150)],
    'o': [(500, 70), (900, 80), (2400, 120)],
    'm': [(280, 50), (1100, 80), (2300, 120)],
    'a': [(750, 80), (1200, 90), (2500, 120)],
    'u': [(350, 60), (800, 80), (2300, 120)],
    'r': [(450, 60), (1300, 90), (1700, 100)],
    'sh': [(2000, 200), (3500, 300), (5000, 400)],
    'i': [(270, 50), (2300, 100), (3000, 150)],
    'p': [(300, 60), (1200, 100), (2400, 150)],
    'n': [(290, 50), (1200, 80), (2400, 120)],
    's': [(4000, 300), (5500, 400), (7000, 500)],
    't': [(400, 70), (1800, 100), (2800, 150)],
    'e': [(530, 60), (1850, 90), (2500, 120)],
}

WORDS = {
    'isro': ['i', 's', 'r', 'o'],
    'intro': ['i', 'n', 't', 'r', 'o'],
    'astro': ['a', 's', 't', 'r', 'o'],
    'esro': ['e', 's', 'r', 'o'],
    'zero': ['s', 'e', 'r', 'o'],
    'stop': ['s', 't', 'o', 'p'],
    'listen': ['l', 'i', 's', 'e', 'n'],
    'system': ['s', 'i', 's', 't', 'e', 'm'],
}


def synthesize_phoneme(phoneme, duration_s, sample_rate=16000, f0=130.0):
    """Synthesizes a single phoneme using source-filter formant synthesis."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    
    # Glottal source: impulse train + harmonics with jitter
    glottal_pulses = signal.sawtooth(2 * np.pi * f0 * t)
    
    # Unvoiced phonemes use white noise excitation
    if phoneme in ['s', 'sh', 't', 'p']:
        source = np.random.normal(0, 0.4, n_samples)
    else:
        noise_mix = np.random.normal(0, 0.05, n_samples)
        source = 0.9 * glottal_pulses + 0.1 * noise_mix

    formants = PHONEME_FORMANTS.get(phoneme, [(500, 80), (1500, 100), (2500, 150)])
    output = np.zeros(n_samples, dtype=np.float32)

    # Resonate through formants (2nd order bandpass filters)
    for freq, bw in formants:
        q = max(1.0, freq / bw)
        b, a = signal.iirpeak(freq, q, fs=sample_rate)
        filtered = signal.lfilter(b, a, source)
        output += filtered

    # Smooth attack and decay envelope
    envelope = np.hanning(n_samples)
    return output * envelope


def synthesize_word(word_phonemes, sample_rate=16000, f0=130.0, speed_factor=1.0):
    """Synthesizes an entire word by concatenating smoothly modulated phonemes."""
    audio_segments = []
    base_phoneme_dur = 0.12 / speed_factor

    for p in word_phonemes:
        # Slight duration jitter
        dur = base_phoneme_dur * random.uniform(0.85, 1.15)
        # Slight pitch intonation contour
        f0_jittered = f0 * random.uniform(0.95, 1.05)
        seg = synthesize_phoneme(p, dur, sample_rate, f0_jittered)
        audio_segments.append(seg)

    word_audio = np.concatenate(audio_segments)
    # Normalize peak
    max_val = np.max(np.abs(word_audio))
    if max_val > 0:
        word_audio /= max_val
    return word_audio


def apply_reverb(audio, sample_rate=16000, rt60=0.25):
    """Simulates Room Impulse Response (RIR) reverberation."""
    decay_samples = int(rt60 * sample_rate)
    t = np.linspace(0, rt60, decay_samples)
    impulse = np.random.normal(0, 1, decay_samples) * np.exp(-6.91 * t / rt60)
    impulse /= np.max(np.abs(impulse))
    
    reverbed = signal.fftconvolve(audio, impulse, mode='full')[:len(audio)]
    return 0.7 * audio + 0.3 * (reverbed / (np.max(np.abs(reverbed)) + 1e-6))


def generate_augmented_sample(word_name, target_length=16000, sample_rate=16000):
    """
    Generates a 1.0s (16,000 samples) augmented audio buffer.
    Applies pitch shifts, speed variations, RIR reverb, and background noise.
    """
    f0 = random.uniform(95.0, 240.0)           # Diverse speaker pitches (male to female)
    speed = random.uniform(0.85, 1.20)         # Speaking tempo perturbation
    
    if word_name == 'silence':
        # Pure ambient noise
        noise_type = random.choice(['white', 'pink', 'brown'])
        if noise_type == 'white':
            audio = np.random.normal(0, 0.05, target_length)
        else:
            b, a = signal.butter(1, 0.1, btype='low')
            audio = signal.lfilter(b, a, np.random.normal(0, 0.15, target_length))
        return (audio * 32767).astype(np.int16)

    phonemes = WORDS.get(word_name, ['i', 's', 'r', 'o'])
    speech = synthesize_word(phonemes, sample_rate=sample_rate, f0=f0, speed_factor=speed)

    # Apply room impulse reverberation in 50% of samples
    if random.random() > 0.5:
        speech = apply_reverb(speech, sample_rate)

    # Fit into 1.0s buffer with randomized start offset
    buffer = np.zeros(target_length, dtype=np.float32)
    max_start = max(0, target_length - len(speech))
    start = random.randint(0, max_start)
    end = min(target_length, start + len(speech))
    buffer[start:end] = speech[:end - start]

    # Add background noise at varying SNR (5 dB to 25 dB)
    snr_db = random.uniform(5.0, 25.0)
    noise = np.random.normal(0, 0.1, target_length)
    speech_power = np.mean(buffer ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    k = np.sqrt(speech_power / (noise_power * (10 ** (snr_db / 10))))
    mixed = buffer + k * noise

    # Soft clamp and scale to 16-bit PCM integer range
    scaled = np.clip(mixed * 28000, -32768, 32767).astype(np.int16)
    return scaled


def build_dataset(num_positive=800, num_negative=800, num_silence=200):
    """
    Builds full balanced dataset of Mel-spectrogram features and binary labels.
    Label 1: Target wake-word ("isro")
    Label 0: Confusers + Ambient Noise
    """
    print(f"Generating synthetic training dataset (Positive: {num_positive}, Confusers: {num_negative}, Silence: {num_silence})...")
    features = []
    labels = []

    confuser_words = ['intro', 'astro', 'esro', 'zero', 'stop', 'listen', 'system']

    # 1. Target keyword ("isro") -> Class 1
    for _ in range(num_positive):
        raw_pcm = generate_augmented_sample('isro')
        mel = extract_mel_spectrogram(raw_pcm)
        features.append(mel)
        labels.append(1)

    # 2. Confuser words -> Class 0
    for _ in range(num_negative):
        cword = random.choice(confuser_words)
        raw_pcm = generate_augmented_sample(cword)
        mel = extract_mel_spectrogram(raw_pcm)
        features.append(mel)
        labels.append(0)

    # 3. Ambient silence / noise -> Class 0
    for _ in range(num_silence):
        raw_pcm = generate_augmented_sample('silence')
        mel = extract_mel_spectrogram(raw_pcm)
        features.append(mel)
        labels.append(0)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Shuffle
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return X[indices], y[indices]


if __name__ == '__main__':
    X, y = build_dataset(num_positive=50, num_negative=50, num_silence=20)
    print(f"Dataset generated! Features shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Positive samples: {np.sum(y == 1)}, Negative samples: {np.sum(y == 0)}")
