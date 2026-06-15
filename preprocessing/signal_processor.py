import numpy as np

# ==================================
# Remove DC Component
# ==================================
def detrend_signal(signal):

    signal = np.array(signal)

    return signal - np.mean(signal)

# ==================================
# Normalize Signal
# ==================================
def normalize_signal(signal):

    signal = np.array(signal)

    std = np.std(signal)

    if std == 0:
        return signal

    return signal / std

# ==================================
# Moving Average Smoothing
# ==================================
def smooth_signal(signal, window=5):

    return np.convolve(
        signal,
        np.ones(window) / window,
        mode='same'
    )

# ==================================
# Full Processing Pipeline
# ==================================
def process_signal(signal):

    signal = detrend_signal(signal)

    signal = normalize_signal(signal)

    signal = smooth_signal(signal)

    return signal