import numpy as np

from scipy.signal import find_peaks


# =====================================
# Detect Peaks
# =====================================

def detect_peaks(signal):

    peaks, _ = find_peaks(
        signal,
        distance=10,
        prominence=0.3
    )

    return peaks


# =====================================
# RR Intervals
# =====================================

def calculate_rr_intervals(
        peaks,
        fs
):

    if len(peaks) < 2:
        return []

    rr = np.diff(
        peaks
    ) / fs

    return rr


# =====================================
# HRV Features
# =====================================

def calculate_hrv(rr):

    if len(rr) == 0:

        return {
            "mean_rr": 0,
            "sdnn": 0,
            "rmssd": 0
        }

    mean_rr = np.mean(rr)

    sdnn = np.std(rr)

    rmssd = np.sqrt(
        np.mean(
            np.square(
                np.diff(rr)
            )
        )
    )

    return {

        "mean_rr": mean_rr,

        "sdnn": sdnn,

        "rmssd": rmssd
    }
if __name__ == "__main__":

    signal = np.sin(
        np.linspace(
            0,
            20,
            300
        )
    )

    peaks = detect_peaks(
        signal
    )

    rr = calculate_rr_intervals(
        peaks,
        20
    )

    features = calculate_hrv(
        rr
    )

    print(
        "Peaks:",
        len(peaks)
    )

    print(features)