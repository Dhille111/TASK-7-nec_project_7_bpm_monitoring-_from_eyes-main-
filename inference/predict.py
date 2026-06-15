import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import DATASET_DIR, LABELS_CSV, MODEL_CNN_PATH, MODEL_BILSTM_PATH, FS
from models.fuzzy_logic import fuse_predictions
from preprocessing.signal_processor import process_signal

def butter_bandpass(signal, fs):
    lowcut = 0.8
    highcut = 2.0
    nyquist = fs * 0.5
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)

def estimate_dsp_bpm(signal, fs):
    signal = signal - np.mean(signal)
    spectrum = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), d=1/fs)
    
    positive = freqs > 0
    freqs = freqs[positive]
    spectrum = spectrum[positive]
    
    valid = (freqs >= 0.8) & (freqs <= 2.0)
    freqs = freqs[valid]
    spectrum = spectrum[valid]
    
    if len(freqs) == 0:
        return 0.0, 0.0
        
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]
    
    # Simple estimate of SNR as peak height relative to average spectrum height
    mean_val = np.mean(spectrum) if np.mean(spectrum) > 0 else 1.0
    snr = spectrum[peak_idx] / mean_val
    
    bpm = peak_freq * 60.0
    return bpm, snr

def run_offline_evaluation():
    print("=" * 60)
    print("HEART RATE RETINA PROJECT - OFFLINE EVALUATION")
    print("=" * 60)

    # 1. Check if labels file exists
    if not os.path.isfile(LABELS_CSV):
        print(f"Error: Dataset labels file not found at: {LABELS_CSV}")
        print("Please record some samples or run training/test_dataset.py first.")
        return

    # 2. Load Models
    try:
        from tensorflow.keras.models import load_model
        print("Loading Keras Models...")
        cnn_model = load_model(MODEL_CNN_PATH)
        bilstm_model = load_model(MODEL_BILSTM_PATH)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure cnn_model.keras and bilstm_model.keras exist in saved_models/")
        return

    # 3. Read dataset
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"Found {len(labels_df)} dataset samples.")
    print("-" * 115)
    print(f"{'File':<15} | {'True BPM':<8} | {'DSP BPM':<8} | {'CNN BPM':<8} | {'BiLSTM':<8} | {'Fused':<8} | {'Conf':<6} | {'Err (Fused)':<10}")
    print("-" * 115)

    cnn_errors = []
    bilstm_errors = []
    fused_errors = []

    for _, row in labels_df.iterrows():
        filename = row["file"]
        true_bpm = float(row["bpm"])
        filepath = os.path.join(DATASET_DIR, filename)

        if not os.path.exists(filepath):
            print(f"{filename:<15} | File not found at {filepath}")
            continue

        # Load raw signal
        raw_signal = np.load(filepath)
        
        # Use raw signal for deep learning models as they were trained on raw values
        model_input = raw_signal.copy().reshape(1, 300, 1)

        # Run Deep Learning predictions
        cnn_pred = float(cnn_model.predict(model_input, verbose=0)[0][0])
        bilstm_pred = float(bilstm_model.predict(model_input, verbose=0)[0][0])

        # Run DSP / FFT prediction
        filtered_dsp = butter_bandpass(raw_signal - np.mean(raw_signal), FS)
        dsp_pred, dsp_snr = estimate_dsp_bpm(filtered_dsp, FS)

        # Run Fuzzy logic fuser
        fused_pred, conf_score, conf_label, status = fuse_predictions(
            cnn_pred, bilstm_pred, dsp_pred, dsp_snr, raw_signal
        )

        # Compute errors
        cnn_err = abs(cnn_pred - true_bpm)
        bilstm_err = abs(bilstm_pred - true_bpm)
        fused_err = abs(fused_pred - true_bpm)

        cnn_errors.append(cnn_err)
        bilstm_errors.append(bilstm_err)
        fused_errors.append(fused_err)

        print(f"{filename:<15} | {true_bpm:<8.1f} | {dsp_pred:<8.1f} | {cnn_pred:<8.1f} | {bilstm_pred:<8.1f} | {fused_pred:<8.1f} | {conf_label:<6} | {fused_err:<10.1f}")

    print("-" * 115)
    if len(fused_errors) > 0:
        print(f"CNN Mean Absolute Error (MAE):     {np.mean(cnn_errors):.2f} BPM")
        print(f"BiLSTM Mean Absolute Error (MAE):  {np.mean(bilstm_errors):.2f} BPM")
        print(f"Fuzzy Fused Mean Absolute Error:   {np.mean(fused_errors):.2f} BPM")
    print("=" * 60)

if __name__ == "__main__":
    run_offline_evaluation()
