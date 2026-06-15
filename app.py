import os
import sys
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.config import (
    MODEL_CNN_PATH, 
    MODEL_BILSTM_PATH, 
    FS, 
    BUFFER_SIZE,
    DATASET_DIR
)
from models.fuzzy_logic import fuse_predictions
from preprocessing.signal_processor import process_signal
from training.dataset_builder import save_sample
from preprocessing.peak_detector import detect_peaks, calculate_rr_intervals, calculate_hrv

# Initialize Flask App
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global Variables for models and session history
cnn_model = None
bilstm_model = None
last_bpm_cache = None

# Bandpass Filter (0.8Hz - 2.0Hz -> 48BPM - 120BPM)
def butter_bandpass(signal, fs):
    lowcut = 0.8
    highcut = 2.0
    nyquist = fs * 0.5
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)

# Classical DSP BPM Estimation via FFT
def estimate_dsp_bpm(signal, fs):
    signal = signal - np.mean(signal)
    spectrum = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), d=1/fs)
    
    # Filter positive frequencies
    positive = freqs > 0
    freqs = freqs[positive]
    spectrum = spectrum[positive]
    
    # Restrict to physiological range
    valid = (freqs >= 0.8) & (freqs <= 2.0)
    freqs = freqs[valid]
    spectrum = spectrum[valid]
    
    if len(freqs) == 0:
        return 0.0, 0.0
        
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]
    
    # Compute SNR (peak amplitude relative to background noise)
    mean_val = np.mean(spectrum) if np.mean(spectrum) > 0 else 1.0
    snr = spectrum[peak_idx] / mean_val
    
    bpm = peak_freq * 60.0
    return bpm, snr

def load_models_on_startup():
    global cnn_model, bilstm_model
    try:
        from tensorflow.keras.models import load_model
        import sys
        print(f"Loading CNN model from: {MODEL_CNN_PATH}", file=sys.stderr)
        sys.stderr.flush()
        cnn_model = load_model(MODEL_CNN_PATH)
        print(f"Loading BiLSTM model from: {MODEL_BILSTM_PATH}", file=sys.stderr)
        sys.stderr.flush()
        bilstm_model = load_model(MODEL_BILSTM_PATH)
        print("Models loaded successfully.", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        import sys
        import traceback
        print(f"CRITICAL ERROR loading models: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()

# Serve Frontend Dashboard
@app.route('/')
def index():
    return render_template('index.html')

# API for real-time predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    global last_bpm_cache
    
    data = request.get_json()
    if not data or 'signal' not in data:
        return jsonify({"error": "Missing 'signal' parameter in request body"}), 400
        
    raw_signal = np.array(data['signal'], dtype=np.float32)
    
    if len(raw_signal) != BUFFER_SIZE:
        return jsonify({"error": f"Signal must contain exactly {BUFFER_SIZE} samples"}), 400

    try:
        # Use raw signal for deep learning models as they were trained on raw values
        model_input = raw_signal.copy().reshape(1, BUFFER_SIZE, 1)

        # 2. Run Deep Learning Model predictions
        cnn_pred = float(cnn_model.predict(model_input, verbose=0)[0][0])
        bilstm_pred = float(bilstm_model.predict(model_input, verbose=0)[0][0])

        # 3. Run DSP / FFT prediction
        # Filter raw signal first
        filtered_dsp = butter_bandpass(raw_signal - np.mean(raw_signal), FS)
        dsp_pred, dsp_snr = estimate_dsp_bpm(filtered_dsp, FS)

        # 4. Fuzzy logic combiner
        fused_bpm, conf_score, conf_label, status_msg = fuse_predictions(
            cnn_bpm=cnn_pred,
            bilstm_bpm=bilstm_pred,
            dsp_bpm=dsp_pred,
            dsp_snr=dsp_snr,
            raw_signal=raw_signal,
            last_bpm=last_bpm_cache
        )
        
        # Cache the valid consensus BPM for temporal smoothing
        if conf_label != "Low":
            last_bpm_cache = fused_bpm

        # 5. Peak detection & Heart Rate Variability (HRV) metrics
        # Find peaks in filtered signal
        peaks = detect_peaks(filtered_dsp)
        rr_intervals = calculate_rr_intervals(peaks, FS)
        hrv_metrics = calculate_hrv(rr_intervals)

        return jsonify({
            "cnn_bpm": round(cnn_pred, 1),
            "bilstm_bpm": round(bilstm_pred, 1),
            "dsp_bpm": round(dsp_pred, 1),
            "dsp_snr": round(dsp_snr, 2),
            "fused_bpm": fused_bpm,
            "confidence_score": conf_score,
            "confidence_label": conf_label,
            "status_msg": status_msg,
            "hrv": {
                "mean_rr": float(hrv_metrics["mean_rr"]),
                "sdnn": float(hrv_metrics["sdnn"]),
                "rmssd": float(hrv_metrics["rmssd"])
            }
        })

    except Exception as e:
        print(f"Error processing prediction request: {e}")
        return jsonify({"error": str(e)}), 500

# API to save new samples
@app.route('/api/save', methods=['POST'])
def save():
    data = request.get_json()
    if not data or 'signal' not in data or 'bpm' not in data:
        return jsonify({"error": "Missing parameters 'signal' or 'bpm'"}), 400
        
    signal = np.array(data['signal'], dtype=np.float32)
    bpm = data['bpm']
    
    if len(signal) != BUFFER_SIZE:
        return jsonify({"error": f"Signal must contain exactly {BUFFER_SIZE} samples"}), 400

    try:
        # Save sample to directory and log in CSV
        save_sample(signal, bpm)
        
        # Find directory to list filename
        existing_files = [f for f in os.listdir("dataset/eye_sequences") if f.endswith(".npy")]
        filename = f"sample_{len(existing_files)}.npy"

        return jsonify({
            "success": True,
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to get simulated patient samples
@app.route('/api/simulate_sample', methods=['GET'])
def get_simulate_sample():
    sample_id = request.args.get('id', default=1, type=int)
    filename = f"sample_{sample_id}.npy"
    if sample_id == 4: # sample_4 is skipped in the dataset
        filename = "sample_1.npy"
        
    filepath = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(filepath):
        # Fallback to check if any sample files exist
        files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".npy")]
        if files:
            filepath = os.path.join(DATASET_DIR, files[0])
        else:
            # Generate fallback simulated wave
            dummy = np.sin(np.linspace(0, 30, BUFFER_SIZE)) * 4.0 + 150.0
            return jsonify({"signal": dummy.tolist(), "filename": "simulated_dummy.npy"})
            
    try:
        signal = np.load(filepath)
        return jsonify({
            "signal": signal.tolist(),
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load models on startup (required for production WSGI servers like Gunicorn)
load_models_on_startup()

if __name__ == '__main__':
    # Run server
    app.run(host='127.0.0.1', port=5000, debug=False)
