# Heart Rate Retina Project - Real-time PPG Dashboard

A deep-learning and digital signal processing (DSP) web application that estimates a user's heart rate (BPM) in real-time from video sequences of their eye region using photoplethysmography (PPG).

The application crops the eye regions, extracts blood-volume pulse variations from the green color channel, filters the signal, and feeds it into trained Keras neural networks and a spectral estimator. A custom Fuzzy Logic combiner then fuses the estimates to output a robust, high-confidence heart rate reading.

---

## 🛠️ Project Structure

```text
├── app.py                      # Flask web server and backend prediction API
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── models/
│   ├── bilstm_model.py         # Bidirectional LSTM model definition
│   ├── cnn_model.py            # Convolutional Neural Network model definition
│   └── fuzzy_logic.py          # Fuzzy logic fuser and confidence evaluation
├── saved_models/
│   ├── bilstm_model.keras      # Trained BiLSTM weights
│   └── cnn_model.keras         # Trained CNN weights
├── preprocessing/
│   ├── eye_detector.py         # Face-mesh detector standalone script
│   ├── eye_landmarks.py        # Eye landmarks definition helper
│   ├── peak_detector.py        # SciPy-based peak and HRV features detector
│   ├── roi_extractor.py        # Eye region bounding box crops
│   ├── signal_extractor.py     # Live OpenCV signal extractor utility
│   ├── signal_filter.py        # DSP filter and FFT estimation script
│   └── signal_processor.py     # Preprocessing pipeline (detrend, norm, smooth)
├── training/
│   ├── collect_dataset.py      # Standalone opencv dataset collector
│   ├── auto_dataset_collector.py# Autopilot dataset collector via DSP peaks
│   ├── dataset_builder.py      # Saves npy arrays and appends labels.csv
│   ├── dataset.py              # Data loader for CNN/LSTM training
│   ├── train.py                # CNN training script
│   └── train_bilstm.py         # BiLSTM training script
├── inference/
│   └── predict.py              # Offline evaluator of dataset predictions
├── templates/
│   └── index.html              # Frontend user interface dashboard
├── static/
│   ├── css/
│   │   └── styles.css          # Glassmorphic dashboard styles
│   └── js/
│       └── app.js              # MediaPipe landmark capture & visualization logic
└── venv/                       # Local Python virtual environment
```

---

## 🚀 Running the Web Application Dashboard

### 1. Prerequisites
Ensure you are using Python 3.10 and have installed the dependencies in the virtual environment.

```bash
# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install requirements
python -m pip install -r requirements.txt
```

### 2. Start the Server
Run the Flask server:
```bash
python app.py
```
This loads the CNN and BiLSTM models and launches the local web server at `http://127.0.0.1:5000`.

### 3. Open the Dashboard
1. Open your web browser (Chrome, Edge, or Firefox) and navigate to `http://127.0.0.1:5000`.
2. Grant **Webcam permissions** when prompted.
3. Click the **Start Acquisition** button in the sidebar to begin tracking.
4. Align your face with the camera. The browser will isolate your eye region, map facial landmarks, crop the ROIs in real-time, and plot the PPG pulse waves.
5. Once 300 samples (~15 seconds) are collected, the backend will display the live CNN, BiLSTM, FFT estimates, and the fused Fuzzy Consensus BPM!

---

## 🧬 Algorithm & Technology Stack

1. **Webcam Tracking (MediaPipe)**: Tracks facial contours using browser-side JavaScript, avoiding local CPU lags caused by python frame transmission.
2. **Signal Preprocessing**: The raw green channel signal from the cropped eye region is:
   - **Detrended**: DC offsets are removed.
   - **Normalized**: Divided by its standard deviation.
   - **Smoothed**: Convolved with a moving average filter to remove high-frequency noise.
3. **Deep Learning Inference (Keras)**:
   - **CNN**: Extracts local spatial patterns from the time-series signal.
   - **BiLSTM**: Learns bidirectional temporal correlations from heartbeats.
4. **Fuzzy Logic Decision Combiner**: Fuses predictions by evaluating:
   - **DSP SNR/Spectral Prominence**: Measures peak quality in the frequency spectrum.
   - **Model Agreement**: Evaluates variance between the CNN and BiLSTM predictions.
   - **Signal Stability**: Flags extreme noise or motion artifacts.
   Fuses the outputs into a single consensus BPM and calculates a confidence score (Low, Medium, High).
5. **Heart Rate Variability (HRV)**: Computes physiological indices (`SDNN`, `RMSSD`) from the distance between detected signal peaks.

---

## ☁️ Deploying to Render (render.com)

This repository is ready for automated deployment to Render as a Python Web Service.

### One-Click Blueprint Deployment
1. Push this project to your GitHub, GitLab, or Bitbucket repository.
2. In the Render Dashboard, click **New +** and select **Blueprint**.
3. Connect your repository. Render will automatically detect the `render.yaml` configuration and deploy the service.

### Manual Web Service Deployment
If you prefer to configure the Web Service manually:
1. In the Render Dashboard, click **New +** and select **Web Service**.
2. Connect your repository.
3. Configure the following settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
4. Click **Deploy Web Service**.
