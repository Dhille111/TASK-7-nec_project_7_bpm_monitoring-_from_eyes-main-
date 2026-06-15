import numpy as np

def compute_dsp_confidence(dsp_snr):
    """
    Computes a confidence score [0, 1] for the DSP/FFT peak frequency.
    Higher peak prominence/SNR yields higher confidence.
    """
    if dsp_snr is None:
        return 0.0
    # Map SNR to [0, 1] confidence
    # Below 1.2: low confidence, Above 3.0: high confidence
    if dsp_snr <= 1.2:
        return 0.0
    elif dsp_snr >= 3.0:
        return 1.0
    else:
        return (dsp_snr - 1.2) / (3.0 - 1.2)

def compute_signal_quality(signal_std, raw_signal=None):
    """
    Evaluates signal quality based on standard deviation of the raw signal.
    Very high std indicates excessive motion artifacts.
    Very low std indicates a flatline or flat camera feed.
    """
    if signal_std is None:
        return 0.0
    
    # Standard deviation thresholds for normalized or raw green signal
    # If the signal is normalized, std is 1.0, so we inspect raw signal std if provided
    val = signal_std
    if raw_signal is not None:
        val = np.std(raw_signal)

    # If raw green channel std is too high (> 25) or too low (< 0.2)
    if val < 0.2 or val > 25.0:
        return 0.1
    elif val > 15.0:
        # Graceful degradation for moderate motion noise
        return 1.0 - (val - 15.0) / (25.0 - 15.0)
    elif val < 1.0:
        # Graceful degradation for low light/flat signal
        return (val - 0.2) / (1.0 - 0.2)
    else:
        return 1.0

def compute_model_agreement(cnn_bpm, bilstm_bpm):
    """
    Computes confidence [0, 1] based on agreement between the two neural networks.
    If they predict very close values, confidence is high.
    """
    diff = abs(cnn_bpm - bilstm_bpm)
    if diff <= 5.0:
        return 1.0
    elif diff >= 25.0:
        return 0.0
    else:
        return 1.0 - (diff - 5.0) / (25.0 - 5.0)

def fuse_predictions(cnn_bpm, bilstm_bpm, dsp_bpm, dsp_snr, raw_signal, last_bpm=None):
    """
    Fuses the predictions from CNN, BiLSTM, and DSP using Fuzzy Logic principles.
    Returns:
        fused_bpm (float): The final consensus heart rate.
        confidence_score (float): Numeric confidence value [0, 1].
        confidence_label (str): Label string ('High', 'Medium', 'Low').
        status_msg (str): Informative message about the signal state.
    """
    # 1. Reject invalid input bounds
    cnn_bpm = max(40, min(180, cnn_bpm))
    bilstm_bpm = max(40, min(180, bilstm_bpm))
    
    if dsp_bpm is None or dsp_bpm < 40 or dsp_bpm > 180:
        dsp_bpm = (cnn_bpm + bilstm_bpm) / 2.0
        dsp_confidence = 0.0
    else:
        dsp_confidence = compute_dsp_confidence(dsp_snr)

    # 2. Compute membership degrees
    raw_std = np.std(raw_signal) if raw_signal is not None else 1.0
    signal_quality = compute_signal_quality(raw_std)
    model_agreement = compute_model_agreement(cnn_bpm, bilstm_bpm)

    # 3. Calculate dynamic weights (Fuzzy inference)
    # DSP is preferred when signal quality is high and peak is clear
    w_dsp = 0.5 * dsp_confidence * signal_quality
    
    # Models are trusted more when they agree and signal quality is decent
    w_cnn = 0.25 * model_agreement * signal_quality
    w_bilstm = 0.25 * model_agreement * signal_quality

    total_weight = w_dsp + w_cnn + w_bilstm

    if total_weight > 0.05:
        fused_bpm = (w_dsp * dsp_bpm + w_cnn * cnn_bpm + w_bilstm * bilstm_bpm) / total_weight
    else:
        # Fallback if everything is low quality
        fused_bpm = (cnn_bpm + bilstm_bpm + dsp_bpm) / 3.0

    # 4. Temporal Plausibility & Damping
    if last_bpm is not None:
        delta = abs(fused_bpm - last_bpm)
        if delta > 20.0:
            # Dampen sudden spikes
            fused_bpm = last_bpm + np.sign(fused_bpm - last_bpm) * (20.0 + 0.1 * (delta - 20.0))
            # Reduce confidence due to sudden change
            confidence_factor = 0.7
        else:
            confidence_factor = 1.0
    else:
        confidence_factor = 1.0

    # Compute final confidence score
    confidence_score = (0.4 * signal_quality + 0.3 * dsp_confidence + 0.3 * model_agreement) * confidence_factor
    confidence_score = float(max(0.0, min(1.0, confidence_score)))

    # Determine label
    if confidence_score >= 0.75:
        confidence_label = "High"
    elif confidence_score >= 0.40:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    # Status description
    if signal_quality < 0.3:
        status_msg = "Warning: High motion artifacts or bad lighting detected."
    elif dsp_confidence < 0.3:
        status_msg = "Weak physiological rhythm, relying on neural models."
    else:
        status_msg = "Stable signal locked."

    return round(fused_bpm, 1), round(confidence_score, 2), confidence_label, status_msg
