import os

# ==========================================
# Signal Acquisition & Preprocessing Settings
# ==========================================
FS = 20.0             # Sampling frequency in Hz
BUFFER_SIZE = 300     # Number of frames/samples per input sequence
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))

# ==========================================
# Paths configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_CNN_PATH = os.path.join(BASE_DIR, "saved_models", "cnn_model.h5")
MODEL_BILSTM_PATH = os.path.join(BASE_DIR, "saved_models", "bilstm_model.h5")
SHAPE_PREDICTOR_PATH = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat", "shape_predictor_68_face_landmarks.dat")

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "eye_sequences")
LABELS_CSV = os.path.join(BASE_DIR, "dataset", "labels.csv")

# Ensure dataset directory exists
os.makedirs(DATASET_DIR, exist_ok=True)
