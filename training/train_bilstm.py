import os
import sys

from sklearn.model_selection import train_test_split

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from training.dataset import load_dataset
from models.bilstm_model import build_bilstm

# ==========================
# Load Dataset
# ==========================

X, y = load_dataset()

print("Dataset Loaded")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ==========================
# Reshape
# ==========================

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Build BiLSTM
# ==========================

model = build_bilstm()

# ==========================
# Train
# ==========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=20,
    batch_size=4
)

# ==========================
# Save Model
# ==========================

os.makedirs(
    "saved_models",
    exist_ok=True
)

model.save(
    "saved_models/bilstm_model.keras"
)

print(
    "\nBiLSTM Model Saved Successfully"
)

# ==========================
# Evaluate
# ==========================

loss, mae = model.evaluate(
    X_test,
    y_test
)

print(
    f"Test MAE: {mae:.2f}"
)