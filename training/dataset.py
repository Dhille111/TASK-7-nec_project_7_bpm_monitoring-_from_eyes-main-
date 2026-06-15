import os
import pandas as pd
import numpy as np

DATA_DIR = "dataset/eye_sequences"
LABEL_FILE = "dataset/labels.csv"


def load_dataset():

    labels = pd.read_csv(
        LABEL_FILE
    )

    X = []
    y = []

    for _, row in labels.iterrows():

        file_name = row["file"]

        bpm = row["bpm"]

        path = os.path.join(
            DATA_DIR,
            file_name
        )

        signal = np.load(path)

        X.append(signal)

        y.append(bpm)

    X = np.array(X)

    y = np.array(y)

    return X, y


if __name__ == "__main__":

    X, y = load_dataset()

    print("Signals Shape:", X.shape)

    print("Labels Shape:", y.shape)

    print("BPM Labels:", y)