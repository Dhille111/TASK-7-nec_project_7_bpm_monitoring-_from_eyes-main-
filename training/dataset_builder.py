import os
import csv
import numpy as np

SAVE_DIR = "dataset/eye_sequences"
CSV_FILE = "dataset/labels.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

def save_sample(signal, bpm):

    existing_files = [
        f for f in os.listdir(SAVE_DIR)
        if f.endswith(".npy")
    ]

    sample_id = len(existing_files) + 1

    filename = f"sample_{sample_id}.npy"

    filepath = os.path.join(
        SAVE_DIR,
        filename
    )

    np.save(
        filepath,
        signal
    )

    file_exists = os.path.isfile(
        CSV_FILE
    )

    with open(
        CSV_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                ["file", "bpm"]
            )

        writer.writerow(
            [filename, bpm]
        )

    print(
        f"Saved {filename} BPM={bpm}"
    )