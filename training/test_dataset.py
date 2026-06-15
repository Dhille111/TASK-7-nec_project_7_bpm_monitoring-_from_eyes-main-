from dataset_builder import save_sample
import numpy as np

fake_signal = np.random.randn(300)

save_sample(
    fake_signal,
    78
)