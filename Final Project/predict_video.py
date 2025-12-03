import numpy as np
from tensorflow.keras.models import load_model

CLIP_LEN = 24

def predict_video(npy_file):
    model = load_model("rehab_cnn_lstm_model.h5")

    H = np.load(npy_file)   # shape: (F, 64, 64)
    F = H.shape[0]

    idx = np.linspace(0, F-1, CLIP_LEN).astype(int)
    clip = H[idx][..., None]
    clip = clip[np.newaxis, ...]

    p = model.predict(clip)[0][0]
    label = "Correct" if p >= 0.5 else "Incorrect"
    print(f"\nPrediction: {label}  (confidence = {p:.3f})")
    return p

# Example:
predict_video("test_heatmaps_npy/incorrect/209_18_2_4_2_wheelchair.npy")
