import os
import glob
import numpy as np
import cv2

HEATMAP_ROOT = "dataset_heatmaps"
CLIP_LEN = 24
CLIPS_PER_VIDEO = 6
M = 64

# correct = 1, incorrect = 2 → map to model labels
CLASS_NAME_TO_ID = {"correct": 1, "incorrect": 2}
ID_TO_ZERO = {1: 1, 2: 0}  # correct→1, incorrect→0

def load_heatmaps_from_folder(folder):
    pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
    if len(pngs) == 0:
        return None
    frames = []
    for p in pngs:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (M, M))
        frames.append(img / 255.0)
    return np.array(frames, dtype=np.float32)  # (F, M, M)

def sample_clip_starts(F, T, n):
    if F < T:
        return []
    max_start = F - T
    return np.linspace(0, max_start, n, dtype=int)

def make_clips(root):
    X, y = [], []
    for label in ["correct", "incorrect"]:
        label_id = CLASS_NAME_TO_ID[label]
        label_zero = ID_TO_ZERO[label_id]

        class_path = os.path.join(root, label)
        for folder in os.listdir(class_path):
            seq_folder = os.path.join(class_path, folder)
            if not os.path.isdir(seq_folder):
                continue

            H = load_heatmaps_from_folder(seq_folder)  # (F, M, M)
            if H is None:
                continue

            F = H.shape[0]
            starts = sample_clip_starts(F, CLIP_LEN, CLIPS_PER_VIDEO)

            for s in starts:
                clip = H[s:s+CLIP_LEN]
                if clip.shape[0] == CLIP_LEN:
                    clip = clip[..., None]  # → (T, M, M, 1)
                    X.append(clip)
                    y.append(label_zero)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print("Done")
    print("X:", X.shape)  # should be (num_clips, 24, 64, 64, 1)
    print("y:", y.shape, "| correct:", (y==1).sum(), " incorrect:", (y==0).sum())
    return X, y

if __name__ == "__main__":
    X, y = make_clips(HEATMAP_ROOT)
    np.save("X.npy", X)
    np.save("y.npy", y)
    print("Saved X.npy and y.npy")
