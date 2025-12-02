import numpy as np
import os
import cv2
import glob

SRC = "dataset_skeleton"          # your input with txt
DST = "dataset_heatmaps"          # output folder
M = 64                            # heatmap size
NUM_JOINTS = 25

def coordinates_to_heatmap(coords):
    # coords is (25,2) normalized to [0,1]
    heatmap = np.zeros((M, M), dtype=np.float32)
    for (x, y) in coords:
        px = int(x * (M - 1))
        py = int(y * (M - 1))
        cv2.circle(heatmap, (px, py), radius=2, color=1.0, thickness=-1)
    return (heatmap * 255).astype(np.uint8)

for label_folder in ["correct", "incorrect"]:
    input_path = os.path.join(SRC, label_folder)
    output_path = os.path.join(DST, label_folder)
    os.makedirs(output_path, exist_ok=True)

    for txt_file in os.listdir(input_path):
        if not txt_file.endswith(".txt"):
            continue

        file_path = os.path.join(input_path, txt_file)
        video_name = os.path.splitext(txt_file)[0]
        video_out = os.path.join(output_path, video_name)
        os.makedirs(video_out, exist_ok=True)

        data = np.loadtxt(file_path, delimiter=',')  # (T, 75)
        data = data.reshape(-1, NUM_JOINTS, 3)      # (T, 25, 3)

        # Take x,y only
        xy = data[:, :, :2]  # (T, 25, 2)

        # Normalize per video
        min_xy = xy.reshape(-1, 2).min(axis=0)
        max_xy = xy.reshape(-1, 2).max(axis=0)
        xy_norm = (xy - min_xy) / (max_xy - min_xy + 1e-6)

        # Generate heatmaps frame-by-frame
        for i in range(xy_norm.shape[0]):
            hm = coordinates_to_heatmap(xy_norm[i])      # (M,M)
            cv2.imwrite(os.path.join(video_out, f"{video_name}_f{i:04d}.png"), hm)

print("Heatmap generation complete.")
