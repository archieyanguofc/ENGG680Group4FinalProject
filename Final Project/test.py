import numpy as np
import os
import cv2
import glob
from make_clips import make_clips

SRC = "test_data"          # your input with txt
DST = "test_heatmaps"          # output folder
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

print("Testing Heatmap generation complete.")

INPUT_ROOT = "test_heatmaps"   # your heatmap PNG folders
OUTPUT_ROOT = "test_heatmaps_npy"  # new output folder

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "correct"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "incorrect"), exist_ok=True)

def convert_folder_to_npy(folder_path, save_path):
    png_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    if len(png_files) == 0:
        print("No images found in", folder_path)
        return

    frames = []
    for img_path in png_files:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0  # normalize 0-1
        frames.append(img)

    arr = np.stack(frames)  # shape (F, H, W)
    np.save(save_path, arr)
    print("Saved:", save_path, arr.shape)

for label in ["correct", "incorrect"]:
    label_in = os.path.join(INPUT_ROOT, label)
    label_out = os.path.join(OUTPUT_ROOT, label)

    for folder in os.listdir(label_in):
        folder_path = os.path.join(label_in, folder)
        if not os.path.isdir(folder_path):
            continue

        save_path = os.path.join(label_out, folder + ".npy")
        convert_folder_to_npy(folder_path, save_path)

print("\n Conversion complete. New npy heatmaps saved to test_heatmaps_npy/")
