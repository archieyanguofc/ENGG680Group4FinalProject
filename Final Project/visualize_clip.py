import numpy as np
import matplotlib.pyplot as plt

clip = np.load("demo_clip.npy")   # loads (T, M, M, 1)
T = clip.shape[0]

# compute grid
cols = 6
rows = (T + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 2*rows))
axes = axes.flatten()

for i in range(T):
    axes[i].imshow(clip[i].squeeze(), cmap='hot')
    # axes[i].set_title(f"Frame {i}")
    axes[i].axis('off')

# Turn off unused axes
for j in range(T, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()