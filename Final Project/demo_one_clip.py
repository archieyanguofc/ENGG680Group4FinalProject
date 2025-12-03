import numpy as np

X = np.load("X.npy")   # (N, T, M, M, 1)
y = np.load("y.npy")   # (N,)

print(X.shape, y.shape)

# pick one clip
i = 0   # or any index
clip = X[i]          # shape (T, M, M, 1)
label = y[i]         # 0 or 1
print("clip label:", ["correct","incorrect"][label])

np.save("demo_clip.npy", clip)

unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))