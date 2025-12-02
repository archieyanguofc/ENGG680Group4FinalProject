import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# Load prepared data
X = np.load("X.npy").astype("float32")      # shape: (N, T, 64, 64, 1)
y = np.load("y.npy").astype("float32")      # shape: (N,)

print("Dataset:", X.shape, y.shape)

T = X.shape[1]       # number of frames (likely 24)
M = X.shape[2]       # heatmap size (64)
BATCH_SIZE = 4
EPOCHS = 20

# Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def build_model():
    frames = layers.Input(shape=(T, M, M, 1))

    # CNN on each frame
    x = layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu', padding='same'))(frames)
    x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)

    x = layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)

    x = layers.TimeDistributed(layers.Conv2D(128, (3,3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)  # → (T, 128)

    # LSTM over time dimension
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(frames, out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_model()
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

history.history.keys()
np.save("history.npy", history.history)
# Accuracy curve
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("acc_curve.png", dpi=300)

#Loss curve
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("loss_curve.png", dpi=300)

model.save("rehab_cnn_lstm_model_v3.h5")
print("\n✅ Model training complete. Model saved as rehab_cnn_lstm_model_v3.h5")
