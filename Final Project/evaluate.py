import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = load_model("rehab_cnn_lstm_model_v3.h5")

X = np.load("X.npy")
y = np.load("y.npy")

from sklearn.model_selection import train_test_split

_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

y_pred = (model.predict(X_val) >= 0.5).astype(int).flatten()

print(classification_report(y_val, y_pred, target_names=["incorrect","correct"]))

cm = confusion_matrix(y_val, y_pred)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["correct", "incorrect"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

report_dict = classification_report(
    y_val, y_pred, target_names=["incorrect","correct"], output_dict=True
)
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv("classification_report.csv")
print("Saved → classification_report.csv")
# print("\nConfusion Matrix:\n", cm)

# Precision vs Recall Curve
y_score = model.predict(X_val).ravel()

# probability of class “incorrect” (or choose [:,0] for “correct”)

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_val, y_score)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
plt.plot(recall, precision, linewidth=2)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision–Recall Curve", fontsize=14)
plt.grid(True)
plt.show()

from sklearn.metrics import average_precision_score

pr_auc = average_precision_score(y_val, y_score)
print("PR-AUC:", pr_auc)
# Prob Histogram
y_prob = model.predict(X_val).flatten()   # shape (N,)
correct_probs = y_prob[y_val == 0]      # class 0 → correct
incorrect_probs = y_prob[y_val == 1]    # class 1 → incorrect

plt.figure(figsize=(8,5))

bins = np.linspace(0, 1, 30)

plt.hist(correct_probs, bins, alpha=0.6, label="Correct (class 0)", color="steelblue")
plt.hist(incorrect_probs, bins, alpha=0.6, label="Incorrect (class 1)", color="darkorange")

plt.xlabel("Predicted Probability (Sigmoid Output)")
plt.ylabel("Number of Samples")
plt.title("Probability Distribution of Predicted Classes")
plt.legend()
plt.grid(alpha=0.3)

plt.show()


