"""
=========================================================
TAHAP 5: EVALUASI MODEL
=========================================================
Proyek : Deteksi Kantuk Pengemudi (Drowsiness Detection)

Evaluasi model terbaik (MobileNetV2) pada test set:
1. Load model terbaik
2. Predict test set
3. Classification report (precision, recall, f1-score)
4. Confusion matrix
5. ROC Curve (per class)
6. Analisis kesalahan (salah prediksi)

Output di folder ini:
- classification_report.txt
- confusion_matrix.png
- roc_curve.png
- test_results.npy (y_true, y_pred, y_prob)
=========================================================
"""

import os
os.add_dll_directory(r"D:\conda_envs\drowsiness\Library\bin")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, f1_score, precision_score, recall_score)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings
warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)

# =========================================================
# KONFIGURASI
# =========================================================
PROCESSED_DIR = r"D:\TA Mesin\tahap_2_Preprocessing\processed"
MODELS_DIR    = r"D:\TA Mesin\models"
OUTPUT_DIR    = r"D:\TA Mesin\tahap_5_Evaluasi"

IMG_SIZE = 64
NUM_CLASSES = 4
CLASSES = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================
print("=" * 60)
print("1. LOAD DATA TEST")
print("=" * 60)

X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")
print(f"Distribusi:")
for i in range(NUM_CLASSES):
    count = (y_test == i).sum()
    print(f"  {CLASSES[i]}: {count} gambar")

# One-hot encode
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Preprocess untuk MobileNetV2
X_test_proc = preprocess_input(X_test * 255.0)

# =========================================================
# 2. LOAD MODEL
# =========================================================
print()
print("=" * 60)
print("2. LOAD MODEL TERBAIK")
print("=" * 60)

model_path = os.path.join(MODELS_DIR, "mobilenetv2_best.h5")
if not os.path.exists(model_path):
    print(f"[ERROR] Model tidak ditemukan: {model_path}")
    exit(1)

model = keras.models.load_model(model_path)
model.summary()

# =========================================================
# 3. PREDIKSI
# =========================================================
print()
print("=" * 60)
print("3. PREDIKSI TEST SET")
print("=" * 60)

y_prob = model.predict(X_test_proc, batch_size=32, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# Simpan hasil
np.save(os.path.join(OUTPUT_DIR, "test_results.npy"),
        {"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob})
print("Hasil prediksi disimpan.")

# =========================================================
# 4. METRIK EVALUASI
# =========================================================
print()
print("=" * 60)
print("4. METRIK EVALUASI")
print("=" * 60)

acc = (y_pred == y_test).mean()
print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")

# Classification report
report = classification_report(y_test, y_pred, target_names=CLASSES, digits=4)
print("\nClassification Report:")
print(report)

# Simpan ke file
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)\n\n")
    f.write(report)

# Per-class metrics
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print("\nPer-Class Metrics:")
for i in range(NUM_CLASSES):
    print(f"  {CLASSES[i]:15s} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | F1: {f1[i]:.4f}")

# =========================================================
# 5. CONFUSION MATRIX
# =========================================================
print()
print("=" * 60)
print("5. CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (baris=true, kolom=pred):")
print("        ", end="")
for c in CLASSES:
    print(f"{c:>14s}", end="")
print()
for i in range(NUM_CLASSES):
    print(f"{CLASSES[i]:>12s}", end="")
    for j in range(NUM_CLASSES):
        print(f"{cm[i, j]:>14d}", end="")
    print()

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(NUM_CLASSES), yticks=np.arange(NUM_CLASSES),
       xticklabels=CLASSES, yticklabels=CLASSES,
       title="Confusion Matrix - MobileNetV2",
       xlabel="Predicted Label", ylabel="True Label")

# Tambah angka di setiap sel
thresh = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Confusion matrix disimpan.")

# =========================================================
# 6. ROC CURVE (One-vs-Rest)
# =========================================================
print()
print("=" * 60)
print("6. ROC CURVE")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"{CLASSES[i]} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - MobileNetV2 (One-vs-Rest)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"ROC curve disimpan.")

# =========================================================
# 7. ANALISIS KESALAHAN
# =========================================================
print()
print("=" * 60)
print("7. ANALISIS KESALAHAN")
print("=" * 60)

# Hitung kesalahan per kelas
error_mask = y_pred != y_test
total_errors = error_mask.sum()
print(f"\nTotal kesalahan: {total_errors}/{len(y_test)} ({total_errors/len(y_test)*100:.2f}%)")

for i in range(NUM_CLASSES):
    class_mask = y_test == i
    class_errors = error_mask & class_mask
    n_class = class_mask.sum()
    n_err = class_errors.sum()
    if n_class > 0:
        print(f"  {CLASSES[i]:15s}: {n_err}/{n_class} salah ({n_err/n_class*100:.2f}%)")

# Confusion breakdown (kesalahan terbanyak)
print("\n5 kesalahan terbanyak:")
error_pairs = {}
for t, p in zip(y_test[error_mask], y_pred[error_mask]):
    pair = (CLASSES[t], CLASSES[p])
    error_pairs[pair] = error_pairs.get(pair, 0) + 1

sorted_errors = sorted(error_pairs.items(), key=lambda x: -x[1])
for (true_label, pred_label), count in sorted_errors[:5]:
    print(f"  True={true_label:15s} -> Pred={pred_label:15s}: {count} kali")

# =========================================================
# RINGKASAN
# =========================================================
print()
print("=" * 60)
print("RINGKASAN EVALUASI")
print("=" * 60)
print(f"\nModel       : MobileNetV2 (Transfer Learning)")
print(f"Input size  : {IMG_SIZE}x{IMG_SIZE}x3 (letterbox padding)")
print(f"Test set    : {len(X_test)} gambar")
print(f"Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
print(f"F1 Score    : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"\nFile output:")
print(f"  - confusion_matrix.png")
print(f"  - roc_curve.png")
print(f"  - classification_report.txt")
print(f"  - test_results.npy")
print()
print("=" * 60)
print("TAHAP 5 (EVALUASI) SELESAI")
print("=" * 60)
