"""
=========================================================
TAHAP 4B: HYPERPARAMETER TUNING (RandomSearch)
=========================================================
Proyek : Deteksi Kantuk Pengemudi (Drowsiness Detection)

Method : RandomSearch via Keras Tuner
Model  : MobileNetV2 (Transfer Learning)

Parameter yang dituning:
  - learning_rate : [1e-4, 5e-4, 1e-3]
  - dense_units   : [64, 128, 256]
  - dropout_rate  : [0.3, 0.5, 0.7]
  - optimizer     : ['adam', 'rmsprop']
  - batch_size    : [16, 32]

Output:
  - models/mobilenetv2_tuned.h5   (model hasil tuning)
  - best_params.json              (hyperparameter terbaik)
  - tuning_results.csv            (semua percobaan)
  - summary.json                  (ringkasan hasil)
  - tuning_history.png            (visualisasi tuning)
=========================================================
"""

import os
cuda_path = r"D:\conda_envs\drowsiness\Library\bin"
os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(cuda_path)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import json
import csv
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")
import sys

# Redirect stdout ke file log juga agar semua output tercatat
LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tuning_log.txt"
)
log_fh = open(LOG_FILE, "w", encoding="utf-8")

class Tee:
    def write(self, msg):
        sys.__stdout__.write(msg)
        log_fh.write(msg)
        log_fh.flush()
        sys.__stdout__.flush()
    def flush(self):
        sys.__stdout__.flush()
        log_fh.flush()

sys.stdout = Tee()

# =========================================================
# REPRODUCIBILITY
# =========================================================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =========================================================
# KONFIGURASI
# =========================================================
PROCESSED_DIR = r"D:\TA Mesin\tahap_2_Preprocessing\processed"
MODELS_DIR    = r"D:\TA Mesin\models"
OUTPUT_DIR    = r"D:\TA Mesin\tahap_4_Modeling"

IMG_SIZE = 64
NUM_CLASSES = 4
CLASSES = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]
EPOCHS = 15
TRIALS = 20

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)
print(f"Seed      : {SEED}")
print(f"Trials    : {TRIALS}")
print(f"Epochs    : {EPOCHS}")

# =========================================================
# 1. LOAD DATA
# =========================================================
print("\n" + "=" * 60)
print("1. LOAD DATA")
print("=" * 60)

X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
X_val   = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat   = keras.utils.to_categorical(y_val, NUM_CLASSES)

print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")

# Preprocess untuk MobileNetV2: [0,1] -> [0,255] -> [-1,1]
X_train_proc = preprocess_input(X_train * 255.0)
X_val_proc   = preprocess_input(X_val * 255.0)

# =========================================================
# 2. BUILD MODEL FUNCTION
# =========================================================
def build_model_direct(lr, dense_units, dropout_rate, optimizer_name):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_Tuned")

    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================================================
# 3. PARAMETER GRID
# =========================================================
print("\n" + "=" * 60)
print("2. PARAMETER GRID")
print("=" * 60)

param_grid = {
    "learning_rate": [0.0001, 0.0005, 0.001],
    "dense_units": [64, 128, 256],
    "dropout_rate": [0.3, 0.5, 0.7],
    "optimizer": ["adam", "rmsprop"],
    "batch_size": [16, 32]
}

print("Parameter space:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

# Random search
from itertools import product
all_combinations = list(product(*param_grid.values()))
random.shuffle(all_combinations)
selected = all_combinations[:TRIALS]

print(f"\nTotal kombinasi: {len(all_combinations)}")
print(f"Trials: {TRIALS}")

# =========================================================
# 4. TUNING LOOP
# =========================================================
print("\n" + "=" * 60)
print("3. TUNING")
print("=" * 60)

results = []

# Simpan hasil sementara tiap 5 trial agar tidak hilang jika crash
def save_interim():
    with open(os.path.join(OUTPUT_DIR, "tuning_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "trial", "learning_rate", "dense_units", "dropout_rate",
            "optimizer", "batch_size", "best_val_accuracy", "best_epoch", "final_val_loss"
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"    (interim saved: {len(results)} trials)")

for trial_idx, (lr, units, dropout, opt_name, batch_size) in enumerate(selected, 1):
    print(f"\nTrial {trial_idx}/{TRIALS}")
    print(f"  lr={lr}, units={units}, dropout={dropout}, opt={opt_name}, batch={batch_size}")

    try:
        model = build_model_direct(lr, units, dropout, opt_name)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0
            )
        ]

        train_datagen = ImageDataGenerator(
            rotation_range=5, width_shift_range=0.1, height_shift_range=0.1,
            zoom_range=0.1, horizontal_flip=True, brightness_range=[0.9, 1.1],
            fill_mode="nearest"
        )

        history = model.fit(
            train_datagen.flow(X_train_proc, y_train_cat, batch_size=batch_size),
            validation_data=(X_val_proc, y_val_cat),
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        best_acc = max(history.history["val_accuracy"])
        best_epoch = history.history["val_accuracy"].index(best_acc) + 1
        final_loss = history.history["val_loss"][-1]

        results.append({
            "trial": trial_idx,
            "learning_rate": lr,
            "dense_units": units,
            "dropout_rate": dropout,
            "optimizer": opt_name,
            "batch_size": batch_size,
            "best_val_accuracy": round(float(best_acc), 4),
            "best_epoch": best_epoch,
            "final_val_loss": round(float(final_loss), 4)
        })

        print(f"  -> best_val_acc={best_acc:.4f} @ epoch {best_epoch}")

        # Bersihkan GPU memory
        del model
        keras.backend.clear_session()

        # Save interim setiap trial
        save_interim()

    except Exception as e:
        print(f"  -> ERROR: {e}")
        keras.backend.clear_session()
        continue

# =========================================================
# 5. SAVE RESULTS
# =========================================================
print("\n" + "=" * 60)
print("4. SAVE RESULTS")
print("=" * 60)

# Best trial
best = max(results, key=lambda r: r["best_val_accuracy"])
print(f"\nBest trial: {best['trial']}")
print(f"  val_accuracy: {best['best_val_accuracy']}")
print(f"  params: lr={best['learning_rate']}, units={best['dense_units']}, "
      f"dropout={best['dropout_rate']}, opt={best['optimizer']}, batch={best['batch_size']}")

# best_params.json
with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w") as f:
    json.dump({
        "learning_rate": best["learning_rate"],
        "dense_units": best["dense_units"],
        "dropout_rate": best["dropout_rate"],
        "optimizer": best["optimizer"],
        "batch_size": best["batch_size"]
    }, f, indent=2)
print("  -> best_params.json")

# summary.json
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump({
        "random_seed": SEED,
        "n_trials": TRIALS,
        "n_epochs": EPOCHS,
        "best_trial": best["trial"],
        "best_val_accuracy": best["best_val_accuracy"],
        "best_hyperparameters": {
            "learning_rate": best["learning_rate"],
            "dense_units": best["dense_units"],
            "dropout_rate": best["dropout_rate"],
            "optimizer": best["optimizer"],
            "batch_size": best["batch_size"]
        }
    }, f, indent=2)
print("  -> summary.json")

# tuning_results.csv
with open(os.path.join(OUTPUT_DIR, "tuning_results.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "trial", "learning_rate", "dense_units", "dropout_rate",
        "optimizer", "batch_size", "best_val_accuracy", "best_epoch", "final_val_loss"
    ])
    writer.writeheader()
    for r in sorted(results, key=lambda r: r["trial"]):
        writer.writerow(r)
print("  -> tuning_results.csv")

# =========================================================
# 6. TRAIN BEST MODEL & SAVE
# =========================================================
print("\n" + "=" * 60)
print("5. TRAIN BEST MODEL & SAVE")
print("=" * 60)

print("Training model dengan hyperparameter terbaik...")
best_model = build_model_direct(
    best["learning_rate"], best["dense_units"],
    best["dropout_rate"], best["optimizer"]
)

callbacks_best = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "mobilenetv2_tuned.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    )
]

train_datagen_best = ImageDataGenerator(
    rotation_range=5, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.1, horizontal_flip=True, brightness_range=[0.9, 1.1],
    fill_mode="nearest"
)

history_best = best_model.fit(
    train_datagen_best.flow(
        X_train_proc, y_train_cat, batch_size=best["batch_size"]
    ),
    validation_data=(X_val_proc, y_val_cat),
    epochs=50,
    callbacks=callbacks_best,
    verbose=1
)

# Save tuned model
model_path = os.path.join(MODELS_DIR, "mobilenetv2_tuned.h5")
best_model.save(model_path)
print(f"\nModel tuned saved: {model_path}")

# Load best checkpoint
best_checkpoint = keras.models.load_model(model_path)
val_acc = best_checkpoint.evaluate(X_val_proc, y_val_cat, verbose=0)[1]
print(f"Validation accuracy: {val_acc:.4f}")

# =========================================================
# 7. PLOT TUNING HISTORY
# =========================================================
print("\n" + "=" * 60)
print("6. PLOT TUNING HISTORY")
print("=" * 60)

sorted_results = sorted(results, key=lambda r: r["trial"])
trials = [r["trial"] for r in sorted_results]
accuracies = [r["best_val_accuracy"] for r in sorted_results]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Hyperparameter Tuning Results (RandomSearch)", fontsize=14, fontweight="bold")

# Plot 1: Accuracy per trial
axes[0].bar(trials, accuracies, color="#2196F3", alpha=0.8)
axes[0].axhline(y=best["best_val_accuracy"], color="red", linestyle="--",
                label=f"Best: {best['best_val_accuracy']:.4f}")
axes[0].set_xlabel("Trial")
axes[0].set_ylabel("Best Val Accuracy")
axes[0].set_title("Accuracy per Trial")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Learning rate vs accuracy
colors_lr = {"0.0001": "#E91E63", "0.0005": "#FF9800", "0.001": "#4CAF50"}
for r in sorted_results:
    axes[1].scatter(r["learning_rate"], r["best_val_accuracy"],
                   c=colors_lr.get(str(r["learning_rate"]), "#333"), s=80, alpha=0.7)
axes[1].set_xlabel("Learning Rate")
axes[1].set_ylabel("Best Val Accuracy")
axes[1].set_title("Learning Rate vs Accuracy")
axes[1].set_xscale("log")
axes[1].grid(True, alpha=0.3)

# Plot 3: Optimizer + batch_size vs accuracy
for r in sorted_results:
    marker = "o" if r["optimizer"] == "adam" else "s"
    axes[2].scatter(r["dense_units"], r["best_val_accuracy"],
                   s=80, alpha=0.7, marker=marker)
axes[2].set_xlabel("Dense Units")
axes[2].set_ylabel("Best Val Accuracy")
axes[2].set_title("Optimizer & Batch Size")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "tuning_history.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Tuning plot saved: {plot_path}")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("TUNING SELESAI")
print("=" * 60)
print(f"\nBest trial      : {best['trial']}")
print(f"Best val_acc    : {best['best_val_accuracy']:.4f}")
print(f"Learning rate   : {best['learning_rate']}")
print(f"Dense units     : {best['dense_units']}")
print(f"Dropout rate    : {best['dropout_rate']}")
print(f"Optimizer       : {best['optimizer']}")
print(f"Batch size      : {best['batch_size']}")
print(f"\nOutput files:")
print(f"  - models/mobilenetv2_tuned.h5")
print(f"  - best_params.json")
print(f"  - tuning_results.csv")
print(f"  - summary.json")
print(f"  - tuning_history.png")
print("=" * 60)
