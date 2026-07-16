"""
=========================================================
TAHAP 4: MODELING & TRAINING
=========================================================
Proyek : Deteksi Kantuk Pengemudi (Drowsiness Detection)

Model yang akan dilatih:
1. CNN Custom    - Arsitektur buatan sendiri (fleksibel)
2. MobileNetV2   - Transfer Learning (ringan, cepat)

Alur training:
1. Load data dari .npy (hasil preprocessing)
2. Define arsitektur model
3. Compile dengan optimizer Adam
4. Train dengan augmentasi (ImageDataGenerator)
5. EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
6. Simpan model terbaik (.h5)
7. Plot history training (loss & accuracy)

Output di folder ini:
- models/cnn_custom_best.h5
- models/mobilenetv2_best.h5
- history_cnn_custom.npy
- history_mobilenetv2.npy
- training_curves.png
=========================================================
"""

import os
# Tambah path CUDA DLL agar GPU terdeteksi
os.add_dll_directory(r"D:\conda_envs\drowsiness\Library\bin")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
import warnings
warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)

# =========================================================
# KONFIGURASI
# =========================================================
PROCESSED_DIR = r"D:\TA Mesin\tahap_2_Preprocessing\processed"
MODELS_DIR    = r"D:\TA Mesin\models"
OUTPUT_DIR    = r"D:\TA Mesin\tahap_4_Modeling"

IMG_SIZE = 64
NUM_CLASSES = 4
CLASSES = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================
print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)
print()

print("Memuat data dari .npy...")
X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
X_val   = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# One-hot encode labels untuk categorical crossentropy
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat   = keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test_cat  = keras.utils.to_categorical(y_test, NUM_CLASSES)

# =========================================================
# 2. AUGMENTASI (real-time saat training)
# =========================================================
print()
print("=" * 60)
print("2. AUGMENTASI")
print("=" * 60)
print()

train_datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode="nearest"
)

# Val/Test: tanpa augmentasi
val_datagen = ImageDataGenerator()

print("Train augmentasi: rotation=5, shift=0.1, zoom=0.1, flip=True, brightness=[0.9,1.1]")
print("Val/Test: tanpa augmentasi")


# =========================================================
# 3. CALLBACKS
# =========================================================
def get_callbacks(model_name):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{model_name}_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]


# =========================================================
# 4A. MODEL: CNN CUSTOM
# =========================================================
def build_cnn_custom():
    """
    Arsitektur CNN Custom (VGG-style sederhana):
    - 3 blok Conv2D + MaxPooling + Dropout
    - Filter: 32 -> 64 -> 128
    - Flatten + Dense 128 + Dropout 50%
    - Output 4 kelas (softmax)
    """
    model = models.Sequential(name="CNN_Custom")

    model.add(layers.Conv2D(32, 3, activation="relu", padding="same",
                            input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, 3, activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return model


# =========================================================
# 4B. MODEL: MOBILENETV2 (TRANSFER LEARNING)
# =========================================================
def build_mobilenetv2():
    """
    Arsitektur MobileNetV2 (Transfer Learning):
    - Base model: MobileNetV2 (pre-trained ImageNet) — weights frozen
    - GlobalAveragePooling2D
    - Dense layer 128
    - Dropout 50%
    - Output 4 kelas (softmax)

    Kelebihan: ringan, cepat, cocok untuk real-time di perangkat terbatas
    """
    # Load pre-trained MobileNetV2 (tanpa top layer)
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base model (tidak dilatih ulang)
    base_model.trainable = False

    # Bangun model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2")

    return model


# =========================================================
# 5. FUNGSI TRAINING
# =========================================================
def train_model(model, model_name, X_tr, y_tr, X_v, y_v):
    """
    Melatih model dengan augmentasi dan callbacks.
    """
    print()
    print("=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)
    print()

    model.summary()

    # Compile (dengan gradient clipping untuk mencegah loss explosion)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = get_callbacks(model_name)

    # Training
    print(f"\nMemulai training {model_name}...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")

    history = model.fit(
        train_datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE),
        validation_data=(X_v, y_v),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return history


def train_model_mobilenet(model, model_name, X_tr, y_tr, X_v, y_v):
    """
    Melatih MobileNetV2 dengan preprocessing yang benar:
    input [0,1] -> [0,255] -> [-1,1] via preprocess_input
    """
    print()
    print("=" * 60)
    print(f"TRAINING: {model_name} (dengan preprocess_input)")
    print("=" * 60)
    print()

    # Preprocess: [0,1] -> [0,255] -> [-1,1]
    print("Mengkonversi input ke [-1,1] untuk MobileNetV2...")
    X_tr_proc = preprocess_mobilenet(X_tr * 255.0)
    X_v_proc = preprocess_mobilenet(X_v * 255.0)

    model.summary()

    # Compile (tanpa gradient clipping untuk transfer learning — lebih stabil)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks (patience lebih kecil untuk fine-tuning)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=os.path.join(MODELS_DIR, f"{model_name}_best.h5"),
                       monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    print(f"\nMemulai training {model_name}...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: 0.0005")

    # Gunakan augmentasi pada data yang sudah dipreprocess
    train_gen_mobilenet = ImageDataGenerator(
        rotation_range=5, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True, brightness_range=[0.9, 1.1],
        fill_mode="nearest"
    )

    history = model.fit(
        train_gen_mobilenet.flow(X_tr_proc, y_tr, batch_size=BATCH_SIZE),
        validation_data=(X_v_proc, y_v),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return history


# =========================================================
# 6. PLOT HISTORY
# =========================================================
def plot_training_history(histories):
    """
    Plot perbandingan accuracy & loss kedua model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Perbandingan Training: CNN Custom vs MobileNetV2", fontsize=14, fontweight="bold")

    colors = {"CNN_Custom": "#2196F3", "MobileNetV2": "#FF9800"}

    for model_name, history in histories.items():
        color = colors.get(model_name, "#333")

        # Accuracy
        axes[0, 0].plot(history.history["accuracy"], label=f"{model_name} (train)", color=color, linestyle="-")
        axes[0, 0].plot(history.history["val_accuracy"], label=f"{model_name} (val)", color=color, linestyle="--")
        axes[0, 0].set_title("Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(history.history["loss"], label=f"{model_name} (train)", color=color, linestyle="-")
        axes[0, 1].plot(history.history["val_loss"], label=f"{model_name} (val)", color=color, linestyle="--")
        axes[0, 1].set_title("Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Gap analysis (overfitting detection)
    for i, model_name in enumerate(histories.keys()):
        history = histories[model_name]
        gap = np.array(history.history["accuracy"]) - np.array(history.history["val_accuracy"])
        axes[1, i].plot(gap, color=colors.get(model_name, "#333"))
        axes[1, i].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        axes[1, i].set_title(f"Gap Train-Val Accuracy ({model_name})")
        axes[1, i].set_xlabel("Epoch")
        axes[1, i].set_ylabel("Gap")
        axes[1, i].grid(True, alpha=0.3)
        # Tambah teks kesimpulan
        avg_gap = np.mean(gap[-10:])
        if avg_gap > 0.1:
            axes[1, i].text(0.5, 0.9, f"Overfitting (gap={avg_gap:.3f})",
                          transform=axes[1, i].transAxes, ha="center", color="red")
        else:
            axes[1, i].text(0.5, 0.9, f"Good (gap={avg_gap:.3f})",
                          transform=axes[1, i].transAxes, ha="center", color="green")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGrafik training disimpan di: {plot_path}")


# =========================================================
# EKSEKUSI UTAMA
# =========================================================
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("TAHAP 4: MODELING & TRAINING")
    print("=" * 60)

    histories = {}

    # --- Train CNN Custom ---
    print()
    print("-" * 60)
    print("MEMBANGUN & MELATIH: CNN Custom")
    print("-" * 60)
    cnn_model = build_cnn_custom()
    # CNN Custom pakai learning rate lebih rendah untuk stabilitas
    original_lr = LEARNING_RATE
    LEARNING_RATE = 0.0005
    hist_cnn = train_model(cnn_model, "cnn_custom", X_train, y_train_cat, X_val, y_val_cat)
    LEARNING_RATE = original_lr
    histories["CNN_Custom"] = hist_cnn

    # Simpan history
    np.save(os.path.join(OUTPUT_DIR, "history_cnn_custom.npy"), hist_cnn.history)
    print(f"History CNN Custom disimpan.")

    # --- Train MobileNetV2 ---
    print()
    print("-" * 60)
    print("MEMBANGUN & MELATIH: MobileNetV2 (dengan preprocess_input)")
    print("-" * 60)
    mobilenet_model = build_mobilenetv2()
    hist_mobilenet = train_model_mobilenet(mobilenet_model, "mobilenetv2", X_train, y_train_cat, X_val, y_val_cat)
    histories["MobileNetV2"] = hist_mobilenet

    # Simpan history
    np.save(os.path.join(OUTPUT_DIR, "history_mobilenetv2.npy"), hist_mobilenet.history)
    print(f"History MobileNetV2 disimpan.")

    # --- Plot perbandingan ---
    print()
    print("=" * 60)
    print("PLOT PERBANDINGAN TRAINING")
    print("=" * 60)
    plot_training_history(histories)

    # --- Ringkasan ---
    print()
    print("=" * 60)
    print("RINGKASAN HASIL TRAINING")
    print("=" * 60)

    for model_name, history in histories.items():
        best_acc = max(history.history["val_accuracy"])
        best_epoch = history.history["val_accuracy"].index(best_acc) + 1
        final_loss = history.history["val_loss"][-1]
        print(f"\n{model_name}:")
        print(f"  Best val_accuracy: {best_acc:.4f} (epoch {best_epoch})")
        print(f"  Final val_loss:    {final_loss:.4f}")
        print(f"  Total epochs:      {len(history.history['val_accuracy'])}")

    print()
    print("=" * 60)
    print("FILE OUTPUT:")
    print(f"  - models/cnn_custom_best.h5     (model terbaik CNN Custom)")
    print(f"  - models/mobilenetv2_best.h5    (model terbaik MobileNetV2)")
    print(f"  - tahap_4_Modeling/history_cnn_custom.npy")
    print(f"  - tahap_4_Modeling/history_mobilenetv2.npy")
    print(f"  - tahap_4_Modeling/training_curves.png")
    print("=" * 60)
    print()
    print("TAHAP 4 (MODELING & TRAINING) SELESAI")
