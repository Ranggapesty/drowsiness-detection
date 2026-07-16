"""
=========================================================
TAHAP 5B: GRAD-CAM INTERPRETABILITY
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
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings
warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)

PROCESSED_DIR = r"D:\TA Mesin\tahap_2_Preprocessing\processed"
MODELS_DIR    = r"D:\TA Mesin\models"
OUTPUT_DIR    = r"D:\TA Mesin\tahap_5_Evaluasi"

IMG_SIZE = 64
NUM_CLASSES = 4
CLASSES = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]
N_SAMPLES_PER_CLASS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, layers.Conv2D):
                    return layer, sub.name
    return model, None

def make_gradcam_heatmap(img_array, model, base_model, conv_layer_name, pred_index=None):
    conv_layer = base_model.get_layer(conv_layer_name)

    sub_grad_model = keras.models.Model(
        inputs=base_model.inputs,
        outputs=[conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, base_out = sub_grad_model(img_array, training=False)

        x = base_out
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            if layer.name == base_model.name:
                continue
            if isinstance(layer, keras.layers.Dropout):
                x = layer(x, training=False)
            else:
                x = layer(x, training=False)

        if pred_index is None:
            pred_index = tf.argmax(x[0])
        class_channel = x[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

print("=" * 60)
print("GRAD-CAM VISUALIZATION")
print("=" * 60)

print("\n1. LOAD DATA")
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
print(f"X_test: {X_test.shape}")

X_test_proc = preprocess_input(X_test * 255.0)

print("\n2. LOAD MODEL")
model_path = os.path.join(MODELS_DIR, "mobilenetv2_best.h5")
if not os.path.exists(model_path):
    print(f"[ERROR] Model tidak ditemukan: {model_path}")
    exit(1)
model = keras.models.load_model(model_path)
print(f"Model loaded: {model_path}")

base_model, last_conv_name = find_last_conv_layer(model)
if last_conv_name is None:
    print("[ERROR] No Conv2D layer found")
    exit(1)
print(f"Last conv layer: {last_conv_name} (in {base_model.name})")

print("\n3. GENERATE GRAD-CAM")
np.random.seed(42)
fig, axes = plt.subplots(NUM_CLASSES, N_SAMPLES_PER_CLASS, figsize=(12, 12))
fig.suptitle("Grad-CAM: Class Activation Heatmaps", fontsize=14, fontweight="bold")

for cls_idx, cls_name in enumerate(CLASSES):
    class_indices = np.where(y_test == cls_idx)[0]
    selected = np.random.choice(class_indices, N_SAMPLES_PER_CLASS, replace=False)

    for col, idx in enumerate(selected):
        img = X_test[idx]
        img_proc = X_test_proc[idx:idx+1]

        preds = model.predict(img_proc, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = np.max(preds[0])

        heatmap = make_gradcam_heatmap(img_proc, model, base_model, last_conv_name)

        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE)
        ).numpy().squeeze()

        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]

        axes[cls_idx, col].imshow(img)
        axes[cls_idx, col].imshow(heatmap_colored, alpha=0.4)
        title = f"True: {cls_name}\nPred: {CLASSES[pred_class]} ({confidence:.2f})"
        color = "green" if pred_class == cls_idx else "red"
        axes[cls_idx, col].set_title(title, fontsize=8, color=color)
        axes[cls_idx, col].axis("off")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "gradcam_grid.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Grad-CAM grid saved: {output_path}")

print("\n" + "=" * 60)
print("GRAD-CAM SELESAI")
print("=" * 60)
