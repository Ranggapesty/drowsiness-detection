"""
=========================================================
utils.py — Shared utilities for Streamlit app
=========================================================
Proyek : Deteksi Kantuk Pengemudi (Drowsiness Detection)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = r"D:\TA Mesin\tahap_2_Preprocessing\processed"
MODELS_DIR = r"D:\TA Mesin\models"
EVAL_DIR = r"D:\TA Mesin\tahap_5_Evaluasi"
MODELING_DIR = r"D:\TA Mesin\tahap_4_Modeling"
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")

HF_REPO = "Ranggapesty/drowsiness-models"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"

IMG_SIZE = 64
NUM_CLASSES = 4
CLASSES = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]


def load_data():
    if not os.path.exists(PROCESSED_DIR):
        return None, None, None, None, None, None
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def _download_model(filename):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        url = f"{HF_BASE}/{filename}"
        path = keras.utils.get_file(filename, url, cache_dir=CACHE_DIR, extract=False)
    return keras.models.load_model(path)


def load_model():
    try:
        return _download_model("mobilenetv2_best.h5")
    except Exception as e:
        local = os.path.join(MODELS_DIR, "mobilenetv2_best.h5")
        if os.path.exists(local):
            return keras.models.load_model(local)
        return None


def load_tuned_model():
    try:
        return _download_model("mobilenetv2_tuned.h5")
    except Exception as e:
        local = os.path.join(MODELS_DIR, "mobilenetv2_tuned.h5")
        if os.path.exists(local):
            return keras.models.load_model(local)
        return None


def load_dataset_stats():
    path = os.path.join(BASE_DIR, "assets", "dataset_stats.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_tuning_results():
    path = os.path.join(MODELING_DIR, "tuning_results.csv")
    if not os.path.exists(path):
        return None
    import pandas as pd
    return pd.read_csv(path)


def load_test_results():
    path = os.path.join(EVAL_DIR, "test_results.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def load_classification_report():
    path = os.path.join(EVAL_DIR, "classification_report.txt")
    if not os.path.exists(path):
        return "Report not available."
    with open(path) as f:
        return f.read()


def preprocess_image_pil(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    scale = IMG_SIZE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    x_off = (IMG_SIZE - new_w) // 2
    y_off = (IMG_SIZE - new_h) // 2
    canvas.paste(img, (x_off, y_off))
    return np.array(canvas, dtype=np.float32) / 255.0


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
            if isinstance(layer, keras.layers.InputLayer): continue
            if layer.name == base_model.name: continue
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


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, keras.layers.Conv2D):
                    return layer, sub.name
    return model, None
