import os
import json
import numpy as np
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


def preprocess_input(x):
    """MobileNetV2 preprocessing: scale [0,255] to [-1,1]"""
    return (x / 127.5) - 1.0


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


# ---------------------------------------------------------------------------
# TFLite model (lightweight, works on Streamlit Cloud)
# ---------------------------------------------------------------------------
def _download_file(filename):
    import urllib.request
    import ssl
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(local_path):
        url = f"{HF_BASE}/{filename}"
        ctx = ssl._create_unverified_context()
        urllib.request.urlretrieve(url, local_path)
    return local_path


def _make_tflite(path):
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()


def load_model():
    try:
        path = _download_file("mobilenetv2_best.tflite")
        return _make_tflite(path)
    except Exception as e:
        return None


def load_tuned_model():
    try:
        path = _download_file("mobilenetv2_tuned.tflite")
        return _make_tflite(path)
    except Exception as e:
        return None


def predict_model(model, x):
    if isinstance(model, tuple):
        interpreter, input_details, output_details = model
        interpreter.set_tensor(input_details[0]["index"], x.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]["index"])
    return model.predict(x, verbose=0)


# ---------------------------------------------------------------------------
# Keras model (for Grad-CAM — requires tensorflow locally)
# ---------------------------------------------------------------------------
def load_keras_model(name="mobilenetv2_best.h5"):
    try:
        from tensorflow import keras
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            return keras.models.load_model(path)
        return None
    except ImportError:
        return None


def find_last_conv_layer(model):
    if model is None:
        return None, None
    try:
        from tensorflow import keras
    except ImportError:
        return None, None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, keras.layers.Conv2D):
                    return layer, sub.name
    return model, None


def make_gradcam_heatmap(img_array, model, base_model, conv_layer_name, pred_index=None):
    import tensorflow as tf
    from tensorflow import keras

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


# ---------------------------------------------------------------------------
# Dataset stats / evaluation data
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
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
