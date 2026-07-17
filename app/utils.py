import os
import json
import struct
import math
import io
import wave
import base64
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

# Drowsy classes that trigger alarm
DROWSY_CLASSES = {0, 3}  # Closed_Eyes, Yawn


def _generate_beep(freq=800, duration=0.25, sample_rate=22050):
    buf = io.BytesIO()
    w = wave.open(buf, 'wb')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    for i in range(int(sample_rate * duration)):
        val = int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
        w.writeframes(struct.pack('<h', val))
    w.close()
    return base64.b64encode(buf.getvalue()).decode()


BEEP_BASE64 = _generate_beep()


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
    local_path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(local_path):
        url = f"{HF_BASE}/{filename}"
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        local_path = keras.utils.get_file(filename, url, cache_dir=CACHE_DIR, extract=False)
    return local_path


def load_model():
    try:
        path = _download_model("mobilenetv2_best.h5")
        return keras.models.load_model(path)
    except Exception:
        local = os.path.join(MODELS_DIR, "mobilenetv2_best.h5")
        if os.path.exists(local):
            return keras.models.load_model(local)
        return None


def load_tuned_model():
    try:
        path = _download_model("mobilenetv2_tuned.h5")
        return keras.models.load_model(path)
    except Exception:
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


def predict_model(model, x):
    return model.predict(x, verbose=0)


# ---------------------------------------------------------------------------
# Audio alarm
# ---------------------------------------------------------------------------
def alarm_html():
    """Returns HTML with autoplay beep for drowsy detection."""
    return f'<audio autoplay src="data:audio/wav;base64,{BEEP_BASE64}">'


def is_drowsy(pred_class):
    return pred_class in DROWSY_CLASSES


# ---------------------------------------------------------------------------
# Real-time video processor (streamlit-webrtc)
# ---------------------------------------------------------------------------
_MODEL_LOCK = None
_PREDICTOR = None


def _init_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        import threading
        global _MODEL_LOCK
        _MODEL_LOCK = threading.Lock()
        with _MODEL_LOCK:
            m = load_model()
            if m is not None:
                _PREDICTOR = m
    return _PREDICTOR


def process_frame(frame):
    try:
        import av
    except ImportError:
        return frame

    model = _init_predictor()
    if model is None:
        return frame

    img = frame.to_ndarray(format="bgr24")
    h, w = img.shape[:2]

    pil_img = Image.fromarray(img[:, :, ::-1])
    scale = IMG_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    x_off = (IMG_SIZE - new_w) // 2
    y_off = (IMG_SIZE - new_h) // 2
    canvas.paste(resized, (x_off, y_off))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    x_input = preprocess_input(arr[np.newaxis, ...] * 255.0)
    preds = predict_model(model, x_input)
    cls_idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    label = f"{CLASSES[cls_idx]}: {conf:.2f}"
    drowsy = is_drowsy(cls_idx)

    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil_img)
    color = (255, 0, 0) if drowsy else (0, 255, 0)
    draw.text((10, 30), label, fill=color)
    if drowsy:
        draw.text((10, 70), "DROWSY!", fill=(255, 0, 0))

    out = np.array(pil_img)[:, :, ::-1]
    return av.VideoFrame.from_ndarray(out, format="bgr24")
