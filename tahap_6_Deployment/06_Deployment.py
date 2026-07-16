"""
=========================================================
TAHAP 6: DEPLOYMENT - REAL-TIME DROWSINESS DETECTION
=========================================================
Proyek : Deteksi Kantuk Pengemudi (Drowsiness Detection)

Fitur:
1. Buka webcam dan capture frame real-time
2. Deteksi wajah & mata via OpenCV Haar Cascade
3. Ekstrak ROI mata kiri, mata kanan, dan mulut
4. Preprocessing letterbox 64x64 + inferensi MobileNetV2
5. Tampilkan status per region di layar
6. Alarm suara jika terdeteksi kantuk (Closed_Eyes / Yawn)
7. Logging ke file CSV

Cara pakai:
  python 06_Deployment.py

Tekan 'q' untuk keluar, 'p' untuk pause
=========================================================
"""

import os
# Tambahkan CUDA DLL path ke PATH (wajib untuk TF di Windows)
cuda_path = r"D:\conda_envs\drowsiness\Library\bin"
os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(cuda_path)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
import csv
from datetime import datetime
import winsound
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# KONFIGURASI
# =========================================================
MODEL_PATH = r"D:\TA Mesin\models\mobilenetv2_best.h5"
LOG_PATH   = r"D:\TA Mesin\tahap_6_Deployment\detection_log.csv"

IMG_SIZE = 64
NUM_CLASSES = 4
CLASSES = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]

CONFIDENCE_THRESHOLD = 0.6
EYE_PADDING = 0.5                # 50% padding untuk crop mata
MOUTH_PADDING = 0.3              # 30% padding untuk crop mulut
ALARM_DURATION = 500
ALARM_COOLDOWN = 2.0
DROWSY_STREAK_THRESHOLD = 3
NO_FACE_TIMEOUT = 10

frame_count = 0

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================
print("=" * 60)
print("DROWSINESS DETECTION - DEPLOYMENT")
print("=" * 60)
print(f"\nLoading model: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
print(f"Input shape: {model.input_shape}")

# =========================================================
# INITIALIZE HAAR CASCADE
# =========================================================
haar_dir = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(haar_dir + "haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    print("[ERROR] Gagal memuat Haar Cascade!")
    exit(1)
print("Haar Cascade loaded successfully!")

# =========================================================
# FUNGSI EKSTRAKSI ROI
# =========================================================
def extract_regions(frame):
    """Deteksi wajah & ekstrak ROI mata kiri, mata kanan, mulut via Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        return None

    # Ambil wajah terbesar (paling dekat)
    (fx, fy, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
    face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
    face_roi_color = frame[fy:fy+fh, fx:fx+fw]

    # Deteksi mata di ROI wajah
    eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    if len(eyes) < 2:
        return None

    # Urutkan mata berdasarkan posisi x (kiri ke kanan)
    eyes = sorted(eyes, key=lambda e: e[0])

    # Ambil 2 mata paling kiri dan kanan (untuk menghindari false positive di tengah)
    left_eye = eyes[0]
    right_eye = eyes[-1]

    regions = {}

    # Crop mata kiri
    ex, ey, ew, eh = left_eye
    pad = int(EYE_PADDING * max(ew, eh))
    x1 = max(0, fx + ex - pad)
    y1 = max(0, fy + ey - pad)
    x2 = min(frame.shape[1], fx + ex + ew + pad)
    y2 = min(frame.shape[0], fy + ey + eh + pad)
    regions["left_eye"] = frame[y1:y2, x1:x2]

    # Crop mata kanan
    ex, ey, ew, eh = right_eye
    pad = int(EYE_PADDING * max(ew, eh))
    x1 = max(0, fx + ex - pad)
    y1 = max(0, fy + ey - pad)
    x2 = min(frame.shape[1], fx + ex + ew + pad)
    y2 = min(frame.shape[0], fy + ey + eh + pad)
    regions["right_eye"] = frame[y1:y2, x1:x2]

    # Crop mulut (1/3 bawah wajah)
    mouth_y1 = fy + int(fh * 0.55)
    mouth_y2 = fy + int(fh * 0.90)
    mouth_x1 = fx + int(fw * 0.15)
    mouth_x2 = fx + int(fw * 0.85)
    # Tambah padding vertikal
    mh = mouth_y2 - mouth_y1
    pad = int(MOUTH_PADDING * mh)
    mouth_y1 = max(0, mouth_y1 - pad)
    mouth_y2 = min(frame.shape[0], mouth_y2 + pad)
    regions["mouth"] = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

    return regions


# =========================================================
# FUNGSI PREPROCESSING (sama seperti tahap 2)
# =========================================================
def preprocess_roi(roi):
    """Letterbox padding 64x64 + preprocess MobileNetV2 untuk satu ROI"""
    h, w = roi.shape[:2]
    scale = IMG_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    y_off = (IMG_SIZE - new_h) // 2
    x_off = (IMG_SIZE - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    canvas = canvas.astype(np.float32) / 255.0
    canvas = preprocess_input(canvas * 255.0)
    return canvas[np.newaxis, ...]


# =========================================================
# FUNGSI ALARM
# =========================================================
last_alarm_time = 0

def trigger_alarm():
    """Trigger audio alarm (beep)"""
    global last_alarm_time
    now = time.time()
    if now - last_alarm_time > ALARM_COOLDOWN:
        winsound.Beep(880, ALARM_DURATION)
        last_alarm_time = now


# =========================================================
# FUNGSI LOGGING
# =========================================================
def log_detection(frame_num, left_status, right_status, mouth_status, is_drowsy):
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_num,
            left_status, right_status, mouth_status,
            int(is_drowsy)
        ])

# Inisialisasi CSV header
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame", "left_eye", "right_eye", "mouth", "is_drowsy"])


# =========================================================
# MAIN LOOP
# =========================================================
print("\nMembuka webcam...")
print("Tekan 'q' untuk keluar, 'p' untuk pause")
print("-" * 60)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Tidak dapat membuka webcam!")
    print("Menggunakan dummy frame untuk testing...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = type("DummyCap", (), {"read": lambda self: (True, dummy.copy()),
                                 "release": lambda self: None})()

paused = False
drowsy_streak = 0
no_face_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame")
        break

    if paused:
        cv2.putText(frame, "PAUSED - Tekan 'p' untuk lanjut",
                    (frame.shape[1]//2 - 200, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = False
        continue

    frame_count += 1
    frame_display = frame.copy()
    t_start = time.time()

    # === Ekstraksi ROI wajah via Haar Cascade ===
    regions = extract_regions(frame)

    if regions is None:
        no_face_counter += 1
        if no_face_counter >= NO_FACE_TIMEOUT:
            drowsy_streak = 0

        inference_time = (time.time() - t_start) * 1000

        overlay = frame_display.copy()
        cv2.rectangle(overlay, (5, 5), (350, 100), (0, 0, 0), -1)
        frame_display = cv2.addWeighted(overlay, 0.6, frame_display, 0.4, 0)
        cv2.putText(frame_display, "No Face Detected", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_display, f"Frame: {frame_count}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    else:
        no_face_counter = 0

        # Batch inference 3 ROI
        rois = ["left_eye", "right_eye", "mouth"]
        batch = np.vstack([preprocess_roi(regions[r]) for r in rois])
        preds = model.predict(batch, verbose=0)

        # Evaluasi per region
        # Left eye
        left_pred = np.argmax(preds[0])
        left_closed = (left_pred == 0) and (preds[0][0] >= CONFIDENCE_THRESHOLD)
        left_confidence = preds[0][0] if left_closed else (1 - preds[0][0])
        left_status = "CLOSED" if left_closed else "OPEN"

        # Right eye
        right_pred = np.argmax(preds[1])
        right_closed = (right_pred == 0) and (preds[1][0] >= CONFIDENCE_THRESHOLD)
        right_confidence = preds[1][0] if right_closed else (1 - preds[1][0])
        right_status = "CLOSED" if right_closed else "OPEN"

        # Mouth
        mouth_pred = np.argmax(preds[2])
        mouth_yawn = (mouth_pred == 3) and (preds[2][3] >= CONFIDENCE_THRESHOLD)
        mouth_confidence = preds[2][3] if mouth_yawn else (1 - preds[2][2])
        mouth_status = "YAWN" if mouth_yawn else "NO YAWN"

        # Status keseluruhan
        any_eye_closed = left_closed or right_closed
        is_drowsy = any_eye_closed or mouth_yawn

        inference_time = (time.time() - t_start) * 1000

        if is_drowsy:
            drowsy_streak += 1
            if drowsy_streak >= DROWSY_STREAK_THRESHOLD:
                trigger_alarm()
        else:
            drowsy_streak = 0

        # === Tampilkan informasi ===
        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        status_text = "DROWSY!" if is_drowsy else "AWAKE"

        overlay = frame_display.copy()
        cv2.rectangle(overlay, (5, 5), (420, 170), (0, 0, 0), -1)
        frame_display = cv2.addWeighted(overlay, 0.6, frame_display, 0.4, 0)

        cv2.putText(frame_display, f"Status: {status_text}  streak={drowsy_streak}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame_display, f"Left Eye : {left_status:7s}  conf={left_confidence:.0%}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_display, f"Right Eye: {right_status:7s}  conf={right_confidence:.0%}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_display, f"Mouth    : {mouth_status:7s}  conf={mouth_confidence:.0%}", (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_display, f"Frame: {frame_count} | {inference_time:.0f}ms", (15, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Tampilkan 3 ROI kecil di pojok kanan
        for i, name in enumerate(rois):
            roi_rgb = cv2.cvtColor(regions[name], cv2.COLOR_BGR2RGB)
            roi_small = cv2.resize(roi_rgb, (70, 70))
            y_off = 5 + i * 75
            x_off = frame_display.shape[1] - 75
            frame_display[y_off:y_off+70, x_off:x_off+70] = roi_small

        # Gambar kotak deteksi wajah
        if frame_count % 5 == 0:
            # Gambar kotak wajah setiap 5 frame (hemat CPU)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                (fx, fy, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
                cv2.rectangle(frame_display, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)

    # Logging setiap 30 frame
    if frame_count % 30 == 0 and regions is not None:
        log_detection(frame_count, left_status, right_status, mouth_status, is_drowsy)
        print(f"[{frame_count}] L={left_status[0]} R={right_status[0]} M={mouth_status[0]} | "
              f"drowsy={is_drowsy} | streak={drowsy_streak} | {inference_time:.0f}ms")

    cv2.imshow("Drowsiness Detection", frame_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("\nKeluar...")
        break
    elif key == ord("p"):
        paused = True

# =========================================================
# CLEANUP
# =========================================================
cap.release()
cv2.destroyAllWindows()

print(f"\nTotal frame diproses: {frame_count}")
print(f"Log tersimpan di: {LOG_PATH}")
print("=" * 60)
print("DEPLOYMENT SELESAI")
print("=" * 60)
