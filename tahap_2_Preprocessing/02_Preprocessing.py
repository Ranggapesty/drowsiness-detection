"""
=========================================================
TAHAP 2: PREPROCESSING DATA
=========================================================
Proyek : Deteksi Kantuk Pengemudi (Drowsiness Detection)
Dataset: Drowsiness Dataset (Kaggle) - 11.566 gambar
          Closed_Eyes, Open_Eyes, No_yawn, Yawn

Tujuan Preprocessing:
1. Membaca semua gambar dari folder Train/Val/Test
2. Konversi RGBA -> RGB (khusus .png yang punya 4 channel)
3. Resize semua gambar ke ukuran seragam (128x128 piksel)
4. Normalisasi nilai piksel dari [0, 255] -> [0.0, 1.0]
5. Encoding label kategori ke numerik (0-3)
6. Menyimpan hasil preprocessing ke file .npy

Input : D:\TA Mesin\Data\Train, Val, Test
Output: D:\TA Mesin\tahap_2_Preprocessing\processed\*.npy
         - X_train.npy, y_train.npy
         - X_val.npy,   y_val.npy
         - X_test.npy,  y_test.npy

Ukuran output per gambar: 128x128x3 (float32)
=========================================================
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# KONFIGURASI
# =========================================================
DATA_DIR   = r"D:\TA Mesin\Data"
OUTPUT_DIR = r"D:\TA Mesin\tahap_2_Preprocessing\processed"
TARGET_SIZE = 64           # Ukuran target (square) — lebih kecil karena gambar asli kecil & aspek ratio bervariasi
CLASSES    = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]
SPLITS     = ["Train", "Val", "Test"]

# Mapping label kelas ke angka
LABEL_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}
# Hasil: {"Closed_Eyes": 0, "Open_Eyes": 1, "No_yawn": 2, "Yawn": 3}


def load_and_preprocess(split_name):
    """
    Membaca semua gambar dari satu split (Train/Val/Test),
    melakukan preprocessing, dan mengembalikan array numpy.

    Tahapan preprocessing per gambar:
    1. Baca gambar dengan OpenCV (format BGR)
    2. Jika 4 channel (RGBA/PNG) -> konversi ke 3 channel (RGB)
    3. Resize dengan aspect ratio preserved (long edge = TARGET_SIZE)
    4. Letterbox padding (tengah) untuk membuat square TARGET_SIZExTARGET_SIZE
    5. Normalisasi: bagi nilai piksel dengan 255.0 -> range [0, 1]

    Args:
        split_name (str): "Train", "Val", atau "Test"

    Returns:
        X (np.array): array gambar shape (N, TARGET_SIZE, TARGET_SIZE, 3) dtype float32
        y (np.array): array label shape (N,) dtype int
    """
    X = []   # Menampung gambar
    y = []   # Menampung label

    split_path = os.path.join(DATA_DIR, split_name)
    total_files = sum(len(files) for _, _, files in os.walk(split_path))
    pbar = tqdm(total=total_files, desc=f"Memproses {split_name}", unit="gambar")

    for cls_name in CLASSES:
        cls_path = os.path.join(split_path, cls_name)
        if not os.path.exists(cls_path):
            print(f"  [WARNING] Folder tidak ditemukan: {cls_path}")
            continue

        label = LABEL_MAP[cls_name]

        for fname in os.listdir(cls_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            fpath = os.path.join(cls_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"  [WARNING] Gambar corrupt, dilewati: {fpath}")
                pbar.update(1)
                continue

            # --- Langkah 1: Tangani alpha channel (RGBA -> RGB) ---
            if img.shape[-1] == 4:
                # .png dengan 4 channel (BGRA). Ambil 3 channel pertama (BGR)
                img = img[:, :, :3]
            elif len(img.shape) == 2:
                # Gambar grayscale -> konversi ke 3 channel
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # --- Langkah 2: Resize dengan aspect ratio preserved + letterbox padding ---
            h, w = img.shape[:2]
            scale = TARGET_SIZE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Buat canvas hitam, letakkan gambar di tengah
            canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=img.dtype)
            y_off = (TARGET_SIZE - new_h) // 2
            x_off = (TARGET_SIZE - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            img = canvas

            # --- Langkah 3: Konversi BGR -> RGB (biar warnanya natural) ---
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # --- Langkah 4: Normalisasi [0, 255] -> [0.0, 1.0] ---
            img = img.astype(np.float32) / 255.0

            X.append(img)
            y.append(label)
            pbar.update(1)

    pbar.close()

    # Konversi list ke numpy array
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"  -> {split_name}: {X.shape[0]} gambar, ukuran {X.shape[1]}x{X.shape[2]}")
    return X, y


# =========================================================
# EKSEKUSI UTAMA
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TAHAP 2: PREPROCESSING DATA")
    print("=" * 60)
    print(f"Konfigurasi:")
    print(f"  - Ukuran target : {TARGET_SIZE} x {TARGET_SIZE} piksel (letterbox padding)")
    print(f"  - Channel       : 3 (RGB)")
    print(f"  - Normalisasi   : [0, 255] -> [0.0, 1.0]")
    print(f"  - Label kelas   : {LABEL_MAP}")
    print(f"  - Output folder : {OUTPUT_DIR}")
    print("-" * 60)

    # Buat folder output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Proses setiap split
    data_splits = {}
    for split in SPLITS:
        print(f"\nMemproses {split}...")
        X, y = load_and_preprocess(split)
        data_splits[split] = (X, y)

        # Simpan ke file .npy
        np.save(os.path.join(OUTPUT_DIR, f"X_{split.lower()}.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, f"y_{split.lower()}.npy"), y)

    # =========================================================
    # RINGKASAN HASIL
    # =========================================================
    print("\n" + "=" * 60)
    print("RINGKASAN PREPROCESSING")
    print("=" * 60)

    total = 0
    for split in SPLITS:
        X, y = data_splits[split]
        n = len(X)
        total += n
        # Distribusi label
        unique, counts = np.unique(y, return_counts=True)
        distro = ", ".join([f"{CLASSES[u]}: {c}" for u, c in zip(unique, counts)])
        print(f"\n{split}: {n} gambar")
        print(f"  Shape: {X.shape}, dtype: {X.dtype}")
        print(f"  Distribusi: {distro}")

    print(f"\nTotal gambar diproses: {total}")
    print(f"Ukuran file: ~{total * 128 * 128 * 3 * 4 / 1024**3:.2f} GB (float32)")

    print("\n" + "=" * 60)
    print("FILE OUTPUT:")
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_mb = os.path.getsize(fpath) / 1024**2
        print(f"  - {fname} ({size_mb:.2f} MB)")

    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI")
    print("=" * 60)
    print("\nFile .npy siap digunakan untuk tahap selanjutnya (Augmentasi & Training).")
    print("Cara load di script lain:")
    print('  X_train = np.load(r"tahap_2_Preprocessing/processed/X_train.npy")')
    print('  y_train = np.load(r"tahap_2_Preprocessing/processed/y_train.npy")')
