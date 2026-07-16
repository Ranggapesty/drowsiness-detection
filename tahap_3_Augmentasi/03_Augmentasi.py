"""
=========================================================
TAHAP 3: AUGMENTASI DATA
=========================================================
Proyek : Deteksi Kantuk Pengemudi (Drowsiness Detection)

Apa itu Augmentasi Data?
Augmentasi adalah teknik memperbanyak variasi data training
dengan memodifikasi gambar asli secara acak, TANPA mengubah
label kelas. Augmentasi dilakukan REAL-TIME saat training,
bukan menyimpan file augmentasi ke disk.

Kenapa diperlukan?
- Dataset training hanya 8.548 gambar
- Model jadi lebih robust terhadap variasi kondisi real
- Mencegah overfitting (model hafal, bukan belajar)
- Mensimulasikan kondisi: kepala miring, cahaya redup, dll

Parameter Augmentasi:
- rotation_range     : ±10°  (simulasi kepala miring)
- width_shift_range  : 10%   (posisi wajah tidak di tengah)
- height_shift_range : 10%   (posisi wajah tidak di tengah)
- zoom_range         : ±10%  (jarak wajah ke kamera)
- horizontal_flip    : Ya    (posisi kiri/kanan)
- brightness_range   : 0.8-1.2 (pencahayaan kabin)

Output: Visualisasi (gambar asli vs augmented) untuk verifikasi

Catatan PENTING:
- Augmentasi HANYA untuk TRAIN set
- VAL dan TEST tetap asli (tanpa augmentasi) untuk evaluasi fair
- Augmentasi diterapkan live oleh ImageDataGenerator saat training
=========================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# KONFIGURASI
# =========================================================
PROCESSED_DIR = r"D:\TA Mesin\tahap_2_Preprocessing\processed"
OUTPUT_DIR    = r"D:\TA Mesin\tahap_3_Augmentasi"

IMG_SIZE = 128
CLASSES  = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]

# Parameter augmentasi
AUG_PARAMS = {
    "rotation_range": 10,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest"
}


def visualize_augmentation():
    """
    Memuat sample gambar dari X_train.npy dan menampilkan
    perbandingan gambar ASLI vs hasil AUGMENTASI.
    """
    print("=" * 60)
    print("TAHAP 3: AUGMENTASI DATA")
    print("=" * 60)

    # Load data training
    print("\nMemuat X_train.npy...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    print(f"  Shape: {X_train.shape}")
    print(f"  Range: [{X_train.min():.2f}, {X_train.max():.2f}]")

    # Buat ImageDataGenerator dengan parameter augmentasi
    datagen = ImageDataGenerator(**AUG_PARAMS)

    # Ambil 4 sample (1 dari setiap kelas)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Perbandingan: Gambar Asli (kiri) vs 3 Hasil Augmentasi (kanan)",
                 fontsize=14, fontweight="bold")

    print("\nParameter Augmentasi:")
    for key, val in AUG_PARAMS.items():
        print(f"  - {key}: {val}")

    for i, cls_name in enumerate(CLASSES):
        # Cari index gambar pertama dari kelas ini
        idx = np.where(y_train == i)[0][0]
        img = X_train[idx]

        # Tampilkan gambar asli (kolom 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"ASLI: {cls_name}", fontsize=9)
        axes[i, 0].axis("off")

        # Hasilkan 3 gambar augmented
        img_batch = img.reshape((1, IMG_SIZE, IMG_SIZE, 3))
        aug_iter = datagen.flow(img_batch, batch_size=1)

        for j in range(3):
            aug_img = next(aug_iter)[0]
            axes[i, j + 1].imshow(aug_img)
            axes[i, j + 1].set_title(f"Augmentasi #{j+1}", fontsize=9)
            axes[i, j + 1].axis("off")

        print(f"\n  Sample {cls_name}: index {idx}")

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "augmentasi_sample.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVisualisasi disimpan di: {output_path}")

    # =========================================================
    # SIMPAN KONFIGURASI AUGMENTASI UNTUK DIGUNAKAN DI TRAINING
    # =========================================================
    print("\n" + "=" * 60)
    print("KONFIGURASI AUGMENTASI UNTUK TRAINING")
    print("=" * 60)
    print("""
Cara pakai di script training :

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Untuk TRAIN (dengan augmentasi)
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    # Untuk VAL/TEST (TANPA augmentasi, hanya normalisasi)
    test_datagen = ImageDataGenerator()

    # Training
    model.fit(
        train_datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=50
    )
    """)

    print("\n" + "=" * 60)
    print("TAHAP 3 (AUGMENTASI) SELESAI")
    print("=" * 60)
    print("\nFile .npy tetap utuh. Augmentasi akan diterapkan LIVE saat training.")


if __name__ == "__main__":
    visualize_augmentation()
