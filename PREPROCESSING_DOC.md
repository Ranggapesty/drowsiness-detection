# Dokumentasi Preprocessing

## 1. Tantangan Data

Dataset berisi citra crop dari dua bagian wajah yang berbeda:
- **Closed_Eyes & Open_Eyes**: Crop area mata — ukuran ~80-100px, aspek rasio hampir persegi
- **No_yawn & Yawn**: Crop area mulut — ukuran lebih kecil, aspek rasio lebih lebar (hingga 66×9)

Perbedaan ukuran dan aspek rasio ini membuat preprocessing naif (resize langsung ke 128×128) menghasilkan distorsi parah yang menyebabkan model gagal belajar.

## 2. Alur Preprocessing

```mermaid
flowchart LR
    A[Input Image] --> B{RGBA?}
    B -->|Ya| C[Hapus Alpha Channel]
    B -->|Tidak| D{Grayscale?}
    C --> D
    D -->|Ya| E[Konversi ke BGR 3-channel]
    D -->|Tidak| F[Letterbox Resize]
    E --> F
    F --> G[BGR → RGB]
    G --> H[Normalisasi [0,255] → [0,1]]
    H --> I[Output .npy]
```

## 3. Detail Setiap Langkah

### 3.1 Penanganan Channel

**Masalah:** Dataset campuran `.jpg` (3 channel BGR) dan `.png` (4 channel BGRA — ada alpha).  
**Solusi:** Jika `img.shape[-1] == 4`, ambil 3 channel pertama (BGR). Jika grayscale (`len(shape) == 2`), konversi ke BGR 3-channel via `cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)`.

**Justifikasi:** Model MobileNetV2 menerima input 3-channel RGB. Alpha channel tidak relevan untuk klasifikasi visual.

### 3.2 Letterbox Resize ke 64×64

**Masalah:** Gambar memiliki ukuran dan aspek rasio sangat bervariasi (28×21 hingga 146×138). Resize langsung ke square tanpa menjaga aspek rasio akan mendistorsi fitur visual.

**Solusi:** Letterbox padding:
```
scale = target_size / max(h, w)
new_w, new_h = int(w * scale), int(h * scale)
resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
canvas = zeros(target_size, target_size, 3)
canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
```

**Justifikasi teknik:**
- **INTER_AREA**: Terbaik untuk downsampling — mengurangi aliasing dan mempertahankan informasi
- **Padding hitam (0,0,0)**: Netral, tidak menambah informasi palsu
- **Ukuran 64×64**: Setelah eksperimen, 64×64 mempertahankan fitur penting (bukaan mata/mulut) dengan ukuran lebih kecil dari 128×128 sehingga training lebih cepat dan ringan di memory

### 3.3 Konversi BGR → RGB

**Masalah:** OpenCV membaca gambar dalam format BGR, sementara model TF di-training dengan format RGB.

**Solusi:** `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`

### 3.4 Normalisasi [0,255] → [0,1]

**Masalah:** Nilai piksel mentah (0-255) memiliki skala besar yang dapat membuat training tidak stabil.

**Solusi:** `img = img.astype(np.float32) / 255.0`

**Justifikasi:** Normalisasi ke [0,1] adalah praktik standar. Untuk MobileNetV2, preprocessing lebih lanjut dilakukan via `preprocess_input(x * 255.0)` yang memetakan ke [-1,1] sesuai dengan bobot pre-trained ImageNet.

### 3.5 Train-Val-Test Split

Dataset sudah tersedia dalam split Train/Val/Test dari sumber. Tidak dilakukan shuffling ulang.

| Split | Jumlah | Persentase |
|-------|--------|------------|
| Train | 8.548 | 73.9% |
| Val | 1.554 | 13.4% |
| Test | 1.464 | 12.7% |
| **Total** | **11.566** | **100%** |

## 4. Output

Semua hasil disimpan sebagai file `.npy` di `tahap_2_Preprocessing/processed/`:
- `X_train.npy`, `y_train.npy` — Training set (8548 gambar)
- `X_val.npy`, `y_val.npy` — Validation set (1554 gambar)
- `X_test.npy`, `y_test.npy` — Test set (1464 gambar)

Shape: `(N, 64, 64, 3)`, dtype: `float32`, range: `[0.0, 1.0]`
