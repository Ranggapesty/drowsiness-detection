# Drowsiness Detection — Deteksi Kantuk Pengemudi

Sistem deteksi kantuk berbasis deep learning yang mengklasifikasikan kondisi pengemudi ke dalam 4 kelas: **Closed_Eyes**, **Open_Eyes**, **No_yawn**, **Yawn**.

## 📋 Daftar Isi

- [Ringkasan Proyek](#ringkasan-proyek)
- [Dataset](#dataset)
- [Arsitektur](#arsitektur)
- [Hasil](#hasil)
- [Struktur Folder](#struktur-folder)
- [Cara Menjalankan](#cara-menjalankan)
- [Deployment](#deployment)
- [Tim](#tim)

## Ringkasan Proyek

Proyek ini menggunakan **MobileNetV2** (transfer learning) untuk mendeteksi kantuk pengemudi melalui citra wajah. Model mencapai **93.85% test accuracy** dan **F1-score 0.9385**.

Pipeline:
1. **Preprocessing** — Letterbox padding 64×64, normalisasi [0,1]
2. **Augmentasi** — Rotation, shift, zoom, flip, brightness
3. **Modeling** — CNN Custom vs MobileNetV2 (Transfer Learning)
4. **Hyperparameter Tuning** — RandomSearch (12 trials)
5. **Evaluasi** — Confusion matrix, ROC-AUC, classification report
6. **Interpretasi** — Grad-CAM heatmap
7. **Deployment** — Real-time detection via webcam + Streamlit app

## Dataset

**Sumber:** [Drowsiness Dataset - Kaggle](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)

> **⚠️ Dataset tidak termasuk dalam repository git.**  
> Download dari Kaggle, lalu extract ke folder `Data/` di root proyek.  
> Struktur yang diharapkan: `Data/train/Closed_Eyes/`, `Data/train/Open_Eyes/`, dll.

| Split | Closed_Eyes | Open_Eyes | No_yawn | Yawn | Total |
|-------|-------------|-----------|---------|------|-------|
| Train | 2.029 | 2.204 | 2.165 | 2.150 | **8.548** |
| Val | 336 | 336 | 455 | 427 | **1.554** |
| Test | 361 | 186 | 469 | 448 | **1.464** |
| **Total** | **2.726** | **2.726** | **3.089** | **3.025** | **11.566** |

## Arsitektur

### CNN Custom (from scratch)

```
Input (64x64x3) → Conv(32,3×3) → ReLU → MaxPool(2×2) → Conv(64,3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(128,ReLU) → Dropout(0.5) → Dense(4,Softmax)
```

**Parameter:** ~2M trainable

### MobileNetV2 (Transfer Learning)

```
Input (64x64x3) → MobileNetV2 (frozen) → GAP → Dense(128,ReLU) → Dropout(0.5) → Dense(4,Softmax)
```

**Parameter:** ~2.3M frozen + ~130K trainable

## Hasil

| Model | Arsitektur | Val Acc | Test Acc | F1 (weighted) |
|-------|-----------|:-------:|:--------:|:--------------:|
| CNN Custom | Conv2D → MaxPool ×2 → Flatten → Dense | 21.62% | 33.74% | 0.2094 |
| MobileNetV2 (Baseline) | Transfer Learning + GAP + Dense(128) | 96.01% | **93.85%** | **0.9385** |
| MobileNetV2 (Tuned) | Transfer Learning + Dense(256, lr=0.0005) | **96.78%** | 92.76% | 0.9275 |

> **Mengapa CNN Custom gagal?** Dataset hanya ~11.566 gambar — terlalu kecil untuk
> training from scratch. CNN Custom overfit cepat (val_acc stuck di 21.6%, setara random).
> MobileNetV2 dengan pretrained ImageNet weights langsung mencapai 93.85% —
> membuktikan bahwa **transfer learning wajib** untuk dataset terbatas.

**Hyperparameter Terbaik (RandomSearch):**
- Learning rate: 0.0005
- Dense units: 256
- Dropout: 0.5
- Optimizer: adam
- Batch size: 32

## Struktur Folder

```
TA MESIN/
├── app/                    # Streamlit dashboard (5 halaman)
│   ├── app.py              # Main app + navigasi
│   ├── utils.py            # Shared utilities + Grad-CAM
│   ├── pages/              # Halaman Streamlit
│   │   ├── 1_EDA.py
│   │   ├── 2_Model_Demo.py
│   │   ├── 3_Evaluasi_Model.py
│   │   ├── 4_Dokumentasi.py
│   │   └── 5_Interpretasi_Hasil.py
│   └── assets/             # Dataset stats JSON
├── models/                 # Trained models (.h5)
├── tahap_1_EDA/            # Exploratory Data Analysis
├── tahap_2_Preprocessing/  # Preprocessing pipeline
├── tahap_3_Augmentasi/     # Data augmentation
├── tahap_4_Modeling/       # Training + Hyperparameter tuning
├── tahap_5_Evaluasi/       # Evaluation + Grad-CAM
├── tahap_6_Deployment/     # Real-time deployment script
├── PROBLEM_STATEMENT.md    # Problem statement (Soal 1)
├── PREPROCESSING_DOC.md    # Dokumentasi preprocessing (Soal 2)
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

## Cara Menjalankan

### 1. Clone & Siapkan Data

```bash
git clone https://github.com/Ranggapesty/drowsiness-detection.git
cd drowsiness-detection
```

Download dataset dari [Kaggle](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset), extract ke folder `Data/`:
```
drowsiness-detection/Data/train/
├── Closed_Eyes/
├── Open_Eyes/
├── No_yawn/
└── Yawn/
```

### 2. Setup Environment

```bash
conda create -n drowsiness python=3.9
conda activate drowsiness
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app/app.py
```

### 4. Real-time Camera (Local Only)

Mode **Real-time (WebRTC)** membutuhkan dependencies tambahan yang tidak tersedia di Streamlit Cloud:

```bash
conda activate drowsiness
pip install av==14.0.0
pip install streamlit-webrtc --no-deps
pip install aioice aiortc opencv-python==4.8.1.78
```

> **Catatan:** SSL error (`ssl.SSLError`) bisa diatasi dengan:
> ```bash
> copy sitecustomize.py D:\conda_envs\drowsiness\lib\site-packages\
> ```

Setelah itu jalankan `streamlit run app/app.py`, buka `http://localhost:8501`, pilih tab **Model Demo** → **Real-time (WebRTC)**, dan izinkan akses kamera.

### 5. (Opsional) Training Ulang

```bash
python tahap_4_Modeling/04_Modeling_Training.py
```

## Deployment

Aplikasi dapat diakses secara live atau dijalankan secara lokal.

| Komponen | Link |
|----------|------|
| Live Demo | [drowsiness-detection.streamlit.app](https://drowsiness-detection-gzskeknjxmnwfaor6sl4fd.streamlit.app/) |
| Model (HuggingFace) | [Ranggapesty/drowsiness-models](https://huggingface.co/Ranggapesty/drowsiness-models) |
| EDA Notebook | [01_eda.ipynb](tahap_1_EDA/01_eda.ipynb) |
| Modeling Notebook | [02_modeling.ipynb](tahap_4_Modeling/02_modeling.ipynb) |

