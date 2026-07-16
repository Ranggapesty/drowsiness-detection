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

### MobileNetV2 (Transfer Learning)

```
Input (64x64x3) → MobileNetV2 (frozen) → GAP → Dense(128,ReLU) → Dropout(0.5) → Dense(4,Softmax)
```

**Parameter:** ~2.3M frozen + ~130K trainable

## Hasil

| Model | Best Val Acc | Test Acc | F1 (weighted) |
|-------|-------------|----------|---------------|
| CNN Custom | ~27% | — | — |
| MobileNetV2 (Baseline) | **96.01%** | **93.85%** | **0.9385** |
| MobileNetV2 (Tuned) | **96.78%** | — | — |

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

### 2. Run Streamlit App

```bash
streamlit run app/app.py
```

### 3. (Opsional) Training Ulang

```bash
python tahap_4_Modeling/04_Modeling_Training.py
```

## Deployment

Aplikasi Streamlit di-deploy di Streamlit Community Cloud:

[**Live Demo**](https://drowsiness-detection.streamlit.app) (link placeholder)

## Deployment

Aplikasi dapat diakses di **[Streamlit Cloud (coming soon)]()** atau dijalankan secara lokal.

| Komponen | Link |
|----------|------|
| Model (HuggingFace) | [Ranggapesty/drowsiness-models](https://huggingface.co/Ranggapesty/drowsiness-models) |
| Demo Live (Streamlit) | *(add URL after deploy)* |
| EDA Notebook | [01_eda.ipynb](tahap_1_EDA/01_eda.ipynb) |
| Modeling Notebook | [02_modeling.ipynb](tahap_4_Modeling/02_modeling.ipynb) |

### Catatan Deployment

- Model di-host via Hugging Face → didownload otomatis saat Streamlit app dijalankan
- Aplikasi menggunakan **PIL** (bukan OpenCV) untuk kompatibilitas Streamlit Cloud
- Tidak ada file `.npy` atau `.h5` di git — semuanya di-load dari HuggingFace atau local

## 📄 License

Academic project — Universitas (UAS Deep Learning)
