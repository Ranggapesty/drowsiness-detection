"""
=========================================================
Page 4: Dokumentasi Proyek
=========================================================
"""

import streamlit as st

def main():
    st.title("📚 Dokumentasi Proyek")
    st.markdown("Deteksi Kantuk Pengemudi dengan Deep Learning")

    st.header("1. Latar Belakang")
    st.markdown("""
    Kecelakaan lalu lintas akibat **mengantuk saat mengemudi** merupakan masalah serius.
    Penelitian menunjukkan bahwa **20-30% kecelakaan** disebabkan oleh kelelahan pengemudi.
    Sistem deteksi kantuk berbasis computer vision dapat membantu mengurangi risiko ini.
    """)

    st.header("2. Dataset")
    st.markdown("""
    **Drowsiness Dataset (Kaggle)** — 11.566 gambar, 4 kelas:
    | Kelas | Deskripsi | Contoh |
    |-------|-----------|--------|
    | **Closed_Eyes** | Mata tertutup | Crop area mata |
    | **Open_Eyes** | Mata terbuka | Crop area mata |
    | **No_yawn** | Mulut tidak menguap | Crop area mulut |
    | **Yawn** | Mulut menguap | Crop area mulut |

    **Preprocessing:**
    - Letterbox padding ke **64×64** (preserve aspect ratio)
    - Normalisasi [0, 255] → [0.0, 1.0]
    - Split: Train 74%, Val 13%, Test 13%
    """)

    st.header("3. Arsitektur Model")
    st.markdown("""
    **MobileNetV2** — Transfer Learning:
    - Base model: MobileNetV2 (pre-trained ImageNet), weights **frozen**
    - `GlobalAveragePooling2D` mengurangi dimensi spasial
    - `Dense(128, ReLU)` → `Dropout(0.5)` → `Dense(4, Softmax)`

    **Mengapa MobileNetV2?**
    - ✅ Ringan (3.5M parameters) — cocok untuk real-time
    - ✅ Depthwise Separable Convolution — efisien secara komputasi
    - ✅ Pre-trained ImageNet — feature extraction yang sudah matang
    """)

    st.header("4. Hyperparameter Tuning")
    st.markdown("""
    **Method:** RandomSearch (Keras Tuner), 12/20 trials

    | Parameter | Range | Best |
    |-----------|-------|------|
    | Learning Rate | 1e-4, 5e-4, 1e-3 | **5e-4** |
    | Dense Units | 64, 128, 256 | **256** |
    | Dropout Rate | 0.3, 0.5, 0.7 | **0.5** |
    | Optimizer | adam, rmsprop | **adam** |
    | Batch Size | 16, 32 | **32** |

    Best trial: **val_acc = 0.9678**
    """)

    st.header("5. Hasil Evaluasi")
    st.markdown("""
    **Test Set (1.464 gambar):**
    | Metrik | Nilai |
    |--------|-------|
    | Accuracy | **93.85%** |
    | F1-Score (weighted) | **0.9385** |
    | AUC (per class) | **> 0.99** |

    **Confusion Matrix Highlights:**
    - Closed_Eyes → Open_Eyes: misklasifikasi paling umum
    - No_yawn → Yawn: beberapa false negative
    - Model sangat baik membedakan mata vs mulut
    """)

    st.header("6. Teknologi")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Machine Learning**")
        st.markdown("- TensorFlow 2.10")
        st.markdown("- Keras Tuner")
        st.markdown("- Scikit-learn")
    with col2:
        st.markdown("**Computer Vision**")
        st.markdown("- OpenCV")
        st.markdown("- PIL / Pillow")
        st.markdown("- Matplotlib")
    with col3:
        st.markdown("**Deployment**")
        st.markdown("- Streamlit")
        st.markdown("- Streamlit Cloud")
        st.markdown("- GPU CUDA 11.2")

    st.header("7. Referensi")
    st.markdown("""
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
    - [Drowsiness Dataset (Kaggle)](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)
    - [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
    - [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
    """)

if __name__ == "__main__":
    main()
