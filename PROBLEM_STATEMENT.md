# Problem Statement: Deteksi Kantuk Pengemudi (Drowsiness Detection)

## 1. Latar Belakang

Kecelakaan lalu lintas merupakan salah satu penyebab kematian tertinggi di dunia. Menurut data World Health Organization (WHO), sekitar 1.3 juta orang meninggal setiap tahun akibat kecelakaan di jalan raya. Salah satu faktor utama penyebab kecelakaan adalah **kelelahan dan kantuk saat mengemudi**. Studi menunjukkan bahwa 20-30% kecelakaan lalu lintas berkaitan dengan pengemudi yang mengantuk (National Sleep Foundation).

Kantuk menyebabkan penurunan konsentrasi, waktu reaksi yang lebih lambat, dan pengambilan keputusan yang buruk — kombinasi fatal saat mengemudi. Sayangnya, banyak pengemudi tidak menyadari bahwa mereka mulai mengantuk hingga terlambat untuk bereaksi.

Deteksi dini terhadap tanda-tanda kantuk dapat menyelamatkan nyawa. Dengan memanfaatkan computer vision dan deep learning, kita dapat mengidentifikasi indikator visual kantuk seperti **mata tertutup** (Closed_Eyes) dan **menguap** (Yawn) secara real-time, kemudian memberikan peringatan kepada pengemudi.

## 2. Tujuan Bisnis/Analisis

**Tujuan utama:** Membangun sistem deteksi kantuk berbasis citra yang mampu mengklasifikasikan kondisi pengemudi ke dalam 4 kategori:

| Kelas | Label | Deskripsi |
|-------|-------|-----------|
| 0 | Closed_Eyes | Mata tertutup — indikasi kantuk tinggi |
| 1 | Open_Eyes | Mata terbuka — kondisi waspada |
| 2 | No_yawn | Mulut tidak menguap — kondisi normal |
| 3 | Yawn | Mulut menguap — indikasi kelelahan |

**Tujuan analitis:**
- Mengeksplorasi dan memahami karakteristik dataset citra mata dan mulut pengemudi
- Membangun model klasifikasi dengan akurasi tinggi menggunakan transfer learning (MobileNetV2)
- Membandingkan performa arsitektur CNN custom dengan MobileNetV2
- Mengoptimalkan hyperparameter untuk mencapai performa terbaik
- Menginterpretasikan keputusan model menggunakan Grad-CAM

**Tujuan bisnis:**
- Menyediakan sistem peringatan dini kantuk yang dapat diintegrasikan ke dalam kendaraan
- Mengurangi risiko kecelakaan akibat mengantuk
- Membantu perusahaan transportasi/logistik memonitor tingkat kewaspadaan pengemudi

## 3. Metrik Kesuksesan Proyek

| Metrik | Target | Alasan |
|--------|--------|--------|
| **Accuracy** | ≥ 90% | Akurasi keseluruhan dalam mengklasifikasikan 4 kelas |
| **F1-Score (weighted)** | ≥ 0.90 | Keseimbangan precision dan recall di semua kelas |
| **ROC-AUC** | ≥ 0.95 | Kemampuan model membedakan antar kelas |
| **False Negative (Kantuk terlewat)** | < 5% | Prioritas utama: jangan sampai pengemudi kantuk tidak terdeteksi |
| **Waktu inferensi** | < 100ms per frame | Mendukung real-time deployment |

## 4. Sumber Dataset

**Drowsiness Dataset** — Kaggle  
- **Link:** https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset  
- **Total gambar:** 11.566  
- **Format:** Campuran JPG (7.566) dan PNG (4.000)  
- **Split:** Train (8.548), Val (1.554), Test (1.464)  
- **Kelas:** Closed_Eyes, Open_Eyes, No_yawn, Yawn  
- **Metadata PNG:** Mencakup informasi subject, frame, penggunaan kacamata, blur, dan pencahayaan

## 5. Statistik Deskriptif Awal

| Aspek | Nilai |
|-------|-------|
| Total gambar | 11.566 |
| Dimensi (width) | 28 – 146 px (mean: 85.0) |
| Dimensi (height) | 21 – 138 px (mean: 72.4) |
| Channel | 3 (RGB) |
| Format | 65.4% JPG, 34.6% PNG |
| Train split | 8.548 (73.9%) |
| Val split | 1.554 (13.4%) |
| Test split | 1.464 (12.7%) |
| Distribusi kelas (Train) | Closed_Eyes: 2.029, Open_Eyes: 2.204, No_yawn: 2.165, Yawn: 2.150 |

Distribusi data relatif seimbang antar kelas (no imbalance issue). Namun, gambar memiliki resolusi dan aspek rasio yang sangat bervariasi karena merupakan crop dari bagian wajah yang berbeda (mata vs mulut), sehingga preprocessing yang tepat menjadi krusial.
