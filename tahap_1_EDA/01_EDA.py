"""
Fase 1: Eksplorasi Data (EDA) - Deteksi Kantuk Pengemudi
=========================================================
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sys
warnings.filterwarnings("ignore")

print("Memulai EDA...", flush=True)
DATA_DIR = r"D:\TA Mesin\Data"
SPLITS = ["Train", "Val", "Test"]
CLASSES = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]

print("=" * 60)
print("FASE 1: EKSPLORASI DATA (EDA)")
print("=" * 60)

# =========================================================
# 1. DISTRIBUSI JUMLAH GAMBAR PER KELAS & SPLIT
# =========================================================
print("\n" + "=" * 60)
print("1. DISTRIBUSI JUMLAH GAMBAR PER KELAS & SPLIT")
print("=" * 60)

data_summary = []

for split in SPLITS:
    split_path = os.path.join(DATA_DIR, split)
    for cls in CLASSES:
        cls_path = os.path.join(split_path, cls)
        if os.path.exists(cls_path):
            files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            data_summary.append({
                "Split": split,
                "Class": cls,
                "Count": len(files),
                "JPG": len([f for f in files if f.lower().endswith((".jpg", ".jpeg"))]),
                "PNG": len([f for f in files if f.lower().endswith(".png")])
            })

df_summary = pd.DataFrame(data_summary)
print("\nTabel Distribusi:")
print(df_summary.to_string(index=False))

total = df_summary["Count"].sum()
train_total = df_summary[df_summary["Split"] == "Train"]["Count"].sum()
val_total = df_summary[df_summary["Split"] == "Val"]["Count"].sum()
test_total = df_summary[df_summary["Split"] == "Test"]["Count"].sum()

print(f"\nTotal seluruh gambar: {total}")
print(f"Train: {train_total:,} ({train_total/total*100:.1f}%)")
print(f"Val:   {val_total:,} ({val_total/total*100:.1f}%)")
print(f"Test:  {test_total:,} ({test_total/total*100:.1f}%)")

# =========================================================
# 2. CEK FORMAT, RESOLUSI & CHANNEL GAMBAR
# =========================================================
print("\n" + "=" * 60)
print("2. CEK FORMAT, RESOLUSI & CHANNEL GAMBAR")
print("=" * 60)

sample_sizes = {}
for split in SPLITS:
    for cls in CLASSES:
        cls_path = os.path.join(DATA_DIR, split, cls)
        if not os.path.exists(cls_path):
            continue
        files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            continue
        sample_files = files[:10]
        for fname in sample_files:
            fpath = os.path.join(cls_path, fname)
            try:
                img = cv2.imread(fpath)
                if img is not None:
                    h, w = img.shape[:2]
                    channels = img.shape[2] if len(img.shape) == 3 else 1
                    ext = os.path.splitext(fname)[1].lower()
                    key = f"{split}/{cls}"
                    if key not in sample_sizes:
                        sample_sizes[key] = []
                    sample_sizes[key].append({
                        "file": fname,
                        "width": w,
                        "height": h,
                        "channels": channels,
                        "format": ext
                    })
            except:
                pass

print("\nSample Resolusi (10 gambar per kelas):")
for key, samples in sorted(sample_sizes.items()):
    widths = [s["width"] for s in samples]
    heights = [s["height"] for s in samples]
    fmts = set(s["format"] for s in samples)
    chs = set(s["channels"] for s in samples)
    print(f"  {key}:")
    print(f"    Width: {min(widths)} - {max(widths)}")
    print(f"    Height: {min(heights)} - {max(heights)}")
    print(f"    Format: {fmts}")
    print(f"    Channels: {chs}")

# =========================================================
# 3. CEK GAMBAR CORRUPT
# =========================================================
print("\n" + "=" * 60)
print("3. CEK GAMBAR CORRUPT")
print("=" * 60)

corrupt_count = 0
total_checked = 0
corrupt_files = []

for split in SPLITS:
    for cls in CLASSES:
        cls_path = os.path.join(DATA_DIR, split, cls)
        if not os.path.exists(cls_path):
            continue
        files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for i, fname in enumerate(files):
            total_checked += 1
            if total_checked % 2000 == 0:
                print(f"  Mengecek gambar ke-{total_checked}...", flush=True)
            fpath = os.path.join(cls_path, fname)
            try:
                img = cv2.imread(fpath)
                if img is None:
                    corrupt_count += 1
                    corrupt_files.append(f"{split}/{cls}/{fname}")
            except:
                corrupt_count += 1
                corrupt_files.append(f"{split}/{cls}/{fname}")

print(f"Total gambar dicek: {total_checked}")
print(f"Gambar corrupt: {corrupt_count}")
if corrupt_count == 0:
    print("Status: Semua bersih")
else:
    print("File corrupt:")
    for f in corrupt_files:
        print(f"  - {f}")

# =========================================================
# 4. VISUALISASI SAMPLE GAMBAR
# =========================================================
print("\n" + "=" * 60)
print("4. VISUALISASI SAMPLE GAMBAR (disimpan ke file)")
print("=" * 60)

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle("Sample Gambar per Kelas (Train Set)", fontsize=16)

for i, cls in enumerate(CLASSES):
    cls_path = os.path.join(DATA_DIR, "Train", cls)
    if not os.path.exists(cls_path):
        continue
    files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sample_files = np.random.choice(files, min(5, len(files)), replace=False)

    for j, fname in enumerate(sample_files):
        fpath = os.path.join(cls_path, fname)
        img = plt.imread(fpath)
        axes[i, j].imshow(img)
        axes[i, j].set_title(f"{cls}", fontsize=9)
        axes[i, j].axis("off")

plt.tight_layout()
output_img = r"D:\TA Mesin\notebooks\eda_sample_images.png"
plt.savefig(output_img, dpi=150)
plt.close()
print(f"Gambar sample disimpan di: {output_img}")

# =========================================================
# 5. ANALISIS METADATA FILE .PNG
# =========================================================
print("\n" + "=" * 60)
print("5. ANALISIS METADATA FILE .PNG")
print("=" * 60)

metadata_rows = []
for split in SPLITS:
    for cls in CLASSES:
        cls_path = os.path.join(DATA_DIR, split, cls)
        if not os.path.exists(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith(".png"):
                continue
            parts = fname.replace(".png", "").split("_")
            if len(parts) == 8:
                metadata_rows.append({
                    "Split": split,
                    "Class": cls,
                    "Subject": parts[0],
                    "Frame": parts[1],
                    "Eye": parts[2],
                    "Face": parts[3],
                    "Open": parts[4],
                    "Glasses": parts[5],
                    "Blur": parts[6],
                    "Illumination": parts[7]
                })

df_meta = pd.DataFrame(metadata_rows)
print(f"Total file .png dengan metadata: {len(df_meta)}")

if len(df_meta) > 0:
    print("\nDistribusi per Subject:")
    print(df_meta["Subject"].value_counts().to_string())

    glasses_map = {"0": "Tanpa Kacamata", "1": "Kacamata Biasa", "2": "Kacamata Hitam"}
    df_meta["Glasses_Label"] = df_meta["Glasses"].map(glasses_map)
    print("\nDistribusi Kacamata:")
    print(df_meta["Glasses_Label"].value_counts().to_string())

    blur_map = {"0": "Jelas", "1": "Buram"}
    df_meta["Blur_Label"] = df_meta["Blur"].map(blur_map)
    print("\nDistribusi Blur:")
    print(df_meta["Blur_Label"].value_counts().to_string())

    print("\nDistribusi Pencahayaan:")
    print(df_meta["Illumination"].value_counts().to_string())

# =========================================================
# 6. PERBANDINGAN JPG vs PNG
# =========================================================
print("\n" + "=" * 60)
print("6. PERBANDINGAN FORMAT FILE (JPG vs PNG)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(df_summary))
bar_width = 0.35
bars1 = ax.bar([p - bar_width/2 for p in x], df_summary["JPG"], bar_width, label="JPG", color="#FF9800")
bars2 = ax.bar([p + bar_width/2 for p in x], df_summary["PNG"], bar_width, label="PNG", color="#4CAF50")

ax.set_xlabel("Split / Class")
ax.set_ylabel("Jumlah")
ax.set_title("Perbandingan JPG vs PNG per Split & Kelas")
ax.set_xticks(x)
ax.set_xticklabels([f"{r.Split}\n{r.Class}" for _, r in df_summary.iterrows()], rotation=45, ha="right")
ax.legend()

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h, int(h), ha="center", va="bottom", fontsize=8)
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h, int(h), ha="center", va="bottom", fontsize=8)

plt.tight_layout()
output_chart = r"D:\TA Mesin\notebooks\eda_format_comparison.png"
plt.savefig(output_chart, dpi=150)
plt.close()
print(f"Grafik perbandingan format disimpan di: {output_chart}")

# =========================================================
# KESIMPULAN
# =========================================================
print("\n" + "=" * 60)
print("KESIMPULAN EDA")
print("=" * 60)
print("""
| Aspek                  | Temuan                                      | Tindakan                         |
|------------------------|---------------------------------------------|----------------------------------|
| Distribusi data        | Train +-74%, Val +-13%, Test +-13%          | Bisa pakai split yang ada        |
|                        | Cukup seimbang antar kelas                  | Tidak perlu re-balancing         |
| Resolusi gambar        | (lihat output di atas)                      | Wajib resize seragam             |
| Format file            | Campuran .jpg + .png                        | Konversi semua ke RGB            |
| Gambar corrupt         | (lihat output di atas)                      | Hapus jika ada                   |
| Metadata .png          | Ada info subject, glasses, blur, illum      | Berguna untuk analisis bias      |
| Keseimbangan kelas     | Relatif seimbang                            | Tidak perlu re-balancing         |
""")
print("=" * 60)
print("Fase 1 (EDA) SELESAI")
print("=" * 60)
