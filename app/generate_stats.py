"""
=========================================================
generate_stats.py – Dataset Statistics Generator
=========================================================
"""

import os
import json
import cv2
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"D:\TA Mesin\Data"
OUTPUT_DIR = r"D:\TA Mesin\app\assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = ["Train", "Val", "Test"]
CLASSES = ["Closed_Eyes", "Open_Eyes", "No_yawn", "Yawn"]

stats = {
    "total_images": 0,
    "splits": {},
    "format_distribution": {},
    "png_metadata": {"subjects": {}, "glasses": {}, "blur": {}, "illumination": {}},
    "resolution": {"min_width": 9999, "min_height": 9999, "max_width": 0, "max_height": 0, "mean_width": 0, "mean_height": 0},
    "preprocessing": {
        "target_size": 64,
        "normalization": "[0, 255] -> [0.0, 1.0]",
        "color_mode": "RGB",
        "resize_method": "letterbox padding (long edge = 64)",
        "classes": CLASSES
    }
}

total_jpg = 0
total_png = 0
widths, heights = [], []

for split in SPLITS:
    split_path = os.path.join(DATA_DIR, split)
    split_data = {}
    total_split = 0
    for cls in CLASSES:
        cls_path = os.path.join(split_path, cls)
        if not os.path.exists(cls_path):
            split_data[cls] = {"count": 0, "jpg": 0, "png": 0}
            continue
        files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        n_jpg = sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg")))
        n_png = sum(1 for f in files if f.lower().endswith(".png"))
        total_jpg += n_jpg
        total_png += n_png
        split_data[cls] = {"count": len(files), "jpg": n_jpg, "png": n_png}
        total_split += len(files)
        for fname in files[:20]:
            fpath = os.path.join(cls_path, fname)
            try:
                img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w); heights.append(h)
                    if w < stats["resolution"]["min_width"]: stats["resolution"]["min_width"] = w
                    if h < stats["resolution"]["min_height"]: stats["resolution"]["min_height"] = h
                    if w > stats["resolution"]["max_width"]: stats["resolution"]["max_width"] = w
                    if h > stats["resolution"]["max_height"]: stats["resolution"]["max_height"] = h
                    if fname.lower().endswith(".png") and len(img.shape) == 3 and img.shape[2] >= 3:
                        parts = fname.replace(".png", "").split("_")
                        if len(parts) == 8:
                            s, fr, ey, fa, op, gl, bl, il = parts
                            stats["png_metadata"]["subjects"][s] = stats["png_metadata"]["subjects"].get(s, 0) + 1
                            stats["png_metadata"]["glasses"][gl] = stats["png_metadata"]["glasses"].get(gl, 0) + 1
                            stats["png_metadata"]["blur"][bl] = stats["png_metadata"]["blur"].get(bl, 0) + 1
                            stats["png_metadata"]["illumination"][il] = stats["png_metadata"]["illumination"].get(il, 0) + 1
            except:
                pass
    stats["splits"][split] = {"total": total_split, "classes": split_data}
    stats["total_images"] += total_split

if widths:
    stats["resolution"]["mean_width"] = round(np.mean(widths), 1)
    stats["resolution"]["mean_height"] = round(np.mean(heights), 1)
stats["format_distribution"] = {"jpg": total_jpg, "png": total_png, "total": total_jpg + total_png}

with open(os.path.join(OUTPUT_DIR, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print(f"Dataset stats saved.")
print(f"Total images: {stats['total_images']}")
print(f"Format: {total_jpg} JPG, {total_png} PNG")
