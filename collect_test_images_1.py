"""
Independent Test Image Collector for Plant Disease Detector
===========================================================
Run this in Google Colab. Downloads 15 plant leaf images
from Wikimedia Commons using full-resolution direct URLs
(no /thumb/ path — fixes the HTTP 400 error).

Images saved to: /content/test_images/
"""

import os
import urllib.request
import json

SAVE_DIR = "/content/test_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# All URLs use the direct full-resolution Wikimedia path (no /thumb/, no size prefix).
# This is what fixes the HTTP 400 "Use thumbnail sizes" error.
TEST_IMAGES = [
    {
        "filename": "01_tomato_healthy.jpg",
        "label":    "Tomato___healthy",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg"
    },
    {
        "filename": "02_tomato_early_blight.jpg",
        "label":    "Tomato___Early_blight",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/6/6e/Tomato_-_early_blight.jpg"
    },
    {
        "filename": "03_tomato_late_blight.jpg",
        "label":    "Tomato___Late_blight",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/3/3a/Potato_late_blight.jpg"
    },
    {
        "filename": "04_potato_healthy.jpg",
        "label":    "Potato___healthy",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/a/ab/Potato_and_cross_section.jpg"
    },
    {
        "filename": "05_potato_early_blight.jpg",
        "label":    "Potato___Early_blight",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/e/e1/Alternaria_solani_on_potato_foliage.jpg"
    },
    {
        "filename": "06_pepper_healthy.jpg",
        "label":    "Pepper,_bell___healthy",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/b/b1/Sweet-Pepper.jpg"
    },
    {
        "filename": "07_pepper_bacterial_spot.jpg",
        "label":    "Pepper,_bell___Bacterial_spot",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/5/5b/Xanthomonas_euvesicatoria_on_pepper.jpg"
    },
    {
        "filename": "08_corn_healthy.jpg",
        "label":    "Corn_(maize)___healthy",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/e/ed/Corn_maize_plant.jpg"
    },
    {
        "filename": "09_corn_common_rust.jpg",
        "label":    "Corn_(maize)___Common_rust_",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/e/e9/Puccinia_sorghi_-_common_corn_rust.jpg"
    },
    {
        "filename": "10_grape_healthy.jpg",
        "label":    "Grape___healthy",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/b/bb/Table_grapes_on_the_vine.jpg"
    },
    {
        "filename": "11_grape_black_rot.jpg",
        "label":    "Grape___Black_rot",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/9/9b/Grape_black_rot.jpg"
    },
    {
        "filename": "12_apple_healthy.jpg",
        "label":    "Apple___healthy",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg"
    },
    {
        "filename": "13_apple_scab.jpg",
        "label":    "Apple___Apple_scab",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/c/c9/Apple_scab_lesions.jpg"
    },
    {
        "filename": "14_strawberry_healthy.jpg",
        "label":    "Strawberry___healthy",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/2/29/PerfectStrawberry.jpg"
    },
    {
        "filename": "15_strawberry_leaf_scorch.jpg",
        "label":    "Strawberry___Leaf_scorch",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/9/9f/Strawberry_leaf_scorch.jpg"
    },
]

# Wikimedia requires a descriptive User-Agent for direct (non-thumb) access
headers = {
    "User-Agent": "PlantDiseaseCapstoneProject/1.0 (Google Colab; educational use)"
}

results   = []
downloaded = 0
failed     = 0

print(f"{'='*60}")
print(f"  Downloading {len(TEST_IMAGES)} independent test images")
print(f"{'='*60}\n")

for img in TEST_IMAGES:
    dest = os.path.join(SAVE_DIR, img["filename"])
    try:
        req = urllib.request.Request(img["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        size_kb = os.path.getsize(dest) / 1024
        print(f"  ✅  {img['filename']}  ({size_kb:.1f} KB)  →  {img['label']}")
        results.append({**img, "status": "ok", "path": dest})
        downloaded += 1
    except Exception as e:
        print(f"  ❌  {img['filename']}  →  FAILED: {e}")
        results.append({**img, "status": "failed", "error": str(e)})
        failed += 1

manifest_path = os.path.join(SAVE_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"  Done!  ✅ {downloaded} downloaded   ❌ {failed} failed")
print(f"  Folder  : {SAVE_DIR}")
print(f"  Manifest: {manifest_path}")
print(f"{'='*60}")
print(f"\n  ➡  Next: run batch_test.py to score all images.")
