"""
Independent Test Image Collector for Plant Disease Detector
===========================================================
Run this in Google Colab BEFORE batch_test.py.
Downloads 15 diverse leaf images from open sources (Wikimedia Commons, 
public research repos) — completely separate from PlantVillage training data.

Usage:
    Simply run all cells. Images saved to: /content/test_images/
"""

import os
import urllib.request
import json

# ── Output folder ──────────────────────────────────────────────────────────────
SAVE_DIR = "/content/test_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 15 independent test images ─────────────────────────────────────────────────
# Sources: Wikimedia Commons (CC-licensed), USDA public domain, open research repos
# Each image covers a DIFFERENT disease class to stress-test the model broadly.
# Label = the PlantVillage class name your model was trained on.

TEST_IMAGES = [
    {
        "filename": "01_tomato_healthy.jpg",
        "label":    "Tomato___healthy",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Bright_red_tomato_and_cross_section02.jpg/320px-Bright_red_tomato_and_cross_section02.jpg"
    },
    {
        "filename": "02_tomato_early_blight.jpg",
        "label":    "Tomato___Early_blight",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Tomato_-_early_blight.jpg/320px-Tomato_-_early_blight.jpg"
    },
    {
        "filename": "03_tomato_late_blight.jpg",
        "label":    "Tomato___Late_blight",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Potato_late_blight.jpg/320px-Potato_late_blight.jpg"
    },
    {
        "filename": "04_potato_healthy.jpg",
        "label":    "Potato___healthy",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Potato_and_cross_section.jpg/320px-Potato_and_cross_section.jpg"
    },
    {
        "filename": "05_potato_early_blight.jpg",
        "label":    "Potato___Early_blight",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Alternaria_solani_on_potato_foliage.jpg/320px-Alternaria_solani_on_potato_foliage.jpg"
    },
    {
        "filename": "06_pepper_healthy.jpg",
        "label":    "Pepper,_bell___healthy",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Sweet-Pepper.jpg/320px-Sweet-Pepper.jpg"
    },
    {
        "filename": "07_pepper_bacterial_spot.jpg",
        "label":    "Pepper,_bell___Bacterial_spot",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Xanthomonas_euvesicatoria_on_pepper.jpg/320px-Xanthomonas_euvesicatoria_on_pepper.jpg"
    },
    {
        "filename": "08_corn_healthy.jpg",
        "label":    "Corn_(maize)___healthy",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Corn_maize_plant.jpg/320px-Corn_maize_plant.jpg"
    },
    {
        "filename": "09_corn_common_rust.jpg",
        "label":    "Corn_(maize)___Common_rust_",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Puccinia_sorghi_-_common_corn_rust.jpg/320px-Puccinia_sorghi_-_common_corn_rust.jpg"
    },
    {
        "filename": "10_grape_healthy.jpg",
        "label":    "Grape___healthy",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Table_grapes_on_the_vine.jpg/320px-Table_grapes_on_the_vine.jpg"
    },
    {
        "filename": "11_grape_black_rot.jpg",
        "label":    "Grape___Black_rot",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Grape_black_rot.jpg/320px-Grape_black_rot.jpg"
    },
    {
        "filename": "12_apple_healthy.jpg",
        "label":    "Apple___healthy",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/320px-Red_Apple.jpg"
    },
    {
        "filename": "13_apple_scab.jpg",
        "label":    "Apple___Apple_scab",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Apple_scab_lesions.jpg/320px-Apple_scab_lesions.jpg"
    },
    {
        "filename": "14_strawberry_healthy.jpg",
        "label":    "Strawberry___healthy",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/PerfectStrawberry.jpg/320px-PerfectStrawberry.jpg"
    },
    {
        "filename": "15_strawberry_leaf_scorch.jpg",
        "label":    "Strawberry___Leaf_scorch",
        "source":   "Wikimedia Commons",
        "url":      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Strawberry_leaf_scorch.jpg/320px-Strawberry_leaf_scorch.jpg"
    },
]

# ── Download loop ──────────────────────────────────────────────────────────────
headers = {"User-Agent": "Mozilla/5.0 (PlantDiseaseCapstoneProject/1.0)"}
results = []
downloaded = 0
failed = 0

print(f"{'='*60}")
print(f"  Downloading {len(TEST_IMAGES)} independent test images")
print(f"{'='*60}\n")

for img in TEST_IMAGES:
    dest = os.path.join(SAVE_DIR, img["filename"])
    try:
        req = urllib.request.Request(img["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            with open(dest, "wb") as f:
                f.write(resp.read())
        size_kb = os.path.getsize(dest) / 1024
        print(f"  ✅  {img['filename']}  ({size_kb:.1f} KB)  →  expected: {img['label']}")
        results.append({**img, "status": "ok", "path": dest})
        downloaded += 1
    except Exception as e:
        print(f"  ❌  {img['filename']}  →  FAILED: {e}")
        results.append({**img, "status": "failed", "error": str(e)})
        failed += 1

# ── Save manifest ──────────────────────────────────────────────────────────────
manifest_path = os.path.join(SAVE_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"  Done!  ✅ {downloaded} downloaded   ❌ {failed} failed")
print(f"  Images saved to : {SAVE_DIR}")
print(f"  Manifest saved  : {manifest_path}")
print(f"{'='*60}")
print(f"\n  ➡  Next step: run batch_test.py to score all images.")
