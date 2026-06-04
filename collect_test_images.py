"""
Independent Test Image Collector for Plant Disease Detector
===========================================================
Instead of downloading from the web, this script picks
2 random images from EACH of your 15 PlantVillage classes
(30 images total) that were NOT used in training or validation.

This works because your notebook only uses 40% of the dataset
(validation_split=0.6 for training, 0.2 for validation),
leaving ~40% of images completely untouched — those are
truly independent test images the model has never seen.

Run this AFTER your training cell has run (so the dataset
is already downloaded at /content/PlantVillage).

Images saved to: /content/test_images/
"""

import os
import json
import random
import shutil
import tensorflow as tf

PLANT_DIR = "/content/PlantVillage"   # where your dataset lives
SAVE_DIR  = "/content/test_images"
IMAGES_PER_CLASS = 2                  # 2 per class = 30 total images
SEED = 999                            # different seed from training (which used 123)

os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED)

if not os.path.isdir(PLANT_DIR):
    print(f"ERROR: Dataset not found at {PLANT_DIR}")
    print("Make sure your training cells have run first so the dataset is downloaded.")
    raise SystemExit(1)

classes = sorted(os.listdir(PLANT_DIR))
classes = [c for c in classes if os.path.isdir(os.path.join(PLANT_DIR, c))]
print(f"Found {len(classes)} classes in {PLANT_DIR}\n")

# Build the list of image files that Keras used for training and validation
# so we can EXCLUDE them from our test set.
# Keras with seed=123 is deterministic — we replicate the split to find
# which files it used, then pick from what remains.
print("Loading dataset splits to find untouched images...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    PLANT_DIR,
    validation_split=0.6,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=1,
    shuffle=False
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    PLANT_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=1,
    shuffle=False
)

used_files = set(train_ds.file_paths + val_ds.file_paths)
print(f"  Training images  : {len(train_ds.file_paths)}")
print(f"  Validation images: {len(val_ds.file_paths)}")
print(f"  Total used       : {len(used_files)}")

# Collect all image files per class, excluding used ones
results   = []
copied    = 0
skipped   = 0

print(f"\n{'='*60}")
print(f"  Collecting {IMAGES_PER_CLASS} unseen test images per class")
print(f"{'='*60}\n")

for cls in classes:
    cls_dir = os.path.join(PLANT_DIR, cls)
    all_files = [
        os.path.join(cls_dir, f)
        for f in os.listdir(cls_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    # Only files the model has NEVER seen
    unseen = [f for f in all_files if f not in used_files]

    if len(unseen) < IMAGES_PER_CLASS:
        print(f"  ⚠  {cls}: only {len(unseen)} unseen images (need {IMAGES_PER_CLASS}), using all")
        chosen = unseen
    else:
        chosen = random.sample(unseen, IMAGES_PER_CLASS)

    for i, src in enumerate(chosen, 1):
        ext      = os.path.splitext(src)[1].lower()
        safe_cls = cls.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        filename = f"{safe_cls}_{i}{ext}"
        dest     = os.path.join(SAVE_DIR, filename)
        shutil.copy2(src, dest)
        size_kb  = os.path.getsize(dest) / 1024
        print(f"  ✅  {filename}  ({size_kb:.1f} KB)")
        results.append({
            "filename": filename,
            "label":    cls,
            "source":   "PlantVillage (unseen)",
            "path":     dest,
            "status":   "ok"
        })
        copied += 1

manifest_path = os.path.join(SAVE_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"  Done!  ✅ {copied} images collected across {len(classes)} classes")
print(f"  Folder  : {SAVE_DIR}")
print(f"  Manifest: {manifest_path}")
print(f"{'='*60}")
print(f"\n  ➡  Next: run batch_test.py to score all images.")
