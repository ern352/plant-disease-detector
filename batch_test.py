"""
Batch Test Script for Plant Disease Detector
=============================================
Run this AFTER collect_test_images.py.
Loads your trained model, runs all images in /content/test_images/,
and prints a full accuracy report.

Usage:
    1. Make sure your model is saved as:  /content/plant_disease_model.h5
       (adjust MODEL_PATH below if different)
    2. Run this script in Colab after collect_test_images.py
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# ── CONFIG — adjust if your paths differ ──────────────────────────────────────
MODEL_PATH  = "/content/plant_disease_model.h5"   # path to your saved model
TEST_DIR    = "/content/test_images"               # folder from collect script
IMG_SIZE    = (128, 128)                           # must match training size
# ──────────────────────────────────────────────────────────────────────────────

# Load model
print("Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"  Model loaded  — input shape: {model.input_shape}\n")

# Load class names (same order as training)
# This grabs them from the training data directory structure if still present,
# otherwise falls back to the manifest labels.
TRAIN_DATA_DIR = "/content/plantvillage/PlantVillage"  # adjust if needed
if os.path.isdir(TRAIN_DATA_DIR):
    class_names = sorted(os.listdir(TRAIN_DATA_DIR))
    print(f"  Found {len(class_names)} classes from training directory.\n")
else:
    print("  ⚠️  Training directory not found — using manifest labels only.")
    print("     Predicted class INDEX will be shown instead of name.\n")
    class_names = None

# Load manifest
manifest_path = os.path.join(TEST_DIR, "manifest.json")
with open(manifest_path) as f:
    manifest = json.load(f)

# Filter to successfully downloaded images
images_to_test = [m for m in manifest if m["status"] == "ok"]
print(f"{'='*65}")
print(f"  Testing {len(images_to_test)} images")
print(f"{'='*65}")
print(f"  {'File':<35} {'Expected label':<30} {'Result'}")
print(f"  {'-'*35} {'-'*30} {'-'*20}")

correct   = 0
incorrect = 0
results   = []

for item in images_to_test:
    img_path = item["path"]
    true_label = item["label"]

    # Preprocess
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)   # (1, H, W, 3)

    # Predict
    preds      = model.predict(arr, verbose=0)
    pred_idx   = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    if class_names:
        pred_label = class_names[pred_idx]
    else:
        pred_label = f"class_{pred_idx}"

    # Normalise comparison (strip spaces, lower-case)
    match = pred_label.strip().lower() == true_label.strip().lower()
    status = "✅ CORRECT" if match else "❌ WRONG  "
    if match:
        correct += 1
    else:
        incorrect += 1

    fname = item["filename"]
    print(f"  {fname:<35} {true_label:<30} {status}  ({confidence:.1f}%)")
    if not match:
        print(f"    └─ predicted: {pred_label}")

    results.append({
        "file":          fname,
        "true_label":    true_label,
        "pred_label":    pred_label,
        "confidence":    round(confidence, 2),
        "correct":       match,
    })

# ── Summary ────────────────────────────────────────────────────────────────────
total    = correct + incorrect
accuracy = (correct / total * 100) if total else 0

print(f"\n{'='*65}")
print(f"  RESULTS SUMMARY")
print(f"  Total images tested : {total}")
print(f"  Correct             : {correct}")
print(f"  Incorrect           : {incorrect}")
print(f"  Test Accuracy       : {accuracy:.1f}%")
print(f"{'='*65}")

# Save report
report = {
    "total":    total,
    "correct":  correct,
    "incorrect": incorrect,
    "accuracy_pct": round(accuracy, 2),
    "per_image": results,
}
report_path = os.path.join(TEST_DIR, "test_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Full report saved to: {report_path}")
print(f"  ➡  Screenshot this output for your Checkpoint 2 evidence!")
