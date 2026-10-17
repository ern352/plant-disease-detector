"""
Batch Test Script for Plant Disease Detector
=============================================
Re-scores the saved model on a folder of labelled test images, e.g. in a new
Colab session after you've trained once.

The test folder must contain one sub-folder per class, named exactly like the
training classes (the notebook's step 3 creates this as /content/test_samples):

    test_samples/
        Pepper__bell___Bacterial_spot/xxx.JPG
        Potato___healthy/yyy.JPG
        ...

Usage (Colab):
    1. Upload plant_disease_model.keras and class_names.json to /content
       (both are written by the notebook's step 7).
    2. Unzip test_samples.zip into /content/test_samples.
    3. !python batch_test.py
"""

import json
import os

import numpy as np
import tensorflow as tf

# ── CONFIG — adjust if your paths differ ──────────────────────────────────────
MODEL_PATH       = "/content/plant_disease_model.keras"
CLASS_NAMES_PATH = "/content/class_names.json"
TEST_DIR         = "/content/test_samples"
IMG_SIZE         = (128, 128)                      # must match training size
BATCH_SIZE       = 32
# ──────────────────────────────────────────────────────────────────────────────

print("Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)
print(f"  {len(class_names)} classes, input shape {model.input_shape}\n")

unknown = sorted(
    d for d in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, d)) and d not in class_names
)
if unknown:
    raise SystemExit(f"These test folders are not training classes: {unknown}\n"
                     f"Rename them to one of: {class_names}")

# The model rescales internally, so images go in as raw 0–255 pixels — same as training.
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=class_names,
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

test_loss, test_acc = model.evaluate(test_data, verbose=0)
probs  = model.predict(test_data, verbose=0)
y_pred = probs.argmax(axis=1)
y_true = np.concatenate([y.numpy() for _, y in test_data])

print(f"\n{'='*80}")
print(f"  {'File':<45} {'Result':<10} Predicted (confidence)")
print(f"  {'-'*45} {'-'*10} {'-'*22}")

results = []
for path, t, p, pr in zip(test_data.file_paths, y_true, y_pred, probs):
    rel = os.path.relpath(path, TEST_DIR)
    correct = bool(t == p)
    confidence = float(pr[p]) * 100
    print(f"  {rel[-45:]:<45} {'✅' if correct else '❌':<10} {class_names[p]} ({confidence:.1f}%)")
    results.append({
        "file":       rel,
        "true_label": class_names[t],
        "pred_label": class_names[p],
        "confidence": round(confidence, 2),
        "correct":    correct,
    })

total   = len(results)
correct = sum(r["correct"] for r in results)

print(f"\n{'='*80}")
print("  PER-CLASS")
for i, c in enumerate(class_names):
    mask = y_true == i
    if mask.any():
        print(f"  {c:<45} {int((y_pred[mask] == i).sum())}/{int(mask.sum())}")

print(f"\n{'='*80}")
print("  RESULTS SUMMARY")
print(f"  Total images tested : {total}")
print(f"  Correct             : {correct}")
print(f"  Incorrect           : {total - correct}")
print(f"  Test Accuracy       : {test_acc:.1%}")
print(f"  Test Loss           : {test_loss:.3f}")
print(f"{'='*80}")

report = {
    "total":        total,
    "correct":      correct,
    "incorrect":    total - correct,
    "accuracy_pct": round(test_acc * 100, 2),
    "loss":         round(float(test_loss), 4),
    "per_image":    results,
}
report_path = os.path.join(TEST_DIR, "test_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Full report saved to: {report_path}")
