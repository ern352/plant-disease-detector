# 🌿 Plant Disease Detector (Deep Learning)

This project uses a Convolutional Neural Network (CNN) to detect plant diseases from leaf images using the PlantVillage dataset.

## 📊 Dataset
- PlantVillage Dataset (Kaggle: `emmarex/plantdisease`), 20,638 images
- 15 classes across pepper, potato and tomato leaves
- Images resized to 128x128
- Split:
  - **50 test images** (3–4 per class) moved out of the dataset before training, so the model never sees them
  - the rest split 80% training / 20% validation

## 🧠 Model
- Rescaling (pixels to 0–1, built into the model)
- Data augmentation (random flip and rotation)
- 3 × Conv2D + MaxPooling
- Dropout (0.5) to reduce overfitting
- Dense layers with a softmax output over 15 classes
- Early stopping on validation loss

## ▶️ How to run
1. Open `Plant_Disease_Detector.ipynb` in Google Colab and set **Runtime → Change runtime type → GPU**.
2. Run all cells. When asked, upload your `kaggle.json` (never commit this file).
3. Step 8 prints the accuracy on the 50 held-out images. Step 9 downloads them with `test_report.json`.
4. To re-test later without retraining, run `batch_test.py` (see its docstring).

## ⚙️ Technologies
- Python
- TensorFlow / Keras
- Google Colab
- Matplotlib

## 📈 Results
| Split | Images | Accuracy |
|---|---|---|
| Training | ~16,470 | (fill in from step 6) |
| Validation | ~4,120 | (fill in from step 6) |
| Test (held out) | 50 | (fill in from step 8) |

## 🚀 Future Improvements
- Transfer Learning (MobileNetV2)
- Testing on real field photos, not only lab images
- Deploy as a mobile or web app

## 👨‍💻 Author
Your Name
