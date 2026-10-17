# 🌿 Plant Disease Detector (Deep Learning)

This project uses a Convolutional Neural Network (CNN) to detect plant diseases from leaf images using the PlantVillage dataset.

**🌐 Live demo:** https://huggingface.co/spaces/ERN-low/plant-disease-detector — upload a pepper, potato or tomato leaf photo and get a prediction.

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
2. Add your Kaggle API token as a Colab secret named `KAGGLE_API_TOKEN` (🔑 icon in the left sidebar, Notebook access on), then run all cells. Never put the token in the code.
3. Step 8 prints the accuracy on the 50 held-out images. Step 9 downloads them with `test_report.json`.
4. To re-test later without retraining, run `batch_test.py` (see its docstring).

## ⚙️ Technologies
- Python
- TensorFlow / Keras
- Google Colab
- Matplotlib
- Gradio + Hugging Face Spaces (website)

## 📈 Results
| Split | Images | Accuracy |
|---|---|---|
| Training | 16,471 | 92.7% |
| Validation | 4,117 | 93.4% |
| **Test (held out)** | **50** | **94.0% (47/50)** |

- Trained on a Colab T4 GPU (TensorFlow 2.21.0, Keras 3.13.2). Early stopping kept the weights from epoch 11 of 14.
- Training accuracy is a little lower than validation because data augmentation and dropout are only active during training.
- All 3 test mistakes were predicted as *Tomato Early blight*: a pepper bacterial spot, a potato late blight and a tomato late blight leaf.

## 🚀 Future Improvements
- Transfer Learning (MobileNetV2)
- Testing on real field photos, not only lab images
- Mobile app version

## 👨‍💻 Author
Your Name
