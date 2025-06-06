# Surgical Suture Quality Classification using Transfer Learning

This repository contains the code and documentation for training deep learning models to classify the quality of surgical sutures using transfer learning. The research is described in the paper:

**"Evaluation of transfer learning efficacy for surgical suture quality classification on limited datasets"**, *Journal of Imaging, 2025*.

## 📁 Dataset
The dataset includes images of three types of surgical sutures:
- Continuous over-and-over open suture (COOS)
- Interrupted laparoscopic suture (ILS)
- Interrupted open vascular suture (IOVS)

Each image is labeled as either "high-quality" or "low-quality".  

## 🧠 Models Evaluated
The following pre-trained CNN architectures were evaluated using transfer learning:
- EfficientNetB0
- ResNet50V2
- MobileNetV3Large
- VGG16 / VGG19
- InceptionV3
- Xception
- DenseNet121

## 🛠️ Requirements
- Python 3.11
- TensorFlow 2.18.0
- scikit-learn 1.2.2
- NumPy 2.2.4

## 📋 Training Pipeline
1. Image preprocessing and resizing
2. Data augmentation
3. Two-stage transfer learning:
   - Training only top classification layers
   - Fine-tuning with unfrozen base layers
4. 5-fold stratified cross-validation
5. Performance metrics: F1-score, AUC-ROC, Precision, Recall, Accuracy