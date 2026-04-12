# ⚡ AI vs Real Image Detector

A deep learning-powered system that detects whether an image is **real or AI-generated**, enhanced with **region-level analysis and explainability**.

---

## 🚀 Overview

With the rise of generative AI, distinguishing real images from synthetic ones has become increasingly challenging.

This project builds an **end-to-end computer vision pipeline** that:
- Classifies images as **Real** or **AI-generated**
- Highlights **suspicious regions** using patch-level analysis
- Provides **human-readable explanations** for model decisions
- Offers an interactive **web app interface**

---

## 🧠 Key Features

### 🔍 Patch-Level Detection (Core Innovation)
Instead of giving a single prediction, the image is divided into a **4×4 grid** and each region is analyzed independently.

👉 This allows:
- Detection of **localized AI artifacts**
- Better interpretability compared to traditional heatmaps

---

### 🎯 Suspicious Region Highlighting
Only regions with strong AI signals are highlighted, making the output:
- Cleaner
- More meaningful
- Easier to understand

---

### 🧠 Smart Explanation Engine
The system generates contextual explanations based on:
- Distribution of suspicious regions
- Location (center vs edges)
- Confidence levels

---

### 🎨 Interactive UI (Streamlit)
- Clean, dark-themed interface  
- Real-time predictions  
- Visual + textual feedback  
- Designed for demonstration and usability  

---

## 🏗️ Project Structure

ai-vs-real-image-classifier/
│
├── data/
│ ├── raw/
│ ├── processed/
│ ├── train/
│ └── test/
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ └── 03_model_training.ipynb
│
├── src/
│ ├── preprocess.py
│ ├── data_loader.py
│ ├── train.py
│ └── evaluate.py
│
├── models/
│ └── saved_models/
│
├── deployment/
│ ├── app.py
│ └── assets/
│
├── outputs/
│ ├── plots/
│ └── reports/
│
└── requirements.txt


---

## ⚙️ Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **OpenCV**
- **Streamlit**
- **Scikit-learn**

---

## 📊 Model Details

- Architecture: **ResNet18 (Transfer Learning)**
- Input Size: `224 × 224`
- Training Strategy:
  - Frozen backbone
  - Fine-tuned classifier layer
- Accuracy: **~94% on test set**

---

## ⚙️ Pipeline

1. Data Collection (Real + AI images)
2. Preprocessing (resize, clean, split)
3. Model Training (ResNet18)
4. Evaluation (accuracy + confusion matrix)
5. Deployment (Streamlit app)
6. Explainability (patch-level analysis)

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt


---

## ⚙️ Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **OpenCV**
- **Streamlit**
- **Scikit-learn**

---

## 📊 Model Details

- Architecture: **ResNet18 (Transfer Learning)**
- Input Size: `224 × 224`
- Training Strategy:
  - Frozen backbone
  - Fine-tuned classifier layer
- Accuracy: **~94% on test set**

---

## ⚙️ Pipeline

1. Data Collection (Real + AI images)
2. Preprocessing (resize, clean, split)
3. Model Training (ResNet18)
4. Evaluation (accuracy + confusion matrix)
5. Deployment (Streamlit app)
6. Explainability (patch-level analysis)

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
