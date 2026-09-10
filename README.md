# 👁️ SPECTRA – Violence Detection System

**SPECTRA** is a Deep Learning and Computer Vision-based system designed to detect violent activities in video footage.

The system uses **MobileNetV2** to analyze video frames and classify them as **Violent** or **Non-Violent**. A Streamlit interface provides an interactive way to upload videos and view detection results.

## ✨ Features

* 🎥 Video upload and frame-by-frame analysis
* 🧠 MobileNetV2-based violence classification
* 📊 Prediction confidence scores
* 🚨 Violence detection alerts
* 🖥️ Interactive Streamlit interface

## 🔄 Workflow

```text
Video Input
     ↓
Frame Extraction
     ↓
Image Preprocessing
     ↓
MobileNetV2
     ↓
Violent / Non-Violent
     ↓
Confidence Score
     ↓
Alert
```

## 📊 Dataset

The dataset was obtained from **Kaggle** and contains video samples categorized into **Violent** and **Non-Violent** classes.

The dataset is not included in this repository.

## 🛠️ Tech Stack

**Python • TensorFlow • Keras • MobileNetV2 • OpenCV • NumPy • Streamlit**

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run my_violence_detector.py
```

## 📌 Project Information

**Project:** SPECTRA
**Domain:** Deep Learning & Computer Vision
**Model:** MobileNetV2
**Application:** Video Violence Detection
