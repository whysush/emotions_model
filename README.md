# Facial Emotion Recognition using TensorFlow and Gradio

This project implements a complete facial emotion recognition system trained on the FER2013 dataset using a custom Convolutional Neural Network (CNN). It supports both static image inference and real-time webcam-based emotion detection through Gradio.

---

## Features

- Uses FER2013, a widely used 7-class facial expression dataset  
- Custom CNN architecture built with TensorFlow and Keras  
- Efficient preprocessing using tf.data pipelines  
- Real-time webcam inference with Gradio  
- Face detection using OpenCV Haar Cascades  
- Fully reproducible workflow designed for Google Colab  

---

## Model Architecture

The model processes 48×48 grayscale facial images through three convolutional blocks, followed by dense layers.

### Architecture Overview

1. **Input Layer**: 48×48×1  
2. **Convolution Block 1**
   - Conv2D (32) → BatchNorm  
   - Conv2D (32) → BatchNorm  
   - MaxPooling  
   - Dropout  
3. **Convolution Block 2**
   - Conv2D (64) → BatchNorm  
   - Conv2D (64) → BatchNorm  
   - MaxPooling  
   - Dropout  
4. **Convolution Block 3**
   - Conv2D (128) → BatchNorm  
   - Conv2D (128) → BatchNorm  
   - MaxPooling  
   - Dropout  
5. **Dense Layers**
   - Flatten  
   - Dense(256) → BatchNorm → Dropout  
   - Dense(7) with softmax activation  

### Training Configuration
- Optimizer: Adam (learning rate 1e-3)  
- Loss: Sparse Categorical Crossentropy  
- Epochs: ~25  
- Batch Size: 64  

---

## Tech Stack

**Libraries Used**
- TensorFlow / Keras  
- TensorFlow Datasets  
- NumPy  
- Matplotlib  
- OpenCV  
- Gradio  

**Techniques Applied**
- Custom CNN design  
- Batch Normalization and Dropout regularization  
- tf.data preprocessing pipelines  
- Haar Cascade face detection  
- Real-time webcam streaming with Gradio  

---

## Workflow Summary

1. Load FER2013 using `tensorflow_datasets`  
2. Preprocess images (resize, normalize, grayscale)  
3. Train the custom CNN  
4. Validate on a held-out set  
5. Evaluate on FER2013 test split  
6. Save the trained model  
7. Run inference on uploaded images and webcam input through Gradio  

