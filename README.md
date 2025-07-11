# Image-Captioning-AI
# 🧠 Emotion-Aware Multimodal Image Captioning  
**Using BERT-Based Emotion Extraction, ResNet50 Visual Features, and LSTM-Based Language Generation**

This project integrates Computer Vision, Natural Language Processing, and Affective Computing to generate emotionally aligned captions for images.

---

## 🔍 Overview

Given an image and a target emotion (e.g., joy, sadness, fear), the model generates a caption that describes the image while also reflecting the chosen emotional tone.

The system leverages:
- ResNet50 for visual feature extraction
- BERT for emotion classification on human-written captions
- LSTM for language generation conditioned on image and emotion features

---

## 🧩 Architecture

Image (ResNet50) ---> Visual Feature (2048-d)  
                             |  
Caption (BERT Emotion Classifier) ---> Emotion Vector (6-d one-hot)  
                             |  
         [Fused Feature Vector]  
                   |  
     --> LSTM Decoder --> Generated Caption

---

## 🛠️ Tech Stack

PyTorch, Hugging Face Transformers (BERT), TorchVision (ResNet50), LSTM, Pandas, JSON, Pillow (PIL)

---

## 📁 Dataset

- 1000+ image-caption pairs (from captions.txt)
- Each image has:
  - 1 caption (human-written)
  - Emotion label (predicted using BERT)
  - Visual features (ResNet50)
- Final dataset stored as: final_preprocessed_dataset.json

---

## ⚙️ Pipeline

1. **Caption Emotion Classification**
   - Classify each caption using `nateraw/bert-base-uncased-emotion`

2. **Visual Feature Extraction**
   - Extract 2048-d image embeddings using pretrained ResNet50

3. **Dataset Building**
   - Combine image + caption + emotion + token IDs into a single JSON dataset

4. **Model Training**
   - Use LSTM decoder to generate captions conditioned on image and emotion

5. **Inference**
   - Given a test image and emotion label → generate a human-like caption

---

## 🧪 Sample Output

Input Image: A child playing in the park  
Target Emotion: Joy  
Generated Caption: "A smiling child playing happily with toys"

---
