# 🧬 Skin Disease CNN Classification with Explainable AI (Grad-CAM)

An end-to-end deep learning project for automated skin disease classification using Convolutional Neural Networks, Transfer Learning, and Explainable AI (XAI) techniques. This repository covers the entire ML lifecycle — from exploratory data analysis (EDA) and dataset inspection to model training, fine-tuning, evaluation, explainability with Grad-CAM, and deployment using Streamlit.

---

## 📌 Project Overview

Skin diseases often present visually similar symptoms, making accurate diagnosis challenging even for trained professionals. This project aims to assist dermatological screening by leveraging deep learning models capable of learning discriminative visual features from skin images while also providing **visual explanations** for their predictions.

### Key Highlights

* ✅ Classification of **8 skin disease categories**
* 🧠 Transfer Learning using **ResNet18**
* ⚖️ Class imbalance handling with **weighted loss**
* 📈 Accuracy improved from **67.09% → 88.03% → 95.73%**
* 🔍 **Grad-CAM** for model interpretability
* 🖥️ Deployed as an interactive **Streamlit web ap

> ⚠️ **Disclaimer:** This system is intended for educational and research purposes only and should not replace professional medical diagnosis.

---

## 📂 Dataset Description

### Dataset Structure

The dataset is organized into two primary directories:

```
Dataset/
├── train_set/
│   ├── BA-cellulitis/
│   ├── BA-impetigo/
│   ├── FU-athlete-foot/
│   ├── FU-nail-fungus/
│   ├── FU-ringworm/
│   ├── PA-cutaneous-larva-migrans/
│   ├── VI-chickenpox/
│   └── VI-shingles/
└── test_set/
    └── (same 8 classes)
```

Both training and testing sets consistently contain **8 skin disease classes**, ensuring a clean and reliable evaluation pipeline.

---

## 🏷️ Skin Disease Classes

1. **BA-cellulitis**
2. **BA-impetigo**
3. **FU-athlete-foot**
4. **FU-nail-fungus**
5. **FU-ringworm**
6. **PA-cutaneous-larva-migrans**
7. **VI-chickenpox**
8. **VI-shingles**

---

### 🔍 Key EDA Observations

* The dataset is **relatively balanced**, though some mild imbalance exists (e.g., BA-impetigo).
* Sample images show **high variability in lighting, background, and image quality**, reflecting real-world medical data.
* No missing or corrupted image files were detected.

---

## 🧠 Model Architecture

### Baseline Model

* Custom CNN trained from scratch
* Achieved **67.09% accuracy**
* Served as a reference baseline

### Transfer Learning Model

* **Backbone:** ResNet18 (pre-trained on ImageNet)
* **Classifier Head:** Custom fully connected layers
* **Loss Function:** CrossEntropyLoss with class weights
* **Optimizer:** Adam

---

## ⚖️ Handling Class Imbalance

To mitigate bias toward majority classes:

* **Class weights** were computed (range: ~0.85 – 1.45)
* Integrated directly into the loss function
* Resulted in significantly improved macro-averaged metrics

---

## 📈 Training Strategy & Results

### Phase 1: Frozen Backbone (Transfer Learning)

* Only classifier head trained
* Epochs: 15

**Performance:**

* Accuracy: **88.03%**
* Macro Recall: **88.03%**
* Macro F1-score: **86.81%**

➡️ Large improvement over baseline CNN

---

### Phase 2: Fine-Tuning with Unfrozen Layers

* Unfroze **layer4 of ResNet18**
* Applied **learning rate scheduler**
* Continued training for 15 epochs

### 🔥 Final Model Performance

| Metric             | Value      |
| ------------------ | ---------- |
| **Test Accuracy**  | **95.73%** |
| **Macro Recall**   | **95%**    |
| **Macro F1-score** | **95%**    |

➡️ Significant improvement over both baseline CNN (67.09%) and initial transfer learning (88.03%).

---

## 📊 Per-Class Performance Highlights

### Strengths

* **100% Recall:** BA-impetigo, FU-nail-fungus, VI-chickenpox
* **High F1-scores:**

  * FU-nail-fungus: 98.46%
  * VI-chickenpox: 98.51%
* Excellent generalization across most classes

### Weaknesses

* FU-ringworm shows slightly lower recall (~83%)
* BA-impetigo precision (~87%) indicates some false positives

---

## 📉 Error Analysis

* Confusion matrix revealed minor confusion between visually similar fungal infections
* Underperforming classes benefit most from additional data diversity

---

## 🖥️ Deployment

The trained model is deployed using **Streamlit**:

* Upload a skin image
* Get predicted disease, confidence score, and medical suggestion
* Optimized for **Python 3.10 + TensorFlow 2.13+** compatibility

---

## 🧪 Visualizations Included

* Training vs Testing loss curves
* Accuracy, Recall, F1-score plots
* Confusion matrix
* Sample predictions

---

## 🔍 Model Explainability with Grad-CAM

### Why Grad-CAM?

In medical AI, interpretability is critical. Grad-CAM (Gradient-weighted Class Activation Mapping) provides visual explanations by highlighting image regions that most influenced the model’s prediction.

### Grad-CAM Methodology

* Target Layer: `ResNet18 → layer4[1].conv2`
* Gradients + feature maps used to compute class-discriminative heatmaps
* Heatmaps overlaid on original images

---

### Insights from Grad-CAM Visualizations

#### ✅ Correct Predictions

* Heatmaps focus on **lesion-affected regions**
* Minimal attention to background or artifacts
* Confirms learning of clinically meaningful features

#### ❌ Incorrect Prediction Example

* **Actual:** VI-shingles
* **Predicted:** BA-impetigo

Grad-CAM showed:

* Strong focus on lesion region
* Misclassification due to **visual similarity**, not random behavior

➡️ Indicates correct localization but imperfect feature interpretation.

> 🧠 *Explainability is not optional in medical AI — Grad-CAM bridges the gap between model performance and human trust.*

---

### Interpretability & Bias Analysis

* Grad-CAM verifies the model relies on **pathological regions**
* Helps identify reasons for misclassification
* Useful for detecting dataset bias or shortcut learning

Current observations suggest **low bias risk**, but broader Grad-CAM analysis is recommended.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision (ResNet18)
* NumPy, Matplotlib, Seaborn
* Streamlit

---

## 👤 Author

**Abhay Singh**
BCA (AI & ML + Data Science)
Vivekananda Institute of Professional Studies

🔗 GitHub: [https://github.com/AbhaySingh71](https://github.com/AbhaySingh71)
🔗 LinkedIn: [https://www.linkedin.com/in/abhay-singh-050a5b293/](https://www.linkedin.com/in/abhay-singh-050a5b293/)

---

⭐ If you find this project helpful, consider giving it a star!
