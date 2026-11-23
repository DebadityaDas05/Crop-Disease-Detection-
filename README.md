# Crop Disease Detection Using Deep Transfer Learning (PyTorch)
### *Addressing UN Sustainable Development Goal 2: Zero Hunger*
**Course:** PCCCS582 Neural Networks & Deep Learning (NNDL)

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Project Objectives](#2-project-objectives)
3. [Dataset Description](#3-dataset-description)
4. [Data Preprocessing & Augmentation](#4-data-preprocessing--augmentation)
5. [System & Model Architecture](#5-system--model-architecture)
6. [Implementation & Training Procedure](#6-implementation--training-procedure)
7. [Testing & Evaluation Results](#7-testing--evaluation-results)
8. [Interactive Prototype (Gradio Demo)](#8-interactive-prototype-gradio-demo)
9. [Project Structure](#9-project-structure)
10. [Setup & Execution Guide](#10-setup--execution-guide)
11. [Limitations & Future Scope](#11-limitations--future-scope)

---

## 1. Problem Statement

Plant diseases pose a severe threat to global food security, causing an estimated **20% to 40% loss of global agricultural production annually**. For smallholder farmers who produce over 30% of the world's food supply, late or inaccurate diagnosis of leaf diseases often leads to catastrophic crop failure, excessive chemical pesticide misuse, and severe economic losses.

Traditional disease diagnosis relies heavily on expert manual visual scouting, which is:
- **Labor-intensive and slow:** Agricultural specialists are scarce in rural and developing regions.
- **Error-prone:** Early-stage visual symptoms of bacterial, fungal, and viral infections often overlap.
- **Inaccessible:** Smallholder farmers lack real-time access to diagnostic laboratories.

### Alignment with UN SDG 2: Zero Hunger
This project directly addresses **United Nations Sustainable Development Goal 2 (Zero Hunger)**, specifically:
- **Target 2.4:** *Ensure sustainable food production systems and implement resilient agricultural practices that increase productivity and production.*
- Providing farmers with fast, accurate, and accessible automated leaf disease classification ensures early intervention, preserves crop yields, reduces indiscriminate chemical spraying, and supports resilient food supply chains.

---

## 2. Project Objectives

1. **Deep Learning Classification:** Train a Convolutional Neural Network (CNN) using Transfer Learning on crop-leaf imagery to distinguish between healthy leaves and 38 distinct crop-disease categories across 14 crop species.
2. **Robust Generalization:** Implement a multi-stage training strategy (frozen backbone feature extraction followed by end-to-end fine-tuning) with rigorous data augmentation to prevent overfitting.
3. **Comprehensive Empirical Evaluation:** Benchmark model performance on an independent, held-out test set (9,224 images) using multi-class precision, recall, F1-score, and confusion matrix analysis.
4. **Deployable Prototype:** Build an accessible, real-time interactive prototype application using **Gradio** that enables users to upload leaf photos and receive instant disease diagnoses with probabilistic confidence scores.

---

## 3. Dataset Description

The project uses the benchmark **PlantVillage** dataset, containing **61,486 high-resolution leaf images** categorized into **39 distinct classes** (14 plant species, 38 crop-disease conditions, and 1 non-leaf background class).

### Dataset Breakdown
- **Total Images:** 61,486
- **Number of Classes:** 39
- **Crops Covered (14 species):** Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange (Citrus), Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
- **Conditions Covered:** Healthy foliage, fungal infections (scab, rust, powdery mildew, blights), bacterial infections (bacterial spot), viral infections (yellow leaf curl, mosaic virus), and pest infestations (spider mites).
- **Background Class:** `Background_without_leaves` (prevents false positives on non-foliar images).

### Data Partitioning
The dataset is split randomly using a fixed random seed (`seed=42`) to ensure strict reproducibility:
- **Training Set (70%):** 43,040 images
- **Validation Set (15%):** 9,222 images
- **Held-out Test Set (15%):** 9,224 images

---

## 4. Data Preprocessing & Augmentation

Raw agricultural images vary in lighting and orientation. To standardize inputs for deep network ingestion and improve generalization against real-world variations, the following pipeline is implemented:

### Preprocessing Pipeline
1. **Resolution Standardization:** Images are resized to **$224 \times 224$ pixels** matching the input resolution of the backbone network.
2. **Channel Normalization:** Pixel values are scaled to $[0, 1]$ and normalized using standard ImageNet mean ($\mu = [0.485, 0.456, 0.406]$) and standard deviation ($\sigma = [0.229, 0.224, 0.225]$).

### Data Augmentation (Training Split Only)
To simulate environmental variations and prevent memorization, the training pipeline incorporates:
- **Random Horizontal Flip:** Simulates leaf orientation asymmetry.
- **Random Affine Rotation ($\pm 20^\circ$):** Accounts for variable camera angles in the field.
- **Color Jitter (Brightness $\pm 0.2$, Contrast $\pm 0.2$, Saturation $\pm 0.2$):** Simulates varying sunlight, shadow, and overcast weather conditions.

Validation and test sets use deterministic resizing and normalization without stochastic augmentation.

---

## 5. System & Model Architecture

### Backbone Selection: MobileNetV2
Rather than training a massive network from scratch, the system utilizes **MobileNetV2** pretrained on **ImageNet-1k**:
- **Why MobileNetV2?** Agricultural diagnosis tools require low computational overhead to run on edge devices, drones, or smartphones with limited battery and memory.
- **Inverted Residuals & Linear Bottlenecks:** Uses depthwise separable convolutions that dramatically reduce parameter count and multiply-accumulate operations (MACs) while preserving representational capacity.
- **Backbone Parameters:** ~2.2 million parameters (model file size: **~9.3 MB**).

### Custom Classification Head
The original ImageNet 1,000-class output layer is replaced with a specialized classification head:
$$\text{Input Feature Vector } (1280\text{-d}) \longrightarrow \text{Dropout}(p=0.3) \longrightarrow \text{Linear}(1280 \to 39) \longrightarrow \text{Cross-Entropy Loss}$$

```text
[Input Leaf Image: 3 x 224 x 224]
               │
               ▼
   [MobileNetV2 Feature Extractor]
     - Inverted Residual Blocks
     - Depthwise Separable Convolutions
               │
               ▼
    [Global Average Pooling]
               │
               ▼
       [1280-d Feature Vector]
               │
               ▼
        [Dropout (p = 0.3)]
               │
               ▼
     [Linear Layer: 1280 -> 39]
               │
               ▼
    [Softmax / Class Probabilities (39 classes)]
```

---

## 6. Implementation & Training Procedure

### Implementation Details
- **Framework:** PyTorch 2.x & Torchvision (CUDA accelerated)
- **Loss Function:** Categorical Cross-Entropy Loss (`nn.CrossEntropyLoss`)
- **Optimizer:** Adam
- **Batch Size:** 32 (with GPU memory pinning)

### Two-Phase Transfer Learning Strategy
1. **Phase 1: Feature Extraction (Epochs 1 – 4):**
   - The entire MobileNetV2 backbone is **frozen** (`requires_grad = False`).
   - Only the custom 39-class classifier head is trained with learning rate $\eta = 10^{-3}$.
   - Allows the new classification head to stabilize without disrupting pretrained feature weights.
2. **Phase 2: Full Network Fine-Tuning (Epochs 5 – 8):**
   - The entire network backbone is **unfrozen** (`requires_grad = True`).
   - The learning rate is reduced by a factor of 10 to $\eta = 10^{-4}$.
   - Backpropagation fine-tunes all convolutional layers end-to-end to adapt general visual filters to subtle botanical disease symptoms.

### Model Checkpointing
Validation accuracy is monitored after each epoch. The model weights yielding the highest validation accuracy are saved to `crop_model.pt`.

---

## 7. Testing & Evaluation Results

The best checkpoint was evaluated on the **9,224 unseen test samples**.

### Overall Performance Metrics
| Metric | Score |
| :--- | :--- |
| **Overall Test Accuracy** | **99.5%** |
| **Macro Average Precision** | **99.5%** |
| **Macro Average Recall** | **99.4%** |
| **Macro Average F1-Score** | **99.4%** |
| **Weighted Average F1-Score** | **99.5%** |

### Per-Class Performance Summary (Sample Representative Classes)
| Crop & Condition | Precision | Recall | F1-Score | Test Samples |
| :--- | :---: | :---: | :---: | :---: |
| **Apple — Apple Scab** | 0.993 | 1.000 | 0.996 | 140 |
| **Apple — Black Rot** | 0.992 | 1.000 | 0.996 | 131 |
| **Apple — Cedar Apple Rust** | 1.000 | 1.000 | 1.000 | 149 |
| **Apple — Healthy** | 0.996 | 1.000 | 0.998 | 245 |
| **Corn — Common Rust** | 1.000 | 1.000 | 1.000 | 178 |
| **Corn — Northern Leaf Blight** | 0.958 | 0.970 | 0.964 | 166 |
| **Grape — Black Rot** | 0.994 | 1.000 | 0.997 | 180 |
| **Orange — Citrus Greening (Huanglongbing)** | 0.999 | 1.000 | 0.999 | 856 |
| **Potato — Early Blight** | 1.000 | 1.000 | 1.000 | 157 |
| **Potato — Late Blight** | 0.989 | 1.000 | 0.994 | 172 |
| **Tomato — Bacterial Spot** | 0.991 | 0.997 | 0.994 | 346 |
| **Tomato — Early Blight** | 0.986 | 0.960 | 0.973 | 151 |
| **Tomato — Late Blight** | 0.997 | 0.986 | 0.991 | 289 |
| **Tomato — Yellow Leaf Curl Virus** | 0.999 | 0.997 | 0.998 | 799 |
| **Tomato — Healthy** | 1.000 | 0.996 | 0.998 | 230 |
| **Background (Non-leaf imagery)** | 1.000 | 0.980 | 0.990 | 151 |

### Visual Artifacts
- **`confusion_matrix.png`:** Demonstrates minimal off-diagonal confusion across all 39 classes.
- **`training_curves.png`:** Illustrates steady loss convergence and high validation accuracy across training epochs.

---

## 8. Interactive Prototype (Gradio Demo)

An interactive, user-friendly graphical web prototype is built with **Gradio** (`crop_disease_detection.py demo`):
- **Image Upload:** Supports drag-and-drop or file browsing for leaf photos (`.jpg`, `.png`, `.jpeg`).
- **Real-Time Inference:** Normalizes the input, runs inference through the trained MobileNetV2 network, and applies softmax.
- **Top-5 Probabilistic Breakdown:** Displays the most likely disease diagnosis alongside confidence percentages for transparent agricultural decision support.

---

## 9. Project Structure

```text
nndl-crop-disease-detection/
├── .gitignore                     # Git ignore rules for virtual environments, dataset & weights
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Python package dependencies
├── crop_disease_detection.py      # Core script (training, prediction, Gradio web demo)
├── NN_DLProject.ipynb             # Jupyter Notebook for exploratory development
├── test_report.txt                # Full classification report on held-out test set
├── classes.json                   # List of 39 target class labels
├── confusion_matrix.png           # Confusion matrix visualization
├── training_curves.png            # Loss and accuracy progression plots
├── crop_model.pt                  # Best trained PyTorch model checkpoint (generated)
└── data/
    └── PlantVillage/              # Extracted dataset subfolders (39 classes)
```

---

## 10. Setup & Execution Guide

### Prerequisites
- Python 3.10+ (compatible with Python 3.11, 3.12, 3.14)
- Recommended: NVIDIA GPU with CUDA support for accelerated training

### 1. Environment Setup
```powershell
# Create an isolated virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```
*(For dedicated CUDA acceleration on Windows, install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org)).*

### 3. Usage Commands

#### A. Train the Model
```powershell
python crop_disease_detection.py train --data data/PlantVillage --epochs 8 --bs 32 --workers 2
```
*Outputs: `crop_model.pt`, `classes.json`, `confusion_matrix.png`, and `training_curves.png`.*

#### B. Single Image Prediction (CLI)
```powershell
python crop_disease_detection.py predict --image path/to/leaf.jpg
```

#### C. Launch the Interactive Web Prototype (Gradio)
```powershell
python crop_disease_detection.py demo
```
*Launches a local web server (e.g., `http://127.0.0.1:7860`) accessible directly from any browser.*

---

## 11. Limitations & Future Scope

### Limitations
1. **Controlled vs. In-Field Imagery:** The PlantVillage dataset consists largely of individual leaves photographed against uniform laboratory backdrops. Real-world images taken in open fields feature complex backgrounds, soil, varying shadows, and overlapping vegetation.
2. **Single-Disease Assumption:** The current architecture models disease diagnosis as a single-label multi-class problem. In natural agricultural ecosystems, plants may suffer from co-infections (e.g., fungal and viral diseases simultaneously).
3. **Foliar Scope:** The dataset focuses primarily on foliar (leaf) symptoms and does not account for stem, root, or fruit pathology.

### Future Improvements
1. **In-the-Wild Domain Adaptation:** Retrain or fine-tune with datasets containing natural field backgrounds and diverse lighting conditions.
2. **Multi-Label Pathology:** Formulate the classification head as a multi-label sigmoid task to detect concurrent infections.
3. **Edge Optimization:** Quantize the trained PyTorch model to **INT8 / ONNX / TorchScript / TFLite** to run offline on low-power mobile devices for rural farmers without constant internet connectivity.
4. **Grad-CAM Saliency Maps:** Integrate visual explainability (Gradient-weighted Class Activation Mapping) into the Gradio UI so farmers and agronomists can verify which leaf regions influenced the model's diagnosis.
