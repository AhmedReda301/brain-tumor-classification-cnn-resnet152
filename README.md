# <span style="color:#FFA500;"> Brain Tumor Classification (CNN & ResNet-152)</span>

A <span style="color:#00BFFF;">deep learning</span> project for classifying MRI brain scans into **Brain Tumor** and **Healthy** categories.  
It includes full <span style="color:#32CD32;">data preprocessing</span>, <span style="color:#32CD32;">model training</span>, and <span style="color:#32CD32;">evaluation</span> using a **custom CNN** and a **fine-tuned ResNet-152**.

---

## <span style="color:#32CD32;"> Key Highlights:</span>  
-  **Data Handling:** Organized MRI dataset with tumor and healthy brain images.  
-  **Modeling:** Custom CNN architecture and ResNet-152 fine-tuned for classification.  
-  **Evaluation:** Detailed classification reports with <span style="color:#DC143C;">Precision</span>, <span style="color:#DC143C;">Recall</span>, and <span style="color:#DC143C;">F1-Score</span>.  
-  **Visualization:** Training/validation accuracy & loss plots for each model.

## <span style="color:#32CD32;"> Dataset:</span>  
-  **Source:** [Brain Tumor Dataset - Kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)  
-  **Description:** MRI scans labeled as **Brain Tumor** or **Healthy**.  
-  **Classes:**  
  - `0` → Healthy  
  - `1` → Brain Tumor

## <span style="color:#32CD32;"> Models:</span>  

Download the Model Checkpoint
Use the Kaggle Hub API:
```python
import kagglehub

# Download latest version
path = kagglehub.model_download("ahmedredaahmedali/cnn-model-resnet152/pyTorch/default")

print("Path to model files:", path)

```

#### CNN_TUMOR Architecture Summary

-  **Input:** Image tensor of shape `(C, H, W)` (channels, height, width)

-  **Convolutional Layers:** 4 layers with increasing filters:
  - Conv1: `initial_filters` filters, 3×3 kernel
  - Conv2: 2× `initial_filters` filters, 3×3 kernel
  - Conv3: 4× `initial_filters` filters, 3×3 kernel
  - Conv4: 8× `initial_filters` filters, 3×3 kernel  
Each conv layer is followed by ReLU activation and 2×2 max pooling (downsampling spatial dimensions by half).

-  **Fully Connected Layers:**
  - FC1: `num_fc1` units with ReLU and Dropout
  - FC2: Output layer with units equal to number of classes

-  **Output:** Log-softmax probabilities for classification

-  **Total layers:**
  - 4 convolutional layers
  - 2 fully connected layers

---

#### ResNet152_TUMOR Summary

-  **Backbone:** Pretrained ResNet-152 (ImageNet weights by default)

-  **Fine-tuning strategy:**
  - Freeze first half of ResNet-152 layers (weights not updated)
  - Unfreeze second half for fine-tuning on tumor classification

-  **Modification:** Final fully connected layer replaced to output `num_classes` units

-  **Forward pass:** Utilizes standard ResNet forward method with custom classification head

-  **Freezing rationale:**  
  Freezing early layers preserves pretrained low-level features, while unfreezing later layers adapts higher-level representations to the target dataset.

---

## <span style="color:#1E90FF;"> Model Performance:</span>  

| Model           | Train Acc | Val Acc | Train Loss | Val Loss | F1-Train | F1-Val |
|-----------------|-----------|---------|------------|----------|----------|--------|
| CNN_TUMOR       | 0.9828    | 0.9812  | 0.0491     | 0.0560   | 0.9891   | 0.9815 |
| ResNet152_TUMOR | 0.9916    | 0.9870  | 0.015      | 0.019    | 0.9923   | 0.9880 |


---

### CNN_TUMOR Final Classification Report (Train):
 -  **Training:** Batch size=64, epochs=35

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.9886    | 0.9874 | 0.9880   | 1670    |
| 1     | 0.9896    | 0.9905 | 0.9901   | 2010    |

- **Accuracy:** 0.9891  
- **Macro Avg:** Precision 0.9891, Recall 0.9890, F1-Score 0.9890  
- **Weighted Avg:** Precision 0.9891, Recall 0.9891, F1-Score 0.9891  

---

### CNN_TUMOR Final Classification Report (Validation):
-  **Training:** Batch size=16, epochs=10

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.9831    | 0.9760 | 0.9795   | 417     |
| 1     | 0.9802    | 0.9861 | 0.9832   | 503     |

- **Accuracy:** 0.9815  
- **Macro Avg:** Precision 0.9817, Recall 0.9811, F1-Score 0.9813  
- **Weighted Avg:** Precision 0.9815, Recall 0.9815, F1-Score 0.9815 
---


##  How to Run
1. Clone the repository or navigate to the project folder.  
2. Install the required libraries:  
---
```bash
pip install -r requirements.txt
```
3. Run training

- 3.1 Train using Custom CNN
```bash
python src/train.py --model cnn
```
- 3.2 Train using ResNet-152
```bash
python src/train.py --model resnet152
```


## <span style="color:#FF69B4;"> Project Structure:</span>

<details>
<summary> Click to expand</summary>

```bash
Brain Tumor Classification/
├── Data/                                   # Dataset folder
│   └── Brain Tumor Data Set/
│       ├── Brain Tumor/
│       └── Healthy/
│
├── models/                                 # Saved trained models
│   ├── CNN_TUMOR.pth
│   └── ResNet152_TUMOR.pth
│
├── requirments/                            # Project dependencies
│   └── requirements.txt
│
├── results/                                # Output per model
│   ├── CNN_TUMOR/
│   │   ├── accuracy.png
│   │   ├── loss.png
│   │   ├── history.json
│   │   └── classification_report.txt
│   └── ResNet152_TUMOR/
│       ├── accuracy.png
│       ├── loss.png
│       ├── history.json
│       └── classification_report.txt
│
├── src/                                    # Core codebase
│   ├── config.py
│   ├── custom_data.py
│   ├── eval_and_plots.py
│   ├── load_dataset.py
│   ├── models.py
│   └── train.py
│
└── README.md




