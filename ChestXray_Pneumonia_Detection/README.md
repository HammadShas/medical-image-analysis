# Chest X-ray Pneumonia Detection 

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Dataset](https://img.shields.io/badge/Dataset-ChestX--ray--Pneumonia-brightgreen)

> A deep learning–based medical imaging project for detecting **Pneumonia** in **Chest X-rays** using **PyTorch**, **ResNet-50**, and transfer learning. Developed as part of the broader **Medical Image Analysis** repository, this project aims to support healthcare professionals by offering an automated and reliable diagnostic tool. To ensure balanced training and improve generalization, it integrates two publicly available datasets. The overall objective is to facilitate early and accurate detection of pneumonia through scalable and interpretable AI models.

---

## Objectives

- Explore and visualize the Chest X-ray dataset.
- Balance the dataset by augmenting normal samples from another source.
- Preprocess X-ray images using `torchvision.transforms`.
- Train a deep convolutional model (ResNet-50) using transfer learning.
- Evaluate model performance with standard classification metrics.
- Visualize loss curves, confusion matrix, and predictions.

---

## Project Structure

```
ChestXray_Pneumonia_Detection/
├── data/                          # Combined dataset (train/test/val)
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
├── notebooks/                     # Jupyter Notebooks (modular workflow)
│   ├── 01_data_loading.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluate_model.ipynb
├── scripts/                       # (Optional) Python script versions
│   ├── data_loading.py
│   └── data_preprocessing.py
│   └── model_training.py
├── results/                       # Metrics, confusion matrix, and plots
├── outputs/                       # Saved model checkpoints, logs
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

---

## Datasets Used

### 1. Chest X-Ray Images (Pneumonia)
- **Source:** [Kaggle - Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** `NORMAL`, `PNEUMONIA`
- **Structure:** Images organized into `train/`, `test/`, `val/` folders

### 2. TB Chest X-ray Dataset (for Normal Samples Only)
- **Source:** [Kaggle - Tawsifur Rahman](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- **Usage:** Only the `NORMAL` class images from this dataset were used to balance the underrepresented class.

> Note: Ensure ethical usage by referring to licensing and citations provided by the original dataset authors.

### Dataset Citation

The primary dataset used in this project was originally curated by **Kermany et al. (2018)** for pediatric pneumonia diagnosis, and later made publicly available on Kaggle by **Paul Mooney**.

> *Kermany, Daniel S., et al. ["Identifying medical diagnoses and treatable diseases by image-based deep learning."](https://doi.org/10.1016/j.cell.2018.02.010)*  
> *Cell 172.5 (2018): 1122–1131.*

Please ensure appropriate citation and adherence to dataset licensing when using this resource in derivative works or publications.

---

## Model Details

- **Architecture:** ResNet-50 (pretrained on ImageNet)
- **Technique:** Transfer Learning
- **Loss Function:** Binary Cross Entropy Loss
- **Optimizer:** Adam
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## How to Run

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/medical-image-analysis.git
    cd medical-image-analysis/ChestXray_Pneumonia_Detection
    ```

2. **Create Virtual Environment (Recommended)**

    ```bash
    conda create -n medai python=3.10
    conda activate medai
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download and Prepare Datasets**

    - [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    - [TB Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) – *Only `NORMAL` images used*

    Organize as:

    ```
    data/
    └── chest_xray/
        ├── train/
        ├── test/
        └── val/
    ```

5. **Launch Jupyter Notebooks**

    ```bash
    jupyter notebook
    ```

    Run notebooks step-by-step in this order:
    - `01_data_loading.ipynb`
    - `02_data_preprocessing.ipynb`
    - `03_model_training.ipynb`
    - `04_evaluate_model.ipynb`

---

## Sample Outputs

- **Classification Report:**

    ```
    Class         Precision     Recall   F1-Score
    --------------------------------------------
    Normal            98.01      92.06      94.94
    Pneumonia         92.51      98.13      95.24
    --------------------------------------------
    Macro Avg         95.26      95.09      95.09
    Weighted Avg      95.26      95.09      95.09
    ```

- **Confusion Matrix:**
    - Raw counts and class-wise percentages visualized using heatmaps.
    - High true positive and true negative rates observed.

- **ROC Curve:**
    - **AUC Score:** `0.9884`
    - Clear separation between classes with minimal false positives.

---

## Results

| Metric         | Value     |
|----------------|-----------|
| **Accuracy**   | 95.09%    |
| **Precision**  | 92.51–98.01% |
| **Recall**     | 92.06–98.13% |
| **F1-Score**   | 94.94–95.24% |
| **AUC Score**  | 0.9884    |

> These results are based on the final evaluation of the model using the test set. Performance may vary slightly depending on training runs and data augmentations.

---

## Dependencies

- Python 3.10+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- opencv-python
- tqdm
- jupyter
- kaggle

To install all packages:

```bash
pip install -r requirements.txt
```

---

## Future Work

- Integrate more complex architectures like DenseNet or EfficientNet.
- Add Grad-CAM for explainable AI (XAI).
- Deploy as a web app using Streamlit or Flask.
- Expand dataset diversity (age groups, imaging conditions).
- Perform multi-label classification (e.g., TB + Pneumonia detection).

---

## Acknowledgements

- [Kaggle: Chest X-ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Kaggle: TB Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- ResNet authors & PyTorch community

---

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for more details.

Developed by **HammadShas**
