## 🩺 Chest X-ray Pneumonia Detection using Deep Learning

This project leverages deep learning techniques to detect pneumonia from chest X-ray images. It is part of the broader **Medical Image Analysis** repository and aims to assist healthcare professionals by providing an automated and reliable diagnostic aid. The project explores data preprocessing, visualization, and model development using convolutional neural networks (CNNs), enabling early detection and improving diagnostic efficiency.

### Key Goals:
- 📊 Load, visualize, and explore the dataset  
- 🧼 Preprocess X-ray images for model readiness  
- 🧠 Train and evaluate deep learning models for binary classification (Pneumonia vs. Normal)  
- 🧪 Plan for future improvements including model deployment and performance tuning
## 🗂️ Project Structure

```
ChestXray_Pneumonia_Detection/
├── README.md                       ← Project-specific documentation
├── data/                           ← Dataset folder (excluded from GitHub)
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
├── notebooks/
│   └── 01_data_loading.ipynb       ← Data loading & visualization notebook
├── models/                         ← (Empty) Model files will be saved here
├── results/                        ← (Empty) Visual results, metrics, graphs
```
## Dataset Information

The dataset used in this project is the **Chest X-Ray Images (Pneumonia)** dataset, which contains chest radiographs categorized into three subsets: training, validation, and testing. Each subset includes two classes — **NORMAL** and **PNEUMONIA**.

This dataset was originally curated by **Kermany et al. (2018)** for pediatric pneumonia diagnosis and later made publicly available on Kaggle by **Paul Mooney**.

- 📄 Original Source: *Kermany, Daniel S., et al. "Identifying medical diagnoses and treatable diseases by image-based deep learning." Cell 172.5 (2018): 1122-1131.*
- 📁 Kaggle Repository: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

> Note: For ethical use and citation, please refer to the license or publication guidelines provided by the dataset authors.
## 💻 How to Run

Follow the steps below to set up and run the project locally:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/medical-image-analysis.git
   cd medical-image-analysis/ChestXray_Pneumonia_Detection
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv medai
   source medai/bin/activate      # For Linux/macOS
   medai\Scripts\activate         # For Windows
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset**

   - This project uses the **Chest X-Ray Images (Pneumonia)** dataset available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
   - Extract the dataset into the following folder structure:

     ```
     ChestXray_Pneumonia_Detection/
     └── data/
         └── chest_xray/
             ├── train/
             ├── test/
             └── val/
     ```

5. **Run the Notebook**

   Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Open and execute:  
   `notebooks/01_data_loading.ipynb`
## Dependencies

This project uses the following Python packages:

- numpy
- pandas
- matplotlib
- seaborn
- opencv-python
- scikit-learn
- tqdm
- jupyter
- torch
- torchvision
- kaggle

It is recommended to use a virtual environment to manage these dependencies.

To install all packages, run:

```bash
pip install -r requirements.txt
```
The requirements.txt file contains the exact package versions used in this project.
## Notebooks Explained

This project contains several Jupyter notebooks, each focusing on different aspects of the Chest X-ray Pneumonia Detection pipeline:

- **Data_Preprocessing.ipynb**: Data cleaning, augmentation, and preparation steps.
- **Model_Training.ipynb**: Building, training, and validating deep learning models.
- **Model_Evaluation.ipynb**: Performance metrics, visualization of results, and model comparison.
- **Inference.ipynb**: Demonstration of running the trained model on new chest X-ray images for pneumonia detection.

Each notebook is designed to be run sequentially, but can also be used independently for specific tasks.
## Future Work

- Enhance model performance through hyperparameter tuning and advanced architectures.
- Expand the dataset with more diverse and annotated chest X-ray images.
- Develop a user-friendly interface for easier deployment and real-time inference.
- Integrate explainability methods to improve model interpretability and trust.
- Explore multi-modal data integration to improve diagnostic accuracy.

## Credits / License

- Dataset provided by [Paul Mooney](https://github.com/ieee8023/covid-chestxray-dataset).
- Developed by HammadShas.
- This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
