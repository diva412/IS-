# Intrusion Detection System (IDS) using Machine Learning

## 📌 Project Overview
This project implements a simple Intrusion Detection System (IDS) using a Machine Learning model on the KDD Cup 99 dataset. The goal is to classify network traffic as normal or attack based on selected features.

---

## 📂 Dataset
- **Dataset Used:** KDD Cup 99 (10% subset)
- The dataset contains network traffic data used for intrusion detection.
- It includes various features like protocol type, service, source bytes, destination bytes, etc.

---

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn

---

## 🧠 Methodology

### 1. Data Preprocessing
- Loaded dataset using `fetch_kddcup99`
- Converted byte strings to readable format
- Converted columns to numeric values
- Handled missing values by replacing them with 0

### 2. Feature Selection
- Selected only two features:
  - `src_bytes`
  - `dst_bytes`

### 3. Label Encoding
- Converted attack labels into numerical form using `LabelEncoder`

### 4. Model Training
- Used **Decision Tree Classifier**
- Limited model complexity:
  - `max_depth = 1` (Decision Stump)
- Used only **1% of dataset** for training

### 5. Evaluation
- Measured performance using **accuracy score**

---

## 📊 Output
The program prints:
```
--- IDS Lab Submission ---
Dataset: KDD Cup 99
Accuracy: XX.XX%
```

---

## 🎯 Key Points
- Simple and fast implementation
- Uses minimal features
- Demonstrates basic IDS concept
- Model intentionally kept simple for academic purposes

---

## 🚀 How to Run

1. Install required libraries:
```
pip install pandas scikit-learn
```

2. Run the script:
```
python your_script_name.py
```

---

## 📌 Conclusion
This project shows how machine learning can be applied to detect network intrusions. Even with limited data and a simple model, it provides a basic understanding of IDS systems.

---
