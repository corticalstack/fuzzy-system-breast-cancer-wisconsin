# 🧠 Fuzzy Inference System for Breast Cancer Classification

A comprehensive implementation of a Fuzzy Inference System (FIS) for breast cancer classification using the Wisconsin Breast Cancer Dataset (WDBC).

## 📋 Description

This repository contains a complete pipeline for breast cancer classification using fuzzy logic and machine learning. The system analyzes the Wisconsin Breast Cancer Dataset to classify tumors as benign or malignant using various fuzzy membership functions and rule-based inference systems.

The implementation includes:
- Data preprocessing and feature selection
- Fuzzy Inference System with multiple membership function configurations
- Comparison with traditional machine learning models
- Comprehensive performance evaluation

## 🔍 Features

- **Dataset Analysis**: Comprehensive analysis of the Wisconsin Breast Cancer Dataset
  - Feature selection using chi-squared and Random Forest Classifier
  - Correlation analysis and heatmap visualization
  - Distribution analysis with KDE plots
  - Linearity analysis using convex hull
  - K-means clustering

- **Fuzzy Inference System**:
  - Multiple membership function configurations (Gaussian, Z-shaped, S-shaped, Triangular, Trapezoidal)
  - Rule-based inference system for classification
  - Five different defuzzification methods (centroid, bisector, mean of maximum, smallest of maximum, largest of maximum)
  - Threshold-based crisp output conversion

- **Machine Learning Comparison**:
  - Logistic Regression
  - Decision Tree Classification
  - Random Forest Classification
  - Neural Network

- **Performance Evaluation**:
  - Accuracy, Sensitivity, Specificity metrics
  - Confusion matrix visualization
  - Comprehensive logging and result analysis

## 🛠️ Setup Guide

### Prerequisites
- Python 3.6+
- Required packages:
  - pandas
  - numpy
  - scikit-fuzzy
  - scikit-learn
  - tensorflow
  - matplotlib
  - seaborn

### Installation
1. Clone this repository
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-fuzzy scikit-learn tensorflow matplotlib seaborn
   ```
3. Download the Wisconsin Breast Cancer Dataset and place it in the `data` directory
   - The dataset should be named `wdbc.data`
   - Create the data directory if it doesn't exist:
     ```bash
     mkdir -p data
     ```

## 📊 Usage

### Dataset Analysis
Run the dataset analysis script to preprocess the data and perform feature selection:
```bash
python dataset_analysis.py
```
This will:
- Load and preprocess the Wisconsin Breast Cancer Dataset
- Perform feature selection
- Generate visualizations in the `plots` directory
- Save the processed dataset to `data/wdbc_selected_cols.csv`

### Fuzzy Inference System
Run the FIS implementation to classify breast cancer samples:
```bash
python fis.py
```
This will:
- Load the preprocessed dataset
- Configure and run multiple FIS tests with different membership functions
- Compare FIS performance with traditional machine learning models
- Save results to `data/predict_log.csv` and `data/predict_log_sum.csv`
- Generate confusion matrix visualizations in the `plots/cm` directory

## 📚 Resources

- [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-Fuzzy Documentation](https://pythonhosted.org/scikit-fuzzy/)
- [Fuzzy Logic in Medical Diagnosis](https://www.sciencedirect.com/science/article/pii/S1877050915034663)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
