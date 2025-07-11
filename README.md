# Credit-Card-Fraud-Detection-Project
<h5>Done by <b>Ethan Alexander Gounder</b></h5>
A machine learning project that uses logistic regression to detect fraudulent credit card transactions. This project demonstrates data preprocessing, exploratory data analysis, handling imbalanced datasets, and model evaluation techniques.
## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Key Insights](#key-insights)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)

## 🎯 Overview

Credit card fraud detection is a critical application of machine learning in the financial sector. This project implements a binary classification model to distinguish between legitimate and fraudulent transactions using various transaction features.

**Key Objectives:**
- Detect fraudulent credit card transactions with high accuracy
- Handle severely imbalanced dataset using undersampling techniques
- Provide comprehensive data analysis and visualization
- Evaluate model performance using multiple metrics

## 📊 Dataset

The project uses a credit card transactions dataset containing:
- **Total Transactions:** ~284,807 transactions
- **Features:** 30 numerical features (V1-V28 are PCA-transformed, plus Time and Amount)
- **Target Variable:** Class (0 = Legitimate, 1 = Fraudulent)
- **Imbalance Ratio:** Highly imbalanced with only 0.17% fraudulent transactions
- Download the dataset here: [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)

### Data Characteristics
- **Legitimate Transactions:** 284,315 (99.83%)
- **Fraudulent Transactions:** 492 (0.17%)
- **No Missing Values:** Complete dataset with no null values

## 🔧 Features

### Data Analysis Features
- **Exploratory Data Analysis (EDA)** with comprehensive visualizations
- **Class Distribution Analysis** using pie charts and histograms
- **Statistical Summary** for both transaction types
- **Correlation Matrix** heatmap for feature relationships
- **PCA Visualization** for 2D data projection

### Model Features
- **Imbalanced Data Handling** using undersampling technique
- **Logistic Regression** classifier with optimized parameters
- **Train-Test Split** with stratified sampling
- **Performance Evaluation** using multiple metrics

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
```

## 💻 Usage

### 1. Data Loading
```python
# Load the dataset
credit_card_data = pd.read_csv('path/to/creditcard.csv')
```

### 2. Data Exploration
```python
# Basic dataset information
credit_card_data.info()
credit_card_data.head()

# Class distribution
credit_card_data['Class'].value_counts()
```

### 3. Data Preprocessing
```python
# Separate legitimate and fraudulent transactions
legit = credit_card_data[credit_card_data['Class'] == 0]
fraud = credit_card_data[credit_card_data['Class'] == 1]

# Handle imbalanced data using undersampling
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
```

### 4. Model Training
```python
# Prepare features and target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)
```

### 5. Model Evaluation
```python
# Make predictions
X_test_prediction = model.predict(X_test)

# Calculate accuracy
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Test Accuracy: {test_accuracy}")
```

## 📈 Model Performance

### Accuracy Metrics
- **Training Accuracy:** ~94-96%
- **Test Accuracy:** ~91-93%
- **Model Type:** Logistic Regression with max_iter=10000

### Evaluation Methods
- **Confusion Matrix** visualization
- **Feature Importance** analysis
- **Cross-validation** with stratified sampling

## 📊 Visualizations

The project includes several comprehensive visualizations:

### 1. Class Distribution
- **Pie Chart** showing the imbalance between legitimate and fraudulent transactions
- **Histogram** comparing transaction amounts by class

### 2. Feature Analysis
- **KDE Plots** for key features (V14, V12, V10, V16, V17)
- **Correlation Matrix** heatmap showing feature relationships

### 3. Advanced Visualizations
- **PCA 2D Projection** for dimensionality reduction visualization
- **Confusion Matrix** heatmap for model performance
- **Feature Importance** bar chart for top 10 predictive features

### 4. Statistical Comparisons
- **Descriptive Statistics** comparison between fraud and legitimate transactions
- **Group-wise Mean** analysis by transaction class

## 🔍 Key Insights

### Data Insights
- Fraudulent transactions tend to have different statistical distributions compared to legitimate ones
- Certain PCA components (V14, V12, V10, V16, V17) show significant differences between classes
- Transaction amounts vary considerably between fraud and legitimate transactions

### Model Insights
- Logistic regression performs well on this balanced dataset after undersampling
- Feature importance analysis reveals which transformed features are most predictive
- The model achieves good performance despite the complexity of fraud detection

## 🛠 Technical Details

### Data Preprocessing
- **Undersampling Strategy:** Reduced legitimate transactions to match fraudulent ones (492 each)
- **Stratified Splitting:** Maintains class distribution in train-test split
- **Feature Scaling:** Not explicitly applied (may be beneficial for future improvements)

### Model Configuration
- **Algorithm:** Logistic Regression
- **Max Iterations:** 10,000 (prevents convergence issues)
- **Random State:** 2 (for reproducibility)
- **Test Size:** 20% of the balanced dataset

### Evaluation Metrics
- **Primary Metric:** Accuracy Score
- **Additional Metrics:** Confusion Matrix
- **Visualization:** Multiple performance and data visualization charts

## 🚀 Future Improvements

### Model Enhancements
- **Advanced Algorithms:** Try Random Forest, XGBoost, or Neural Networks
- **Feature Engineering:** Create additional meaningful features
- **Hyperparameter Tuning:** Optimize model parameters using GridSearchCV
- **Cross-Validation:** Implement k-fold cross-validation for robust evaluation

### Data Handling
- **Alternative Sampling:** Try SMOTE for oversampling instead of undersampling
- **Feature Scaling:** Apply StandardScaler or MinMaxScaler
- **Outlier Detection:** Identify and handle outliers in the dataset

### Evaluation Metrics
- **Precision/Recall:** Important for imbalanced classification
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the ROC curve
- **Precision-Recall Curve:** Better for imbalanced datasets

### Production Considerations
- **Real-time Prediction:** Implement streaming prediction capabilities
- **Model Monitoring:** Track model performance over time
- **A/B Testing:** Compare different model versions
- **Explainability:** Add SHAP or LIME for model interpretation

---

## 📝 Notes

- This project demonstrates a complete machine learning pipeline from data exploration to model evaluation
- The undersampling approach significantly reduces the dataset size but helps with class balance
- Results may vary due to the random sampling in the undersampling process
- Consider the trade-offs between false positives and false negatives in fraud detection scenarios

## 🤝 Contributing

Feel free to contribute to this project by:
- Implementing additional algorithms
- Adding more sophisticated evaluation metrics
- Improving data visualization
- Optimizing model performance
- Adding real-world deployment considerations