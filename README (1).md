# Normalizing Flows for Data Generation and Credit Card Fraud Detection

## Overview

This project combines two powerful concepts: **Normalizing Flows** for data generation and **Credit Card Fraud Detection** using machine learning techniques. The primary objectives are to generate synthetic data using Normalizing Flows and utilize this data to enhance the performance of fraud detection models on real-world credit card transaction datasets.

## Dataset

### Credit Card Dataset

The credit card dataset consists of various features relevant to transactions, including:

- **scaled_amount**: The transaction amount, scaled for normalization.
- **scaled_time**: The time of the transaction, scaled for normalization.
- **V1 to V28**: Anonymized features representing various characteristics of the transactions.
- **Class**: The target variable, where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.

### Generated Data

Synthetic data is generated using Normalizing Flows, which can be used to augment the existing dataset or to test the robustness of fraud detection models.

## Installation

To run this project, you need to install the following Python libraries. You can install them using pip:
- **Numpy**
- **Pandas**
- **MatPlotLib**
- **Seaborn**
- **Pytorch**
- **Jupyter**
- **Scikit-Learn**


## Normalizing Flows

Normalizing Flows are used to model complex distributions by transforming a simple distribution into a more complex one. The following steps are involved:

1. **Data Preparation**: Load and preprocess the data.
2. **Flow Model Definition**: Define a Normalizing Flow model for generating synthetic data.
3. **Training the Flow Model**: Train the model on the available dataset.
4. **Data Generation**: Generate synthetic samples and save them for further analysis.

## Credit Card Fraud Detection

The project implements various machine learning models to detect fraudulent transactions, including:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Gradient Boosting**

### Steps Involved:

1. **Data Loading**: Load the original and generated datasets.
2. **Data Preprocessing**: Clean and prepare the data for modeling.
3. **Model Training**: Train different models on the combined dataset.
4. **Evaluation**: Evaluate the models based on accuracy, precision, recall, and F1-score.

## Results

The results showcase the effectiveness of using synthetic data generated from Normalizing Flows in improving the performance of fraud detection models. Key performance metrics are documented for comparison.

## Future Work

Future enhancements may include:

- Exploring advanced Normalizing Flow architectures for better data generation.
- Hyperparameter tuning for fraud detection models.
- Implementing additional anomaly detection techniques.

## Acknowledgments

- **Kaggle** for providing the original credit card dataset.
- The **scikit-learn** and **PyTorch** libraries for machine learning functionalities.
- **Matplotlib** and **Seaborn** for data visualization.
