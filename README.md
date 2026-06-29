# Credit Card Fraud Detection

Machine learning project for detecting fraudulent financial transactions using feature engineering and multiple classification models.

The project includes:

* Data exploration
* Data preprocessing
* Feature engineering
* Model comparison
* Performance evaluation

## Dataset

The project uses a publicly available fraud detection dataset containing transaction information and fraud labels.
Kaggle: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data

Target variable:

* `is_fraud`

Technologies: Python, Pandas, NumPy, Scikit-Learn, Matplotlib

## Workflow

1. Load dataset
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Feature engineering
5. Statistics showcase
6. Train multiple ML models
7. Compare model performance
8. Evaluate the final model

## Feature Engineering

Examples of new several created features, which improved model performace:

* Hour of transaction
* Night transaction (`is_night`)
* Age
* Transaction amount
* Gender

## Machine Learning Models

The project compares several classification algorithms, including:

* Logistic Regression
* Random Forest
* XGBoost

## Evaluation Metrics

Performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC, PR-AUC
* GricSearchCV

Special attention was given to Recall and Precision because fraud detection datasets are highly imbalanced.

## Key Findings

* Feature engineering improved classification performance.
* Time-based features such as `is_night` provided additional predictive information.
* Different machine learning models showed different trade-offs between precision and recall.
* Class imbalance requires evaluating more than overall accuracy.

## Future Improvements

* Hyperparameter optimization
* SMOTE or other imbalance handling techniques
* Model deployment

## Author Erick
Data analysis and machine learning portfolio project
Focus areas:

-Feature engineering
-Fraud detection
-Data analysis
-Machine learning