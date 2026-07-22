# Credit Card Fraud Detection

## Project Overview

This project develops and compares machine learning models for detecting fraudulent credit card transactions. The primary challenge is the extreme class imbalance in the dataset, where fraudulent transactions represent fewer than 0.2% of all observations.

The analysis compares traditional classification methods, ensemble models, and neural networks. It also uses Monte Carlo cross-validation to evaluate how consistently selected models perform across repeated stratified samples.

## Dataset

The project uses the Credit Card Fraud Detection dataset containing European cardholder transactions.

* **Transactions:** 284,807
* **Fraudulent transactions:** 492
* **Non-fraudulent transactions:** 284,315
* **Fraud rate:** Approximately 0.17%
* **Target variable:** `Class`

  * `0` = Non-fraudulent transaction
  * `1` = Fraudulent transaction

The dataset contains:

* `Time`: Seconds elapsed since the first transaction
* `Amount`: Transaction amount
* `V1` through `V28`: Anonymized features produced through principal component analysis
* `Class`: Fraud indicator

The dataset is not included in this repository because of its size. Download `creditcard.csv` from the Kaggle Credit Card Fraud Detection dataset and update the file path in the notebook before running it.

## Objectives

* Explore the distribution of fraudulent and non-fraudulent transactions
* Train multiple classification models
* Address severe class imbalance through class weighting and SMOTE
* Tune classification thresholds using validation-set F1 score
* Compare models using metrics appropriate for imbalanced classification
* Evaluate model stability through Monte Carlo cross-validation
* Identify a model that captures fraud while limiting false-positive alerts

## Methods

### Data Preparation

The data was divided into training, validation, and test sets.

Standardization was applied for models that are sensitive to feature scale, including:

* Logistic Regression
* K-Nearest Neighbors
* Neural Networks

SMOTE was also prepared as an oversampling strategy for the minority fraud class.

### Exploratory Data Analysis

The exploratory analysis included:

* Fraud and non-fraud transaction counts
* Transaction amount distribution
* Transaction time distribution
* Correlation heatmap across numerical features
* Examination of the dataset's class imbalance

### Models Evaluated

The initial model comparison included:

* Linear Discriminant Analysis
* Quadratic Discriminant Analysis
* Logistic Regression
* K-Nearest Neighbors
* Random Forest
* Gradient Boosting
* Scikit-learn Multilayer Perceptron
* PyTorch Neural Network

For models that generate probabilities, decision thresholds were tuned on the validation set to maximize F1 score.

A second evaluation used 100 iterations of stratified Monte Carlo cross-validation for:

* Logistic Regression
* XGBoost
* LightGBM

## Evaluation Metrics

Accuracy alone can be misleading for highly imbalanced datasets. Models were evaluated using:

* **Accuracy:** Overall proportion of correct predictions
* **Precision:** Proportion of flagged transactions that were actually fraudulent
* **Recall:** Proportion of fraudulent transactions successfully detected
* **F1 Score:** Balance between precision and recall
* **Brier Score:** Mean squared error of predicted probabilities, where lower values indicate better probability calibration

## Initial Test-Set Results

| Model                       | Accuracy | Precision | Recall |   F1 Score |  Brier Score |
| --------------------------- | -------: | --------: | -----: | ---------: | -----------: |
| LDA                         |   0.9994 |    0.8690 | 0.7449 |     0.8022 |     0.000632 |
| QDA                         |   0.9832 |    0.0834 | 0.8776 |     0.1523 |     0.016801 |
| Logistic Regression         |   0.9990 |    0.6587 | 0.8469 |     0.7411 |     0.001018 |
| KNN                         |   0.9995 |    0.9367 | 0.7551 |     0.8362 |     0.000509 |
| Random Forest               |   0.9996 |    0.9213 | 0.8367 | **0.8770** | **0.000404** |
| Gradient Boosting           |   0.9985 |    0.6429 | 0.2755 |     0.3857 |     0.001510 |
| Scikit-learn Neural Network |   0.9994 |    0.8144 | 0.8061 |     0.8103 |     0.000650 |
| PyTorch Neural Network      |   0.9993 |    0.8315 | 0.7551 |     0.7914 |     0.000685 |

Random Forest produced the strongest initial test-set F1 score while maintaining high precision and recall.

## Monte Carlo Cross-Validation Results

The selected models were evaluated across 100 stratified train-test splits.

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |  Brier Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: | -----------: |
| Logistic Regression |     0.9772 |     0.0650 | **0.9069** |     0.1213 |     0.022588 |
| XGBoost             |     0.9983 |     0.5098 |     0.8577 |     0.6375 |     0.001741 |
| LightGBM            | **0.9996** | **0.9130** |     0.8182 | **0.8625** | **0.000416** |

## Key Findings

* Logistic Regression detected the highest proportion of fraudulent transactions, but its low precision generated many false-positive alerts.
* XGBoost provided a stronger balance between fraud detection and false-positive control.
* LightGBM achieved the best overall Monte Carlo performance.
* LightGBM produced an average F1 score of approximately **0.863**, precision of **0.913**, and recall of **0.818**.
* Random Forest performed best during the initial train-validation-test evaluation.
* Threshold tuning substantially improved several models compared with using the default classification threshold.
* Accuracy was not sufficient for comparing models because a model could achieve more than 99% accuracy by primarily predicting the non-fraud class.

## Conclusion

LightGBM was the strongest model in the repeated Monte Carlo evaluation because it maintained high precision while detecting more than 81% of fraudulent transactions. This balance is valuable in fraud detection because excessive false-positive alerts can increase investigation costs and inconvenience legitimate customers.

The results also demonstrate that the best model depends on the business objective. A company prioritizing maximum fraud capture may accept the lower precision of Logistic Regression, while a company seeking a more balanced alert system may prefer LightGBM or Random Forest.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* XGBoost
* LightGBM
* PyTorch
* TensorFlow
* Google Colab

## Repository Structure

```text
credit-card-fraud-detection/
├── README.md
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/rwagner03/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Place `creditcard.csv` in a local `data` directory:

```text
credit-card-fraud-detection/
└── data/
    └── creditcard.csv
```

Update the notebook's dataset path to:

```python
df = pd.read_csv("data/creditcard.csv")
```

Then open the notebook:

```bash
jupyter notebook notebooks/credit_card_fraud_detection.ipynb
```

## Future Improvements

* Move all preprocessing into reusable scikit-learn pipelines
* Prevent possible data leakage by fitting scalers separately within each cross-validation training split
* Tune model hyperparameters using randomized or Bayesian optimization
* Tune decision thresholds within each Monte Carlo training iteration
* Compare precision-recall curves and PR-AUC scores
* Analyze the financial cost of false positives and false negatives
* Add SHAP-based model interpretation
* Save the best model for deployment
* Build an interactive dashboard or fraud-scoring application

## Author

**Russell Wagner**
B.S. in Statistics, Minor in Mathematics
Texas A&M University
