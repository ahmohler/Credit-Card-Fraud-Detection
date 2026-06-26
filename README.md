# Credit Card Fraud Detection Engine

This project focuses on building a machine learning system capable of detecting fraudulent credit card transactions with high precision, specifically addressing the extreme class imbalance commonly found in financial data.

## The Business Problem
Financial institutions face a delicate dual challenge every day:
1. **Direct Losses:** Every undetected fraudulent transaction leads to direct financial loss and regulatory/chargeback costs for the bank.
2. **Customer Friction:** If the security system is overly sensitive, it will flag and block legitimate transactions from real customers, leading to customer frustration and increased support overhead.
3. **The Data Imbalance Obstacle:** Fraudulent activities make up only 0.17% of daily volume, making traditional rule-based systems virtually blind to subtle, evolving fraud patterns.

## The Proposed Solution
Developing a balanced, cost-sensitive machine learning framework that handles severe data imbalance (using techniques like SMOTE) and optimizes the decision thresholds to perfectly balance security risk against seamless customer experience.

## Dataset Insights
* **Source:** The dataset captures real European cardholders' transactions from September 2013 over a two-day period.
* **Dataset Link:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Volume:** 284,807 total transactions, with only 492 instances of fraud (0.172%).
* **Features:** Columns V1 through V28 are PCA-transformed features used to protect user privacy. Time and Amount are the only un-anonymized features.