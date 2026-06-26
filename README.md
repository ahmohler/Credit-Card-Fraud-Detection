# Credit Card Fraud Detection Engine

This project focuses on building a machine learning system capable of detecting fraudulent credit card transactions with high precision, specifically addressing the extreme class imbalance commonly found in financial data.

## The Business Problem
Financial institutions face a delicate dual challenge every day:
* **Direct Losses:** Every undetected fraudulent transaction leads to direct financial loss and regulatory/chargeback costs for the bank.
* **Customer Friction:** If the security system is overly sensitive, it will flag and block legitimate transactions from real customers, leading to customer frustration and increased support overhead.
* **The Data Imbalance Obstacle:** Fraudulent activities make up only 0.17% of daily volume, making traditional rule-based systems virtually blind to subtle, evolving fraud patterns.

## The Proposed Solution
Developing a balanced, cost-sensitive machine learning framework that handles severe data imbalance (using techniques like SMOTE) and optimizes the decision thresholds to perfectly balance security risk against seamless customer experience.

## Dataset Profile
* **Source:** The dataset captures real European cardholders' transactions from September 2013 over a two-day period.
* **Dataset Link:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Volume:** 284,807 total transactions, with only 492 instances of fraud (0.173%).
* **Features:** Columns $V_1$ through $V_{28}$ are PCA-transformed features used to protect user privacy. `Time` and `Amount` are the only un-anonymized features.

---

## Exploratory Data Analysis (EDA) Insights

We conducted a comprehensive statistical and correlation analysis on the dataset, revealing critical behavioral patterns of fraudulent operations:

### 1. Class Distribution
* **Genuine Transactions (Class 0):** 284,315 (99.827%)
* **Fraudulent Transactions (Class 1):** 492 (0.173%)
* *Note: This extreme imbalance requires stratified sampling and advanced resampling techniques to ensure the model doesn't overfit to the majority class.*

### 2. Transaction Amount Profiling
Statistical analysis of transaction volumes exposed distinct spending behaviors between normal users and attackers:
* **The Micro-Transaction Strategy:** While genuine transactions have a lower mean ($88.29) compared to fraud ($122.21), the **median fraud transaction is only $9.25** (compared to $22.00 for genuine ones). Furthermore, 25% of all fraud is **$1.00 or less**. This mathematically confirms that fraudsters frequently use low-value "test transactions" to check card validity without triggering basic rule-based alerts.
* **Cap Limits:** Genuine transactions peak at an outlier max of **$25,691.16**, whereas fraudulent attempts completely drop off after **$2,125.87**.

### 3. Feature Latent Signals (Correlation Top Drivers)
Since $V_1$ through $V_{28}$ are orthogonal PCA components, they exhibit zero multicollinearity. Their direct linear correlations with the target `Class` highlighted the strongest features for predictive modeling:
* **Top Strongest Negative Correlations (Inverse Risk Drivers):** * $V_{17}$ (-0.33)
  * $V_{14}$ (-0.30)
  * $V_{12}$ (-0.26)
  * *Insight: As the values of these features decrease significantly, the mathematical probability of fraud spikes.*
* **Top Strongest Positive Correlations (Direct Risk Drivers):**
  * $V_{11}$ (+0.15)
  * $V_{4}$ (+0.13)