import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Loads the credit card fraud dataset from the specified path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Dataset not found at {file_path}")
        
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Total Rows: {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    return df

def analyze_class_imbalance(df):
    """Calculates and displays the distribution of genuine vs fraud transactions."""
    fraud_count = df['Class'].value_counts()
    fraud_percentage = df['Class'].value_counts(normalize=True) * 100
    
    print(f"\nGenuine Transactions (Class 0): {fraud_count[0]:,} ({fraud_percentage[0]:.3f}%)")
    print(f"Fraud Transactions (Class 1): {fraud_count[1]:,} ({fraud_percentage[1]:.3f}%)")
    return fraud_count

def plot_distribution(df, output_path="transaction_distribution.png"):
    """Generates a log-scaled distribution plot and saves it as an image."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 5))
    
    ax = sns.countplot(x='Class', data=df, hue='Class', palette=['#4A90E2', '#D0021B'], legend=False)
    plt.title('Distribution of Transactions (Genuine vs. Fraud)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Class (0: Genuine | 1: Fraud)', fontsize=11)
    plt.ylabel('Count (Log Scale)', fontsize=11)
    ax.set_yscale('log') 
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Distribution plot saved to {output_path}")

def analyze_transaction_amounts(df, output_path="amount_distribution.png"):
    """Analyzes the descriptive statistics of transaction amounts and saves a boxplot."""
    print("\n--- Transaction Amount Statistical Summary ---")
    
    print("\nGenuine Transactions (Class 0):")
    print(df[df['Class'] == 0]['Amount'].describe())
    
    print("\nFraudulent Transactions (Class 1):")
    print(df[df['Class'] == 1]['Amount'].describe())
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Class', y='Amount', data=df, hue='Class', palette=['#4A90E2', '#D0021B'], legend=False)
    plt.title('Transaction Amount Distribution by Class', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Class (0: Genuine | 1: Fraud)', fontsize=11)
    plt.ylabel('Amount ($)', fontsize=11)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\nAmount distribution plot saved to {output_path}")

def analyze_correlations(df, output_path="feature_correlation.png"):
    """Calculates feature correlations with the Class variable and saves a heatmap."""
    print("\n--- Feature Correlation with Target (Class) ---")
    
    corr_matrix = df.corr()
    correlations = corr_matrix['Class'].sort_values(ascending=False)
    
    print("\nTop 5 Positively Correlated Features with Fraud:")
    print(correlations.iloc[1:6])  
    
    print("\nTop 5 Negatively Correlated Features with Fraud:")
    print(correlations.tail(5))
    
    top_pos = correlations.index[1:6]
    top_neg = correlations.tail(5).index
    top_features = list(top_pos) + list(top_neg) + ['Class']
    top_features = list(set(top_features))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title('Correlation Matrix of Highly Correlated Features', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\nCorrelation heatmap saved to {output_path}")

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == "notebooks":
        dataset_path = "../data/creditcard.csv"
    else:
        dataset_path = "data/creditcard.csv"
    
    try:
        data = load_data(dataset_path)
        analyze_class_imbalance(data)
        plot_distribution(data)
        analyze_transaction_amounts(data)
        analyze_correlations(data)
        
    except Exception as e:
        print(e)