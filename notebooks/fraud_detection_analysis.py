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
    
    # Fixed warning by assigning x to hue and setting legend=False
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
    
    # Generate boxplot to visualize outliers and distribution differences
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Class', y='Amount', data=df, hue='Class', palette=['#4A90E2', '#D0021B'], legend=False)
    plt.title('Transaction Amount Distribution by Class', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Class (0: Genuine | 1: Fraud)', fontsize=11)
    plt.ylabel('Amount ($)', fontsize=11)
    
    # Limit y-axis slightly to see the box structures clearly due to high outliers
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\nAmount distribution plot saved to {output_path}")

if __name__ == "__main__":
    # Check current working directory to resolve relative pathing dynamically
    if os.path.basename(os.getcwd()) == "notebooks":
        dataset_path = "../data/creditcard.csv"
    else:
        dataset_path = "data/creditcard.csv"
    
    try:
        data = load_data(dataset_path)
        analyze_class_imbalance(data)
        plot_distribution(data)
        analyze_transaction_amounts(data)
        
    except Exception as e:
        print(e)