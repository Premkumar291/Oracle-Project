import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv("online-retail-dataset.csv")

# Remove unnecessary columns to optimize memory
df.drop(columns=['InvoiceNo', 'StockCode', 'InvoiceDate', 'Country'], inplace=True)

# Clean and optimize data
df.dropna(subset=["CustomerID"], inplace=True)
df["CustomerID"] = df["CustomerID"].astype("category")
df["Quantity"] = df["Quantity"].astype("int16")

# Sample a fraction of data (20%) for better rule generation
df_sampled = df.sample(frac=0.2, random_state=42)

# Create a basket matrix for association rule mining
basket = df_sampled.groupby(["CustomerID", "Description"], observed=False)["Quantity"].sum().unstack().fillna(0)

# Convert basket data into binary format
basket = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)

# Apply Apriori algorithm with lower min_support
frequent_itemsets = apriori(basket, min_support=0.005, use_colnames=True)

# Generate rules with lower lift threshold
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

# Sort rules by confidence
rules = rules.sort_values(by="confidence", ascending=False)

# Convert antecedents & consequents to readable strings for visualization
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(map(str, x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(map(str, x)))

# Check if rules exist before plotting
if not rules.empty:
    # Bar Graph of Top 10 Rules
    plt.figure(figsize=(12, 6))
    sns.barplot(data=rules.head(10), x="support", y="antecedents", hue="confidence")
    plt.xlabel("Support")
    plt.ylabel("Antecedents (Top 10 Rules)")
    plt.title("Top 10 Association Rules")
    plt.xticks(rotation=45)
    plt.legend(title="Confidence")
    plt.show()

    # Scatter Plot of Lift vs Confidence
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=rules, x="confidence", y="lift", hue="confidence", size="support", palette="coolwarm", legend=True)
    plt.xlabel("Confidence")
    plt.ylabel("Lift")
    plt.title("Scatter Plot of Association Rules")
    plt.legend(title="Support")
    plt.show()
else:
    print("No association rules were generated. Try lowering min_support further.")