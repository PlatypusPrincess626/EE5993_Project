import pandas as pd

# Step 1: Read your CSV file
df = pd.read_csv('BERT_mental_dataset_predictions.csv', sep='|')
print(df.head())

# Step 2: Generate a cross-tabulation (contingency table)
result = pd.crosstab(df['prediction'], df['label'])

# Step 3: Display the result
print(result)