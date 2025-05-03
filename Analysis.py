import pandas as pd

# Step 1: Read your CSV file
df = pd.read_csv('BERT_mental_dataset_predictions.csv', sep='|')
df_gpt2 = pd.read_csv('GPT2_mental_dataset_predictions.csv', sep='|')

# Step 2: Generate a cross-tabulation (contingency table)
result = pd.crosstab(df['prediction'], df['label'])
result_gpt2 = pd.crosstab(df_gpt2['prediction'], df_gpt2['label'])

# Step 3: Display the result
print(result)
print(result_gpt2)