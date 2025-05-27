import pandas as pd

# Load your CSV
df = pd.read_csv(r"C:\Users\Admin\AI-Essentials\Mod3\Nigerian_Fraud_processed_20250527_042834.csv", encoding='utf-8')

# Define important text columns for analysis
text_columns = ['subject', 'body', 'urls']  # Modify as per actual columns needed for NLP

# Clean each column
for col in text_columns:
    df[col] = df[col].astype(str)
    df[col] = df[col].replace(["", "nan", "NaN", "None"], pd.NA)
    df = df[df[col].notna()]
    df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True).str.strip()
    df = df[df[col].str.len() > 0]

# Reset index
df.reset_index(drop=True, inplace=True)

# Limit to 10 rows
df = df.head(10)

# Save cleaned CSV
output_path = r"C:\Users\Admin\AI-Essentials\Mod6\Nigerian_Fraud_processed_email_10.csv"
df.to_csv(output_path, index=False)

print(f"✅ Cleaned and saved {len(df)} rows to:\n{output_path}")