import pandas as pd

# ---- 1. Load the file ----
input_file = "..\\computing_c_scores\\output_file(s)\\Volume_and_surface_scores_with_C_scores.csv"
df = pd.read_csv(input_file)

# ---- 2. Identify numeric columns ----
numeric_columns = df.select_dtypes(include=["number"]).columns

# ---- 3. Create rounded columns ----
for col in numeric_columns:
    new_col_name = f"{col}_rounded_second_decimal"
    df[new_col_name] = df[col].round(2)

# ---- 4. Save updated file ----
output_file = "Volume_and_surface_scores_with_C_scores_with_rounded_columns.csv"
df.to_csv(output_file, index=False)

print("New file saved as:", output_file)