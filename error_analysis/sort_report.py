import pandas as pd

# Load CSV
df = pd.read_csv("classification_report_with_category.csv")

# Convert percentage strings to float for sorting
score_columns = ["Precision", "Recall", "F1_score"]

for col in score_columns:
    df[col] = (
        df[col]
        .str.replace("%", "", regex=False)
        .astype(float)
    )

# Category mappings
categoryMappings = {
    "countries": {"start": 0, "end": 11, "name": "Countries"},
    "landmarks": {"start": 12, "end": 48, "name": "Special locations"},
    "numbers": {"start": 49, "end": 59, "name": "Numbers"},
    "alphabet": {"start": 60, "end": 108, "name": "Thai Alphabet & Vowels"},
    "family": {"start": 109, "end": 136, "name": "Family & Relationships"},
    "genders": {"start": 137, "end": 145, "name": "Genders"},
    "sports": {"start": 154, "end": 171, "name": "Sports"},
    "groceries": {"start": 172, "end": 183, "name": "Groceries"}
}

# Assign category based on ID
def get_category(id_value):
    for _, info in categoryMappings.items():
        if info["start"] <= id_value <= info["end"]:
            return info["name"]
    return "Unknown"

df["CategoryGroup"] = df["ID"].apply(get_category)

# Sort:
# 1. Group same categories together
# 2. Sort inside each category by F1 score descending
sorted_df = df.sort_values(
    by=["CategoryGroup", "F1_score"],
    ascending=[True, False]
)

# Convert scores back to percentage strings
for col in score_columns:
    sorted_df[col] = sorted_df[col].round(2).astype(str) + "%"

# Save result
sorted_df.to_csv("classification_scores_sorted.csv", index=False)

print(sorted_df)