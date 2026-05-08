import pandas as pd

# Load CSV
df = pd.read_csv("classification_report_with_category.csv")

# Remove % and convert to float
score_columns = ["Precision", "Recall", "F1_score"]

for col in score_columns:
    df[col] = (
        df[col]
        .str.replace("%", "", regex=False)
        .astype(float)
    )

# Your category mappings
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

results = []

# Compute averages by ID range
for key, info in categoryMappings.items():
    start_id = info["start"]
    end_id = info["end"]

    # Filter rows within ID range
    subset = df[(df["ID"] >= start_id) & (df["ID"] <= end_id)]

    # Skip empty categories
    if subset.empty:
        continue

    results.append({
        "Category": info["name"],
        "Avg Precision": subset["Precision"].mean(),
        "Avg Recall": subset["Recall"].mean(),
        "Avg F1": subset["F1_score"].mean(),
        "Total Samples": subset["Support"].sum()
    })

# Convert results to DataFrame
result_df = pd.DataFrame(results)

# Optional: round values
result_df[["Avg Precision", "Avg Recall", "Avg F1"]] = (
    result_df[["Avg Precision", "Avg Recall", "Avg F1"]]
    .round(2)
)

print(result_df)