import re
import pandas as pd

# category mapping
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

# function to map ID -> category
def get_category(class_id):
    for category in categoryMappings.values():
        if category["start"] <= class_id <= category["end"]:
            return category["name"]
    return "Unknown"

# input/output files
input_txt = "classification_report.txt"
output_csv = "classification_report_with_category.csv"

# regex pattern
pattern = re.compile(
    r"^\s*(\d+)\s+(.+?)\s+([\d.]+%)\s+([\d.]+%)\s+([\d.]+%)\s+(\d+)\s*$"
)

rows = []

# read txt
with open(input_txt, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.match(line)

        if match:
            class_id = int(match.group(1))

            rows.append({
                "ID": class_id,
                "Gloss": match.group(2).strip(),
                "Category": get_category(class_id),
                "Precision": match.group(3),
                "Recall": match.group(4),
                "F1_score": match.group(5),
                "Support": int(match.group(6))
            })

# convert to dataframe
df = pd.DataFrame(rows)

# save csv
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"Saved to: {output_csv}")
print(df.head())