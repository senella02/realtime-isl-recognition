# data_preprocess

Normalizes raw landmark CSV files (hand + body pose) so they are ready for model training.

---

## Setup

1. Copy this entire `data_preprocess` folder to your project.
2. Install the dependency:
   ```
   pip install -r data_preprocess/requirements.txt
   ```

---

## Usage in Jupyter Notebook

```python
import sys
sys.path.insert(0, "path/to/data_preprocess")  # point to the folder

from main import preprocess

preprocess("train_raw.csv", "train_normalized.csv")
preprocess("test_raw.csv",  "test_normalized.csv")
```

The function reads your raw CSV, normalizes all hand and body landmarks, and saves the result to the output path you specify.

---

## Input CSV format

Your CSV must have:
- A `labels` column containing the class label for each row.
- One column per landmark (e.g. `wrist_left_X`, `wrist_left_Y`, ...), where each cell is a **list** of frame values.

---

## Output

A new CSV file at the path you specified, with the same columns but normalized coordinate values.
