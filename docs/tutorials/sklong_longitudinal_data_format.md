---
---

## 📊 Sklong Longitudinal Data Format

!!! important "Dataset Used in Tutorials"
    The tutorials use a synthetic dataset mimicking health-related longitudinal data. It's generated for illustrative
    purposes and does not represent real-world data.

    ??? note "Dataset Generation Code"
        ```python
        import pandas as pd
        import numpy as np

        n_rows = 500

        columns = [
            'age', 'gender',
            'smoke_w1', 'smoke_w2',
            'cholesterol_w1', 'cholesterol_w2',
            'blood_pressure_w1', 'blood_pressure_w2',
            'diabetes_w1', 'diabetes_w2',
            'exercise_w1', 'exercise_w2',
            'obesity_w1', 'obesity_w2',
            'stroke_w2'
        ]

        data = []

        for i in range(n_rows):
            row = {}
            row['age'] = np.random.randint(40, 71)  
            row['gender'] = np.random.choice([0, 1])  
            
            for feature in ['smoke', 'cholesterol', 'blood_pressure', 'diabetes', 'exercise', 'obesity']:
                w1 = np.random.choice([0, 1], p=[0.7, 0.3])
                if w1 == 1:
                    w2 = np.random.choice([0, 1], p=[0.2, 0.8])  
                else:
                    w2 = np.random.choice([0, 1], p=[0.9, 0.1])  
                row[f'{feature}_w1'] = w1
                row[f'{feature}_w2'] = w2
            
            if row['smoke_w2'] == 1 or row['cholesterol_w2'] == 1 or row['blood_pressure_w2'] == 1:
                p_stroke = 0.2  
            else:
                p_stroke = 0.05  
            row['stroke_w2'] = np.random.choice([0, 1], p=[1 - p_stroke, p_stroke])
            
            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to a new CSV file
        csv_file = './extended_stroke_longitudinal.csv'
        df.to_csv(csv_file, index=False)
        print(f"Extended CSV file '{csv_file}' created successfully.")
        ```

    The dataset looks like:

    | age | gender | smoke_w1 | smoke_w2 | cholesterol_w1 | cholesterol_w2 | blood_pressure_w1 | blood_pressure_w2 | diabetes_w1 | diabetes_w2 | exercise_w1 | exercise_w2 | obesity_w1 | obesity_w2 | stroke_w2 |
    |-----|--------|----------|----------|----------------|----------------|-------------------|-------------------|-------------|-------------|-------------|-------------|------------|------------|-----------|
    | 66  | 0      | 0        | 1        | 0              | 0              | 0                 | 0                 | 1           | 1           | 0           | 1           | 0          | 0          | 0         |
    | 59  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 1           | 1           | 1           | 1           | 1          | 1          | 1         |
    | 63  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 0           | 0           | 0           | 0           | 0          | 0          | 1         |
    | 47  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 0           | 0           | 0           | 0           | 1          | 0          | 0         |
    | 44  | 0      | 0        | 0        | 1              | 1              | 1                 | 1                 | 0           | 0           | 0           | 0           | 1          | 1          | 1         |
    | 69  | 1      | 0        | 0        | 0              | 0              | 1                 | 1                 | 0           | 0           | 0           | 0           | 0          | 0          | 0         |
    | 63  | 0      | 0        | 0        | 0              | 0              | 0                 | 0                 | 0           | 0           | 0           | 0           | 0          | 0          | 0         |
    | 48  | 1      | 0        | 0        | 0              | 0              | 0                 | 0                 | 0           | 0           | 0           | 1           | 0          | 0          | 0         |
    | 49  | 1      | 0        | 0        | 0              | 0              | 0                 | 0                 | 0           | 0           | 0           | 1           | 0          | 1          | 0         |


This tutorial introduces the data formats supported by Scikit-Longitudinal (`Sklong`), focusing on the wide format for longitudinal data. We'll explain why wide format is preferred to avoid data leakage and provide examples using a simple synthetic dataset.

!!! important "Prerequisite Reading"
    Before proceeding, ensure you've read the [Temporal Dependency Guide](temporal_dependency.md) to understand `features_group` and `non_longitudinal_features`.

## Wide vs. Long Format

Longitudinal data can be represented in two main formats:

- **Long Format**: Each observation for a subject is in a separate row, with a time indicator column. This can lead to data leakage during splitting if rows for the same subject are separated.

- **Wide Format**: Each subject is in one row, with columns for each feature at each time point (e.g., `smoke_w1`, `smoke_w2`). This format prevents leakage as all temporal data for a subject stays together.

`Sklong` focuses on wide format for safety and simplicity in machine learning workflows.

!!! tip "Converting Formats"
    If your data is in long format, pivot it to wide using pandas: `df.pivot(index='subject_id', columns='time', values='feature')`. Open an issue if you need built-in support.

## Synthetic Dataset Example

The dataset used is synthetic, mimicking health data for illustration.

- Longitudinal features: `smoke`, `cholesterol`, `blood_pressure`, `diabetes`, `exercise`, `obesity` (two waves each).
- Non-longitudinal: `age`, `gender`.
- Target: `stroke_w2` (binary).

Load and prepare:

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./extended_stroke_longitudinal.csv')
dataset.load_data()
dataset.load_target(target_column='stroke_w2')
dataset.setup_features_group([[2,3], [4,5], [6,7], [8,9], [10,11], [12,13]])  # Indices for longitudinal features
dataset.load_train_test_split(test_size=0.2, random_state=42)

print(dataset.data.head())
```

This wide format ensures safe splitting and temporal integrity.