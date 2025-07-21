---
---

## 🔍 Sklong: Explore Your First Estimator


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


!!! important "Prerequisite Reading"
    Ensure you've read the [Temporal Dependency Guide](temporal_dependency.md) and [Data Format Tutorial](sklong_longitudinal_data_format.md).

## Step 1: Load and Prepare Data

Using the synthetic `extended_stroke_longitudinal.csv` from the previous tutorial:

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./extended_stroke_longitudinal.csv')
dataset.load_data_target_train_test_split(target_column='stroke_w2', test_size=0.2, random_state=42)
dataset.setup_features_group([[2,3], [4,5], [6,7], [8,9], [10,11], [12,13]])
```

## Step 2: Initialize and Fit the Estimator

Use `LexicoDecisionTreeClassifier`, which prioritizes recent waves:

```python
from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier

clf = LexicoDecisionTreeClassifier(
    features_group=dataset.feature_groups(),
    threshold_gain=0.01,
    random_state=42
)

clf.fit(dataset.X_train, dataset.y_train)
```

## Step 3: Predict and Evaluate

```python
y_pred = clf.predict(dataset.X_test)
print(y_pred)  # Example output

from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(dataset.y_test, y_pred)}")
```

This introduces basic estimator usage. Experiment with hyperparameters like `threshold_gain`.
