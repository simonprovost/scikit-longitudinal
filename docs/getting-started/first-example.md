---
icon: lucide/braces
---

# First Example

Once `Sklong` is installed, a good first milestone is to run a complete longitudinal classification workflow end to end.

The example below uses [`LongitudinalDataset`](../API/data_preparation/longitudinal_dataset.md) to prepare the data and a longitudinal-aware [`LexicoGradientBoostingClassifier`](../API/estimators/ensemble/lexico_gradient_boosting.md) to train on wave-aware feature groups.

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import (
    LexicoGradientBoostingClassifier,
)
from sklearn.metrics import classification_report

# Load your dataset (replace 'stroke.csv' with your actual dataset path)
dataset = LongitudinalDataset("./stroke.csv")

# Set up the target column and split the data into training and testing sets
dataset.load_data_target_train_test_split(
    target_column="class_stroke_wave_4",
)

# Set up feature groups (temporal dependencies)
# Use a pre-set for English Longitudinal Study of Ageing (ELSA) data or define manually
dataset.setup_features_group(input_data="elsa")

# Initialise the classifier with feature groups
model = LexicoGradientBoostingClassifier(
    features_group=dataset.feature_groups(),
    threshold_gain=0.00015,  # Adjust hyperparameters as needed
)

# Fit the model to the training data
model.fit(dataset.X_train, dataset.y_train)

# Make predictions on the test data
y_pred = model.predict(dataset.X_test)

# Print the classification report
print(classification_report(dataset.y_test, y_pred))
```

!!! info "What is the LexicoGradientBoostingClassifier?"
    It is a longitudinal variant of Gradient Boosting that uses a lexicographical strategy to prioritise more recent `waves` when evaluating candidate splits. If you want the theory behind that design, see the [API reference](../API/estimators/ensemble/lexico_gradient_boosting.md) and the [broader paper on the lexicographical split strategy](https://link.springer.com/article/10.1007/s10462-024-10718-1).

!!! tip "Where is the data?"
    `Scikit-longitudinal` does not ship datasets by default, mainly for privacy and redistribution reasons.
    You can use your own longitudinal datasets or start with publicly available ones such as the [ELSA dataset](https://www.elsa-project.ac.uk/).
    If you would like a bundled synthetic example dataset in the future, [open an issue](https://github.com/simonprovost/scikit-longitudinal/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen).
