---
icon: lucide/sliders-horizontal
---

# Hyperparameter Tuning: Grid vs. Random Search

!!! tip "Dataset Used in Tutorials"
    Use the shared synthetic dataset defined in the [tutorials overview](overview.md#dataset-used-in-tutorials). Generate it once there and reuse it here.

Tune longitudinal-aware models to squeeze out extra performance. This guide compares grid search and random search for `LexicoRandomForestClassifier`, focusing on `threshold_gain` plus common random-forest hyperparameters.

The animation below summarises the contrast: grid search sweeps a regular lattice of hyperparameter combinations (thorough but expensive), while random search scatters samples across the same plane and, in practice, often lands inside high-performing regions that a coarse grid would miss.

<figure class="expandable-media" markdown="span" style="text-align: center;">
 [![Grid search vs. random search on the same plane](../assets/images/tutorials/sklong_hyperparameter_tuning/GridVsRandom.avif#only-light){ width="100%" loading="lazy" }](../assets/images/tutorials/sklong_hyperparameter_tuning/GridVsRandom.avif){ .expandable-media__trigger }
 [![Grid search vs. random search on the same plane](../assets/images/tutorials/sklong_hyperparameter_tuning/GridVsRandomDark.avif#only-dark){ width="100%" loading="lazy" }](../assets/images/tutorials/sklong_hyperparameter_tuning/GridVsRandomDark.avif){ .expandable-media__trigger }
 <figcaption>Click the image to expand it.</figcaption>
</figure>

## Step 1: Load data and define temporal dependencies

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./extended_stroke_longitudinal.csv')
dataset.load_data_target_train_test_split(target_column='stroke_w2', test_size=0.2, random_state=42)
dataset.setup_features_group([[2,3], [4,5], [6,7], [8,9], [10,11], [12,13]])
features_group = dataset.feature_groups()
```

## Step 2: Grid search key hyperparameters

```python
from sklearn.model_selection import GridSearchCV
from scikit_longitudinal.estimators.ensemble import LexicoRandomForestClassifier

param_grid = {
 'threshold_gain': [0.0001, 0.001, 0.01],
 'n_estimators': [50, 100, 200],
 'max_depth': [None, 5, 10],
}

grid = GridSearchCV(
 estimator=LexicoRandomForestClassifier(features_group=features_group, random_state=42),
 param_grid=param_grid,
 cv=3,
 n_jobs=-1,
)

grid.fit(dataset.X_train, dataset.y_train)
print(f"Best params (grid search): {grid.best_params_}")
```

## Step 3: Random search broader spaces

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_distributions = {
 'threshold_gain': loguniform(1e-4, 1e-1),
 'n_estimators': randint(50, 300),
 'max_depth': [None, 5, 10, 15],
 'max_features': ['sqrt', 'log2', 0.8],
}

random_search = RandomizedSearchCV(
 estimator=LexicoRandomForestClassifier(features_group=features_group, random_state=42),
 param_distributions=param_distributions,
 n_iter=12,
 cv=3,
 n_jobs=-1,
 random_state=42,
)

random_search.fit(dataset.X_train, dataset.y_train)
print(f"Best params (random search): {random_search.best_params_}")
```

- **Grid search**: exhaustive within a small, carefully chosen grid—great when you have strong priors on useful values.
- **Random search**: samples diverse combinations quickly—useful when exploring larger spaces or when some parameters benefit from logarithmic sampling (e.g., `threshold_gain`).
