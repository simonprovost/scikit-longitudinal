# Longitudinal Pipeline
## Longitudinal Pipeline

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/pipeline.py/#L16)

``` py
LongitudinalPipeline(
   steps: List[Tuple[str, Any]], features_group: List[List[int]],
   non_longitudinal_features: List[Union[int, str]] = None,
   update_feature_groups_callback: Union[Callable, str] = None,
   feature_list_names: List[str] = None
)
```

Custom pipeline for handling and processing longitudinal techniques (preprocessors, classifier, etc.). This class extends scikit-learn's `Pipeline` to offer specialized methods and attributes for working with longitudinal data. It ensures that the longitudinal features and their structure are updated throughout the pipeline's transformations.

## Parameters

- **steps** (`List[Tuple[str, Any]]`): List of (name, transform) tuples (implementing `fit`/`transform`) that are chained, in the order in which they are chained, with the last object being an estimator.
- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list. For more details, see the documentation's "Temporal Dependency" page.
- **non_longitudinal_features** (`List[Union[int, str]]`, optional): A list of indices of features that are not longitudinal attributes. Defaults to None.
- **update_feature_groups_callback** (`Union[Callable, str]`, optional): Callback function to update feature groups. This function is invoked to update the structure of longitudinal features during pipeline transformations.
- **feature_list_names** (`List[str]`, optional): List of names corresponding to the features.

## Attributes

- **_longitudinal_data** (`np.ndarray`): Longitudinal data being processed.
- **selected_feature_indices_** (`np.ndarray`): Indices of the selected features.
- **final_estimator** (`Any`): Final step in the pipeline.

> **Note:**  
> While this class maintains the interface of scikit-learn's `Pipeline`, it includes specific methods and validations to ensure the correct processing of longitudinal data.

## Methods

### Fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/pipeline.py/#L68)

``` py
.fit(
   X: np.ndarray, y: Optional[Union[pd.Series, np.ndarray]] = None,
   **fit_params: Dict[str, Any]
)
```

Fit the transformers in the pipeline and then the final estimator. For each step, the transformers are configured and fitted. The data is transformed and updated for each step, ensuring that the longitudinal feature structure is maintained.

#### Parameters

- **X** (`np.ndarray`): The input data.
- **y** (`Optional[Union[pd.Series, np.ndarray]]`): The target variable.
- **fit_params** (`Dict[str, Any]`): Additional fitting parameters.

#### Returns

- **LongitudinalPipeline**: The fitted pipeline.

### Predict
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/pipeline.py/#L185)

``` py
.predict(
   X: np.ndarray, **predict_params: Dict[str, Any]
)
```

Predict the target values using the final estimator of the pipeline.

#### Parameters

- **X** (`np.ndarray`): The input data.
- **predict_params** (`Dict[str, Any]`): Additional prediction parameters.

#### Returns

- **np.ndarray**: Predicted values.

#### Raises

- **NotImplementedError**: If the final estimator does not have a predict method.

### Predict_proba
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/pipeline.py/#L210)

``` py
.predict_proba(
   X: np.ndarray, **predict_params: Dict[str, Any]
)
```

Predict the probability of the target values using the final estimator of the pipeline.

#### Parameters

- **X** (`np.ndarray`): The input data.
- **predict_params** (`Dict[str, Any]`): Additional prediction parameters.

#### Returns

- **np.ndarray**: Predicted probabilities.

#### Raises

- **NotImplementedError**: If the final estimator does not have a predict_proba method.


### Transform
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/pipeline.py/#L218)

``` py
.transform(
   X: np.ndarray, **transform_params: Dict[str, Any]
)
```

Transform the input data using the final estimator of the pipeline.

#### Parameters

- **X** (`np.ndarray`): The input data.
- **transform_params** (`Dict[str, Any]`): Additional transformation parameters.

#### Returns

- **np.ndarray**: Transformed data.

## Detailed Explanation of `update_feature_groups_callback`

The `update_feature_groups_callback` is a crucial component in the `LongitudinalPipeline`. This callback function is responsible for updating the structure of longitudinal features during each step of the pipeline. Here’s a detailed breakdown of how it works:

### Purpose

The `update_feature_groups_callback` ensures that the structure of longitudinal features is accurately maintained and updated as data flows through the pipeline's transformers. This is essential because longitudinal data often requires specific handling to preserve its temporal or grouped characteristics.

### Default Implementation

By default, the `LongitudinalPipeline` includes a built-in callback function that automatically manages the update of longitudinal features for the current transformers/estimators of the library. This default implementation ensures that users can utilise the pipeline without needing to provide a custom callback for the current available techniques, simplifying the initial setup.

### Custom Implementation

For more advanced use cases, users can provide their custom callback function. E.g because you have a new data pre-processing technique that changes the temporal structure of the dataset. This custom function can be passed as a lambda or a regular function. The custom callback allows users to implement specific logic tailored to their unique longitudinal data processing needs.

### Function Signature

``` py
def update_feature_groups_callback(
    step_idx: int,
    longitudinal_dataset: LongitudinalDataset,
    y: Optional[Union[pd.Series, np.ndarray]],
    name: str,
    transformer: TransformerMixin
) -> Tuple[np.ndarray, List[List[int]], List[Union[int, str]], List[str]]:
    ...
```

### Parameters

- **step_idx** (`int`): The index of the current step in the pipeline.
- **longitudinal_dataset** (`LongitudinalDataset`): A custom dataset object that includes the longitudinal data and feature groups.
- **y** (`Optional[Union[pd.Series, np.ndarray]]`): The target variable.
- **name** (`str`): The name of the current transformer.
- **transformer** (`TransformerMixin`): The current transformer being applied in the pipeline.

### Returns

- **updated_longitudinal_data** (`np.ndarray`): The updated longitudinal data.
- **updated_features_group** (`List[List[int]]`): The updated grouping of longitudinal features.
- **updated_non_longitudinal_features** (`List[Union[int, str]]`): The updated list of non-longitudinal features.
- **updated_feature_list_names** (`List[str]`): The updated list of feature names.

### How It Works

1. **Initialise Dataset**: The function starts by initialising a `LongitudinalDataset` object using the current state of the longitudinal data and feature groups.

2. **Update Features**: The callback function is invoked with the current step index, the `LongitudinalDataset` object, the target variable, the name of the transformer, and the transformer itself.

3. **Return Updated Data**: The callback function returns the updated longitudinal data, features group, non-longitudinal features, and feature list names, which are then used in subsequent steps of the pipeline.

### Flexibility with Lambda Functions

Users can pass a lambda function as the `update_feature_groups_callback` to quickly define custom update logic. For example:

``` py hl_lines="4-6"
pipeline = LongitudinalPipeline(
    steps=[...],
    features_group=[...],
    update_feature_groups_callback=lambda step_idx, dataset, y, name, transformer: (
        custom_update_function(step_idx, dataset, y, name, transformer)
    )
)
```

This allows for easy customisation and experimentation with different feature group update strategies. In the meantime,
for further details on the `LongitudinalPipeline` class, refer to the source code, or open a Github issue.