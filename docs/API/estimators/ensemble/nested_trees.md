# Nested Trees Classifier
## NestedTreesClassifier

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/nested_trees/nested_trees.py/#L19)

``` py
NestedTreesClassifier(
   features_group: List[List[int]] = None,
   non_longitudinal_features: List[Union[int, str]] = None, max_outer_depth: int = 3,
   max_inner_depth: int = 2, min_outer_samples: int = 5,
   inner_estimator_hyperparameters: Optional[Dict[str, Any]] = None,
   save_nested_trees: bool = False, parallel: bool = False, num_cpus: int = -1
)
```

---

The Nested Trees Classifier is a unique and innovative classification algorithm specifically designed for longitudinal 
datasets. This method enhances traditional decision tree algorithms by embedding smaller decision trees within the nodes
of a primary tree structure, leveraging the inherent information in longitudinal data optimally.

!!! quote "Nested Trees Structure"
    The outer decision tree employs a custom algorithm that selects longitudinal attributes, categorised as groups of 
    time-specific attributes. The inner embedded decision tree uses Scikit Learn's decision tree algorithm, 
    partitioning the dataset based on the longitudinal attribute of the parent node.

!!! info "Wrapper Around Sklearn DecisionTreeClassifier"
    This class wraps the `sklearn` DecisionTreeClassifier, offering a familiar interface while incorporating 
    enhancements for longitudinal data. It ensures effective processing and learning from data collected over multiple 
    time points.

## Parameters

- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list. For more details, see the documentation's "Temporal Dependency" page.
- **non_longitudinal_features** (`List[Union[int, str]]`, optional): A list of indices of features that are not longitudinal attributes. Defaults to None.
- **max_outer_depth** (`int`, optional, default=3): The maximum depth of the outer custom decision tree.
- **max_inner_depth** (`int`, optional, default=2): The maximum depth of the inner decision trees.
- **min_outer_samples** (`int`, optional, default=5): The minimum number of samples required to split an internal node in the outer decision tree.
- **inner_estimator_hyperparameters** (`Dict[str, Any]`, optional): A dictionary of hyperparameters to be passed to the inner Scikit-learn decision tree estimators. Defaults to None.
- **save_nested_trees** (`bool`, optional, default=False): If set to True, the nested trees structure plot will be saved, which may be useful for model interpretation and visualization.
- **parallel** (`bool`, optional, default=False): Whether to use parallel processing.
- **num_cpus** (`int`, optional, default=-1): The number of CPUs to use for parallel processing. Defaults to -1 (use all available).

## Attributes

- **root** (`Node`, optional): The root node of the outer decision tree. Set to None upon initialization, it will be updated during the model fitting process.

## Methods

### Fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/nested_trees/nested_trees.py/#L211)

``` py
._fit(
   X: np.ndarray, y: np.ndarray
)
```

Fit the Nested Trees Classifier model according to the given training data.

#### Parameters

- **X** (`np.ndarray`): The training input samples.
- **y** (`np.ndarray`): The target values (class labels).

#### Returns

- **NestedTreesClassifier**: The fitted classifier.

#### Raises

- **ValueError**: If there are less than or equal to 1 feature group.

### Predict
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/nested_trees/nested_trees.py/#L243)

``` py
._predict(
   X: np.ndarray
)
```

Predict class labels for samples in X.

#### Parameters

- **X** (`np.ndarray`): The input samples.

#### Returns

- **np.ndarray**: The predicted class labels for each input sample.

### Predict Proba
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/nested_trees/nested_trees.py/#L260)

``` py
._predict_proba(
   X: np.ndarray
)
```

Predict class probabilities for samples in X.

#### Parameters

- **X** (`np.ndarray`): The input samples.

#### Returns

- **np.ndarray**: The predicted class probabilities for each input sample.

### Print Nested Tree
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/nested_trees/nested_trees.py/#L497)

``` py
.print_nested_tree(
   node: Optional['NestedTreesClassifier.Node'] = None, depth: int = 0, prefix: str = '',
   parent_name: str = ''
)
```

Print the structure of the nested tree classifier.

#### Parameters

- **node** (`Optional[NestedTreesClassifier.Node]`, optional): The current node in the outer decision tree. If None, start from the root node. Default is None.
- **depth** (`int`, optional, default=0): The current depth of the node in the outer decision tree.
- **prefix** (`str`, optional, default=""`): A string to prepend before the node's name in the output.
- **parent_name** (`str`, optional, default=""`): The name of the parent node in the outer decision tree.

## Examples

### Dummy Longitudinal Dataset

!!! example "Consider the following dataset"
    Features:
    
    - `smoke` (longitudinal) with two waves/time-points
    - `cholesterol` (longitudinal) with two waves/time-points
    - `age` (non-longitudinal)
    - `gender` (non-longitudinal)

    Target:
    
    - `stroke` (binary classification) at wave/time-point 2 only for the sake of the example
    
    The dataset is shown below:

    | smoke_wave_1 | smoke_wave_2 | cholesterol_wave_1 | cholesterol_wave_2 | age | gender | stroke_wave_2 |
    |--------------|--------------|--------------------|--------------------|-----|--------|---------------|
    | 0            | 1            | 0                  | 1                  | 45  | 1      | 0             |
    | 1            | 1            | 1                  | 1                  | 50  | 0      | 1             |
    | 0            | 0            | 0                  | 0                  | 55  | 1      | 0             |
    | 1            | 1            | 1                  | 1                  | 60  | 0      | 1             |
    | 0            | 1            | 0                  | 1                  | 65  | 1      | 0             |



### Example 1: Basic Usage

``` py title="Example 1: Basic Usage" linenums="1" hl_lines="8-11"
from scikit_longitudinal.estimators.ensemble import NestedTreesClassifier
from sklearn_fork.model_selection import train_test_split
from sklearn_fork.metrics import accuracy_score

features_group = [(0, 1), (2, 3)] # (1)
non_longitudinal_features = [4, 5] # (2)

clf = NestedTreesClassifier(
    features_group=features_group,
    non_longitudinal_features=non_longitudinal_features,
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred) # (3)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Define the non-longitudinal features or use a pre-set from the LongitudinalDataset class.
3. Fit the model with the training data, make predictions, and evaluate the model using the accuracy score.

### Example 2: Using Custom Hyperparameters for Inner Estimators

``` py title="Example 2: Using Custom Hyperparameters for Inner Estimators" linenums="1" hl_lines="8-18"
from scikit_longitudinal.estimators.ensemble import NestedTreesClassifier
from sklearn_fork.model_selection import train_test_split
from sklearn_fork.metrics import accuracy_score

features_group = [(0, 1), (2, 3)] # (1)
non_longitudinal_features = [4, 5] # (2)

inner_hyperparameters = { # (3)
    "criterion": "gini",
    "splitter": "best",
    "max_depth": 3
}

clf = NestedTreesClassifier(
    features_group=features_group,
    non_longitudinal_features=non_longitudinal_features,
    inner_estimator_hyperparameters=inner_hyperparameters,
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred) # (4)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Define the non-longitudinal features or use a pre-set from the LongitudinalDataset class.
3. Define custom hyperparameters for the inner decision tree estimators.
4. Fit the model with the training data, make predictions, and evaluate the model using the accuracy score.

### Example 3: Using Parallel Processing

``` py title="Example 3: Using Parallel Processing" linenums="1" hl_lines="8-13"
from scikit_longitudinal.estimators.ensemble import NestedTreesClassifier
from sklearn_fork.model_selection import train_test_split
from sklearn_fork.metrics import accuracy_score

features_group = [(0, 1), (2, 3)] # (1)
non_longitudinal_features = [4, 5] # (2)

clf = NestedTreesClassifier(
    features_group=features_group,
    non_longitudinal_features=non_longitudinal_features,
    parallel=True, # (3)
    num_cpus=-1 # (4)
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred) # (5)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Define the non-longitudinal features or use a pre-set from the LongitudinalDataset class.
3. Enable parallel processing.
4. Specify the number of CPUs to use for parallel processing. That is that each available CPU will be used to train one decision tree of one longitudinal attribute.
5. Fit the model with the training data, make predictions, and evaluate the model using the accuracy score.

### Example 4: Saving the Nested Trees Structure

``` py title="Example 4: Saving the Nested Trees Structure" linenums="1" hl_lines="8-12"
from scikit_longitudinal.estimators.ensemble import NestedTreesClassifier
from sklearn_fork.model_selection import train_test_split
from sklearn_fork.metrics import accuracy_score

features_group = [(0, 1), (2, 3)] # (1)
non_longitudinal_features = [4, 5] # (2)

clf = NestedTreesClassifier(
    features_group=features_group,
    non_longitudinal_features=non_longitudinal_features,
    save_nested_trees=True # (3)
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred) # (4)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Define the non-longitudinal features or use a pre-set from the LongitudinalDataset class.
3. Request the nested trees structure plot to be saved.
4. Fit the model with the training data, make predictions, and evaluate the model using the accuracy score.

### Example 5: Printing the Nested Trees Structure (markdown format)

``` py title="Example 5: Printing the Nested Trees Structure" linenums="1" hl_lines="8-11"
from scikit_longitudinal.estimators.ensemble import NestedTreesClassifier
from sklearn_fork.model_selection import train_test_split
from sklearn_fork.metrics import accuracy_score

features_group = [(0, 1), (2, 3)] # (1)
non_longitudinal_features = [4, 5] # (2)

clf = NestedTreesClassifier(
    features_group=features_group,
    non_longitudinal_features=non_longitudinal_features,
)

clf.fit(X_train, y_train)
clf.print_nested_tree() # (3)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Define the non-longitudinal features or use a pre-set from the LongitudinalDataset class.
3. The output will show the structure of the nested tree in markdown format.

## Notes

> For more information, see the following paper on the Nested Trees algorithm:

### References
- **Ovchinnik, Otero, and Freitas (2022)**:
  - **Ovchinnik, S., Otero, F. and Freitas, A.A., 2022, April.** Nested trees for longitudinal classification. In Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing (pp. 441-444). Vancouver.

Here is the initial Java implementation of the Nested Trees algorithm: [Nested Trees GitHub](https://github.com/NestedTrees/NestedTrees)