# Lexico Random Forest Classifier
## LexicoRandomForestClassifier

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_random_forest.py/#L8)

``` py
LexicoRandomForestClassifier(
   n_estimators: int = 100, threshold_gain: float = 0.0015,
   features_group: List[List[int]] = None, max_depth: Optional[int] = None,
   min_samples_split: int = 2, min_samples_leaf: int = 1,
   min_weight_fraction_leaf: float = 0.0, max_features: Optional[Union[int, str]] = 'sqrt',
   max_leaf_nodes: Optional[int] = None, min_impurity_decrease: float = 0.0,
   class_weight: Optional[str] = None, ccp_alpha: float = 0.0, random_state: int = None, **kwargs
)
```

---

The Lexico Random Forest Classifier is an advanced ensemble algorithm specifically designed for longitudinal data analysis. 
This classifier extends the traditional random forest algorithm by incorporating a lexicographic optimisation approach 
to select the best split at each node.

!!! quote "Lexicographic Optimisation"
    The primary goal of this approach is to prioritise the selection of more recent data points (wave ids) when
    determining splits in the decision tree, based on the premise that recent measurements are typically 
    more predictive and relevant than older ones.

    Key Features:
    
    1. **Lexicographic Optimisation:** The approach prioritizes features based on both their information gain ratios and the recency of the data, favoring splits with more recent information.
    2. **Cython Adaptation**: This implementation leverages a fork of Scikit-learn’s fast C++-powered decision tree to ensure that the Lexico Random Forest is fast and efficient, avoiding the potential slowdown of a from-scratch Python implementation. Further details on the algorithm can be found in the Cython adaptation available [here at Scikit-Lexicographical-Trees](https://github.com/simonprovost/scikit-lexicographical-trees/blob/21443b9dce51434b3198ccabac8bafc4698ce953/sklearn/tree/_splitter.pyx#L695) specifically in the `node_lexicoRF_split` function.

    For further scientific references, please refer to the Notes section.

## Parameters

- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list. For more details, see the documentation's "Temporal Dependency" page.
- **threshold_gain** (`float`): The threshold value for comparing gain ratios of features during the decision tree construction.
- **n_estimators** (`int`, optional, default=100): The number of trees in the forest.
- **criterion** (`str`, optional, default="entropy"): The function to measure the quality of a split. Do not change this value.
- **splitter** (`str`, optional, default="lexicoRF"): The strategy used to choose the split at each node. Do not change this value.
- **max_depth** (`Optional[int]`, default=None): The maximum depth of the tree.
- **min_samples_split** (`int`, optional, default=2): The minimum number of samples required to split an internal node.
- **min_samples_leaf** (`int`, optional, default=1): The minimum number of samples required to be at a leaf node.
- **min_weight_fraction_leaf** (`float`, optional, default=0.0): The minimum weighted fraction of the sum total of weights required to be at a leaf node.
- **max_features** (`Optional[Union[int, str]]`, default='sqrt'): The number of features to consider when looking for the best split.
- **random_state** (`Optional[int]`, default=None): The seed used by the random number generator.
- **max_leaf_nodes** (`Optional[int]`, default=None): The maximum number of leaf nodes in the tree.
- **min_impurity_decrease** (`float`, optional, default=0.0): The minimum impurity decrease required for a node to be split.
- **class_weight** (`Optional[str]`, default=None): Weights associated with classes in the form of {class_label: weight}.
- **ccp_alpha** (`float`, optional, default=0.0): Complexity parameter used for Minimal Cost-Complexity Pruning.
- **kwargs** (`dict`): The keyword arguments for the RandomForestClassifier.

## Attributes

- **classes_** (`ndarray` of shape (n_classes,)): The class labels (single output problem).
- **n_classes_** (`int`): The number of classes (single output problem).
- **n_features_** (`int`): The number of features when fit is performed.
- **n_outputs_** (`int`): The number of outputs when fit is performed.
- **feature_importances_** (`ndarray` of shape (n_features,)): The impurity-based feature importances.
- **max_features_** (`int`): The inferred value of max_features.
- **estimators_** (`list` of LexicoDecisionTreeClassifier): The collection of fitted sub-estimators.

## Methods

### Fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_random_forest.py/#L276)

``` py
.fit(
   X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
)
```

Fit the LexicoRandomForestClassifier model according to the given training data.

#### Parameters

- **X** (`np.ndarray`): The training input samples.
- **y** (`np.ndarray`): The target values (class labels).
- **sample_weight** (`Optional[np.ndarray]`, default=None): Sample weights.

#### Returns

- **LexicoRandomForestClassifier**: The fitted random forest classifier.

### Predict
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_random_forest.py/#L316)

``` py
.predict(
   X: np.ndarray
)
```

Predict class labels for samples in X.

#### Parameters

- **X** (`np.ndarray`): The input samples.

#### Returns

- **np.ndarray**: The predicted class labels for each input sample.

### Predict Proba
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_random_forest.py/#L332)

``` py
.predict_proba(
   X: np.ndarray
)
```

Predict class probabilities for samples in X.

#### Parameters

- **X** (`np.ndarray`): The input samples.

#### Returns

- **np.ndarray**: The predicted class probabilities for each input sample.

## Examples

### Dummy Longitudinal Dataset

!!! example "Consider the following dataset: `stroke.csv`"
    Features:
    
    - `smoke` (longitudinal) with two waves/time-points
    - `cholesterol` (longitudinal) with two waves/time-points
    - `age` (non-longitudinal)
    - `gender` (non-longitudinal)

    Target:
    
    - `stroke` (binary classification) at wave/time-point 2 only for the sake of the example
    
    The dataset is shown below (`w` stands for `wave` in ELSA):

    | smoke_w1 | smoke_w2 | cholesterol_w1 | cholesterol_w2 | age | gender | stroke_w2 |
    |--------------|--------------|--------------------|--------------------|-----|--------|---------------|
    | 0            | 1            | 0                  | 1                  | 45  | 1      | 0             |
    | 1            | 1            | 1                  | 1                  | 50  | 0      | 1             |
    | 0            | 0            | 0                  | 0                  | 55  | 1      | 0             |
    | 1            | 1            | 1                  | 1                  | 60  | 0      | 1             |
    | 0            | 1            | 0                  | 1                  | 65  | 1      | 0             |

### Example 1: Basic Usage

```py title="Example_1: Default Parameters" linenums="1" hl_lines="6-8"
from sklearn_fork.metrics import accuracy_score
from scikit_longitudinal.estimators.ensemble.lexicographical import LexicoRandomForestClassifier

features_group = [(0, 1), (2, 3)]  # (1)

clf = LexicoRandomForestClassifier(
    features_group=features_group
)
clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred)  # (2)
```

1. Either define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure.  If the data was from the ELSA database, you could have used the pre-sets such as `.setup_features_group('elsa')`.
2. Calculate the accuracy score for the predictions. Can use other metrics as well.

### Example 2: How-To Set Threshold Gain of the Lexicographical Approach

```py title="Example_2: How-To Set Threshold Gain of the Lexicographical Approach" linenums="1" hl_lines="6-9"
from sklearn_fork.metrics import accuracy_score
from scikit_longitudinal.estimators.ensemble.lexicographical import LexicoRandomForestClassifier

features_group = [(0, 1), (2, 3)]  # (1)

clf = LexicoRandomForestClassifier(
    threshold_gain=0.001,  # (2)
    features_group=features_group
)

clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred)  # (3)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure. If the data was from the ELSA database, you could have used the pre-sets such as `.setup_features_group('elsa')`.
2. Set the threshold gain for the lexicographical approach. The lower the value, the closer the gain ratio needs to be between the two features to be considered equal before employing the lexicographical approach (i.e., the more recent wave will be chosen under certain conditions). The higher the value, the larger the gap can be between the gain ratios of the two features for the lexicographical approach to be employed.
3. Calculate the accuracy score for the predictions. Can use other metrics as well.

### Example 3: How-To Set the Number of Estimators

```py title="Example_3: How-To Set the Number of Estimators" linenums="1" hl_lines="6-9"
from sklearn_fork.metrics import accuracy_score
from scikit_longitudinal.estimators.ensemble.lexicographical import LexicoRandomForestClassifier

features_group = [(0, 1), (2, 3)]  # (1)

clf = LexicoRandomForestClassifier(
    n_estimators=200,  # (2)
    features_group=features_group
)

clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred)  # (3)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure. If the data was from the ELSA database, you could have used the pre-sets such as `.setup_features_group('elsa')`.
2. Set the number of estimators (trees) in the forest.
3. Calculate the accuracy score for the predictions. Can use other metrics as well.

## Notes

> For more information, please refer to the following papers:

### References
- **Ribeiro and Freitas (2020)**:
  - **Ribeiro, C. and Freitas, A., 2020, December.** A new random forest method for longitudinal data classification using a lexicographic bi-objective approach. In 2020 IEEE Symposium Series on Computational Intelligence (SSCI) (pp. 806-813). IEEE.
- **Ribeiro and Freitas (2024)**:
  - **Ribeiro, C. and Freitas, A.A., 2024.** A lexicographic optimisation approach to promote more recent features on longitudinal decision-tree-based classifiers: applications to the English Longitudinal Study of Ageing. Artificial Intelligence Review, 57(4), p.84.
