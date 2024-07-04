# Lexico Gradient Boosting Classifier
## LexicoGradientBoostingClassifier

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_gradient_boosting.py/#L54)

``` py
LexicoGradientBoostingClassifier(
   threshold_gain: float = 0.0015, features_group: List[List[int]] = None,
   criterion: str = 'friedman_mse', splitter: str = 'lexicoRF',
   max_depth: Optional[int] = 3, min_samples_split: int = 2, min_samples_leaf: int = 1,
   min_weight_fraction_leaf: float = 0.0, max_features: Optional[Union[int, str]] = None,
   random_state: Optional[int] = None, max_leaf_nodes: Optional[int] = None,
   min_impurity_decrease: float = 0.0, ccp_alpha: float = 0.0, tree_flavor: bool = False,
   n_estimators: int = 100, learning_rate: float = 0.1
)
```

---

Gradient Boosting Classifier adapted for longitudinal data analysis.

The Lexico Gradient Boosting Classifier is an advanced ensemble algorithm designed specifically for longitudinal datasets, 
Incorporating the fundamental principles of the Gradient Boosting framework. This classifier distinguishes itself 
through the implementation of longitudinal-adapted base estimators, which are intended to capture the temporal 
complexities and interdependencies intrinsic to longitudinal data.

The base estimators of the Lexico Gradient Boosting Classifier are Lexico Decision Tree Regressors, specialised 
decision tree models capable of handling longitudinal data.

!!! quote "Lexicographical Optimisation"
    The primary goal of this approach is to prioritize the selection of more recent data points (wave ids) when determining splits in the decision tree, based on the premise that recent measurements are typically more predictive and relevant than older ones.

    Key Features:
    
    1. **Lexicographic Optimisation:** The approach prioritises features based on both their information gain ratios 
    and the recency of the data, favoring splits with more recent information.
    2. **Cython Adaptation:** This implementation leverages a fork of Scikit-learn’s fast C++-powered
    decision tree to ensure that the Lexico Decision Tree is fast and efficient, avoiding the potential 
    slowdown of a from-scratch Python implementation. Further details on the algorithm can be found in the 
    Cython adaptation available [here at Scikit-Lexicographical-Trees](https://github.com/simonprovost/scikit-lexicographical-trees/blob/21443b9dce51434b3198ccabac8bafc4698ce953/sklearn/tree/_splitter.pyx#L695) specifically in the `node_lexicoRF_split` function.

    For further scientific references, please refer to the Notes section.

## Parameters

- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list. For more details, see the documentation's "Temporal Dependency" page.
- **threshold_gain** (`float`): The threshold value for comparing gain ratios of features during the decision tree construction.
- **criterion** (`str`, optional, default="friedman_mse"): The function to measure the quality of a split. Do not change this value.
- **splitter** (`str`, optional, default="lexicoRF"): The strategy used to choose the split at each node. Do not change this value.
- **max_depth** (`Optional[int]`, default=None): The maximum depth of the tree.
- **min_samples_split** (`int`, optional, default=2): The minimum number of samples required to split an internal node.
- **min_samples_leaf** (`int`, optional, default=1): The minimum number of samples required to be at a leaf node.
- **min_weight_fraction_leaf** (`float`, optional, default=0.0): The minimum weighted fraction of the sum total of weights required to be at a leaf node.
- **max_features** (`Optional[Union[int, str]]`, default=None): The number of features to consider when looking for the best split.
- **random_state** (`Optional[int]`, default=None): The seed used by the random number generator.
- **max_leaf_nodes** (`Optional[int]`, default=None): The maximum number of leaf nodes in the tree.
- **min_impurity_decrease** (`float`, optional, default=0.0): The minimum impurity decrease required for a node to be split.
- **ccp_alpha** (`float`, optional, default=0.0): Complexity parameter used for Minimal Cost-Complexity Pruning.
- **tree_flavor** (`bool`, optional, default=False): Indicates whether to use a specific tree flavor.
- **n_estimators** (`int`, optional, default=100): The number of boosting stages to be run.
- **learning_rate** (`float`, optional, default=0.1): Learning rate shrinks the contribution of each tree by `learning_rate`.

## Methods

### Fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_gradient_boosting.py/#L162)

``` py
._fit(
   X: np.ndarray, y: np.ndarray
)
```

Fit the Lexico Gradient Boosting Longitudinal Classifier model according to the given training data.

#### Parameters

- **X** (`np.ndarray`): The training input samples.
- **y** (`np.ndarray`): The target values (class labels).

#### Returns

- **LexicoGradientBoostingClassifier**: The fitted classifier.

#### Raises

- **ValueError**: If there are less than or equal to 1 feature group.

### Predict
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_gradient_boosting.py/#L211)

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
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_gradient_boosting.py/#L227)

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

## Examples

### Example 1: Basic Usage

``` py title="Example_1: Default Parameters" linenums="1" hl_lines="7-9"
from sklearn_fork.metrics import accuracy_score
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import \
    LexicoGradientBoostingClassifier

features_group = [(0, 1), (2, 3)]  # (1)

clf = LexicoGradientBoostingClassifier(
    features_group=features_group
)
clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred)  # (2)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure.
2. Calculate the accuracy score for the predictions. Can use other metrics as well.

### Example 2: Using Specific Parameters

``` py title="Example_2: Using Specific Parameters" linenums="1" hl_lines="7-12"
from sklearn_fork.metrics import accuracy_score
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import \
    LexicoGradientBoostingClassifier

features_group = [(0, 1), (2, 3)]  # (1)

clf = LexicoGradientBoostingClassifier(
    features_group=features_group,
    threshold_gain=0.0015,  # (2)
    max_depth=3,
    random_state=42
)
clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred)  # (3)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure.
2. Set the threshold gain for the lexicographical approach. The lower the value, the closer will need the gain ratio to be between the two features to be considered equal before employing the lexicographical approach (i.e, the more recent wave will be chosen under certain conditions). The higher the value, the larger the gap needs can be between the gain ratios of the two features for the lexicographical approach to be employed.
3. Calculate the accuracy score for the predictions. Can use other metrics as well.

### Exemple 3: Using the learning rate

``` py title="Example_3: Using the learning rate" linenums="1" hl_lines="7-11"
from sklearn_fork.metrics import accuracy_score
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import \
    LexicoGradientBoostingClassifier
    
features_group = [(0, 1), (2, 3)]  # (1)

clf = LexicoGradientBoostingClassifier(
    features_group=features_group,
    threshold_gain=0.0015,
    learning_rate=0.01  # (2)
)

clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred)  # (3)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure.
2. Set the learning rate for the boosting algorithm. The learning rate shrinks the contribution of each tree by `learning_rate`. There is a trade-off between learning_rate and n_estimators.

## Notes

> For more information, please refer to the following papers:

### References
- **Ribeiro and Freitas (2020)**:
  - **Ribeiro, C. and Freitas, A., 2020, December.** A new random forest method for longitudinal data regression using a lexicographic bi-objective approach. In 2020 IEEE Symposium Series on Computational Intelligence (SSCI).

Here is the initial Python implementation of the Gradient Boosting algorithm: [Gradient Boosting Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)