# Lexico Decision Tree Classifier
## LexicoDecisionTreeClassifier

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/trees/lexicographical/lexico_decision_tree.py/#L8)

``` py
LexicoDecisionTreeClassifier(
   threshold_gain: float = 0.0015, features_group: List[List[int]] = None,
   criterion: str = 'entropy', splitter: str = 'lexicoRF',
   max_depth: Optional[int] = None, min_samples_split: int = 2,
   min_samples_leaf: int = 1, min_weight_fraction_leaf: float = 0.0,
   max_features: Optional[Union[int, str]] = None, random_state: Optional[int] = None,
   max_leaf_nodes: Optional[int] = None, min_impurity_decrease: float = 0.0,
   class_weight: Optional[str] = None, ccp_alpha: float = 0.0
)
```

---

The Lexico Decision Tree Classifier is an advanced classification model specifically designed for longitudinal data. 
This implementation extends the traditional decision tree algorithm by incorporating a lexicographic optimisation approach.

!!! quote "Lexicographic Optimisation"
    The primary goal of this approach is to prioritise the selection of more recent data points (wave ids) when 
    determining splits in the decision tree, based on the premise that recent measurements are typically more 
    predictive and relevant than older ones.

    Key Features:
    
    1. **Lexicographic Optimisation:** The approach prioritises features based on both their information gain ratios 
    and the recency of the data, favoring splits with more recent information.
    2. **Cython Adaptation**: This implementation leverages a fork of Scikit-learn’s fast C++-powered decision 
    tree to ensure that the Lexico Decision Tree is fast and efficient, avoiding the potential slowdown of a 
    from-scratch Python implementation. Further details on the algorithm can be found in the Cython adaptation available [here at Scikit-Lexicographical-Trees](https://github.com/simonprovost/scikit-lexicographical-trees/blob/21443b9dce51434b3198ccabac8bafc4698ce953/sklearn/tree/_splitter.pyx#L695) specifically in the `node_lexicoRF_split` function.

    For further scientific references, please refer to the Notes section.

## Parameters

- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list. For more details, see the documentation's "Temporal Dependency" page.
- **threshold_gain** (`float`): The threshold value for comparing gain ratios of features during the decision tree construction.
- **criterion** (`str`, optional, default="entropy"): The function to measure the quality of a split. Do not change this value.
- **splitter** (`str`, optional, default="lexicoRF"): The strategy used to choose the split at each node. Do not change this value.
- **max_depth** (`Optional[int]`, default=None): The maximum depth of the tree.
- **min_samples_split** (`int`, optional, default=2): The minimum number of samples required to split an internal node.
- **min_samples_leaf** (`int`, optional, default=1): The minimum number of samples required to be at a leaf node.
- **min_weight_fraction_leaf** (`float`, optional, default=0.0): The minimum weighted fraction of the sum total of weights required to be at a leaf node.
- **max_features** (`Optional[Union[int, str]]`, default=None): The number of features to consider when looking for the best split.
- **random_state** (`Optional[int]`, default=None): The seed used by the random number generator.
- **max_leaf_nodes** (`Optional[int]`, default=None): The maximum number of leaf nodes in the tree.
- **min_impurity_decrease** (`float`, optional, default=0.0): The minimum impurity decrease required for a node to be split.
- **class_weight** (`Optional[str]`, default=None): Weights associated with classes in the form of {class_label: weight}.
- **ccp_alpha** (`float`, optional, default=0.0): Complexity parameter used for Minimal Cost-Complexity Pruning.

## Attributes

- **classes_** (`ndarray` of shape (n_classes,)): The classes labels (single output problem).
- **n_classes_** (`int`): The number of classes (single output problem).
- **n_features_** (`int`): The number of features when fit is performed.
- **n_outputs_** (`int`): The number of outputs when fit is performed.
- **feature_importances_** (`ndarray` of shape (n_features,)): The impurity-based feature importances.
- **max_features_** (`int`): The inferred value of max_features.
- **tree_** (`Tree` object): The underlying Tree object.

## Methods

### Fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/trees/lexicographical/lexico_decision_tree.py/#L139)

``` py
.fit(
   X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
   check_input: bool = True, X_idx_sorted: Optional[np.ndarray] = None
)
```

Fit the decision tree classifier.

This method fits the `LexicoDecisionTreeClassifier` to the given data.

#### Parameters

- **X** (`np.ndarray`): The training input samples of shape `(n_samples, n_features)`.
- **y** (`np.ndarray`): The target values of shape `(n_samples,)`.
- **sample_weight** (`Optional[np.ndarray]`, default=None): Sample weights.
- **check_input** (`bool`, default=True): Allow to bypass several input checking. Don’t use this parameter unless you know what you do.
- **X_idx_sorted** (`Optional[np.ndarray]`, default=None): The indices of the sorted training input samples. If many tree are grown on the same dataset, this allows the use of sorted representations in max_features and max_depth searches.

#### Returns

- **LexicoDecisionTreeClassifier**: The fitted decision tree classifier.

### Predict
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/trees/lexicographical/lexico_decision_tree.py/#L205)

``` py
.predict(
   X: np.ndarray
)
```

Predict class or regression value for X.

The predicted class or the predict value of an input sample is computed as the mean predicted class of the trees in the forest.

#### Parameters

- **X** (`np.ndarray`): The input samples of shape `(n_samples, n_features)`.

#### Returns

- **np.ndarray**: The predicted classes.

### Predict Proba
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/trees/lexicographical/lexico_decision_tree.py/#L270)

``` py
.predict_proba(
   X: np.ndarray
)
```

Predict class probabilities for X.

The predicted class probabilities of an input sample are computed as the mean predicted class probabilities of the trees in the forest. The class probability of a single tree is the fraction of samples of the same class in a leaf.

#### Parameters

- **X** (`np.ndarray`): The input samples of shape `(n_samples, n_features)`.

#### Returns

- **np.ndarray**: The predicted class probabilities.


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

``` py title="Example_1: Default Parameters" linenums="1" hl_lines="5-7"
from sklearn_fork.metrics imp mators.tree import LexicoDecisionTreeClassifier

features_group = [(0,1), (2,3)] # (1)

clf = LexicoDecisionTreeClassifier(
    features_group=features_group
)
clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred) # (2)
```
1. Either define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure.
2. Calculate the accuracy score for the predictions. Can use other metrics as well.

### Example 2: How-To Set Threshold Gain of the Lexicographical Approach?

``` py title="example_1: How-To Set Threshold Gain of the Lexicographical Approach" linenums="1" hl_lines="6-9"
from sklearn_fork.metrics import accuracy_score
from scikit_longitudinal.estimators.tree import LexicoDecisionTreeClassifier

features_group = [(0,1), (2,3)] # (1)

clf = LexicoDecisionTreeClassifier(
    threshold_gain=0.001, # (2)
    features_group=features_group
)

clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred) # (3)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure.
2. Set the threshold gain for the lexicographical approach. The lower the value, the closer will need the gain ratio to be between the two features to be considered equal before employing the lexicographical approach (i.e, the more recent wave will be chosen under certain conditions). The higher the value, the larger the gap needs can be between the gain ratios of the two features for the lexicographical approach to be employed.
3. Calculate the accuracy score for the predictions. Can use other metrics as well.

## Notes

> For more information, please refer to the following paper:

### References
- **Ribeiro and Freitas (2020)**:
  - **Ribeiro, C. and Freitas, A., 2020, December.** A new random forest method for longitudinal data classification using a lexicographic bi-objective approach. In 2020 IEEE Symposium Series on Computational Intelligence (SSCI) (pp. 806-813). IEEE.
- **Ribeiro and Freitas (2024)**:
  - **Ribeiro, C. and Freitas, A.A., 2024.** A lexicographic optimisation approach to promote more recent features on longitudinal decision-tree-based classifiers: applications to the English Longitudinal Study of Ageing. Artificial Intelligence Review, 57(4), p.84.
