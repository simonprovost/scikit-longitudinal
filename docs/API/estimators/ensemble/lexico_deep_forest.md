# Lexico Deep Forest Classifier
## LexicoDeepForestClassifier

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_deep_forest.py/#L100)

``` py
LexicoDeepForestClassifier(
   longitudinal_base_estimators: Optional[List[LongitudinalEstimatorConfig]] = None,
   features_group: List[List[int]] = None,
   non_longitudinal_features: List[Union[int, str]] = None,
   diversity_estimators: bool = True, random_state: int = None,
   single_classifier_type: Optional[Union[LongitudinalClassifierType, str]] = None,
   single_count: Optional[int] = None, max_layers: int = 5
)
```

---

The Lexico Deep Forest Classifier is an advanced ensemble algorithm specifically designed for longitudinal data analysis. This classifier extends the fundamental principles of the Deep Forest framework by incorporating longitudinal-adapted base estimators to capture the temporal complexities and interdependencies inherent in longitudinal data.

!!! quote "Lexico Deep Forest with the Lexicographical Optimisation"
    - **Accurate Learners**: Longitudinal-adapted base estimators form the backbone of the ensemble, capable of handling the temporal aspect of longitudinal data.
    - **Weak Learners**: Diversity estimators enhance the overall diversity of the model, improving its robustness and generalization capabilities.
    - **Cython Adaptation**: This implementation leverages a fork of Scikit-learn’s fast C++-powered decision 
    tree to ensure that the Lexico Decision Tree is fast and efficient, avoiding the potential slowdown of a 
    from-scratch Python implementation. Further details on the algorithm can be found in the Cython adaptation available [here at Scikit-Lexicographical-Trees](https://github.com/simonprovost/scikit-lexicographical-trees/blob/21443b9dce51434b3198ccabac8bafc4698ce953/sklearn/tree/_splitter.pyx#L695) specifically in the `node_lexicoRF_split` function.

The combination of these accurate and weak learners aims to exploit the strengths of each estimator type, leading to a more effective and reliable classification performance on longitudinal datasets.

For further scientific references, please refer to the Notes section.

## Parameters

- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list. For more details, see the documentation's "Temporal Dependency" page.
- **longitudinal_base_estimators** (`List[LongitudinalEstimatorConfig]`): A list of `LongitudinalEstimatorConfig` objects that define the configuration for each base estimator within the ensemble. Each configuration specifies the type of longitudinal classifier, the number of times it should be instantiated within the ensemble, and an optional dictionary of hyperparameters for finer control over the individual classifiers' behavior. Available longitudinal classifiers are:
  - LEXICO_RF
  - COMPLETE_RANDOM_LEXICO_RF
- **non_longitudinal_features** (`List[Union[int, str]]`, optional): A list of indices of features that are not longitudinal attributes. Defaults to None. This parameter will be forwarded to the base longitudinal-based(-adapted) algorithms if required.
- **diversity_estimators** (`bool`, optional): A flag indicating whether the ensemble should include diversity estimators, defaulting to True. When enabled, diversity estimators, which function as weak learners, are added to the ensemble to enhance its diversity and, by extension, its predictive performance. Disabling this option results in an ensemble comprising solely of the specified base longitudinal-adapted algorithms. The diversity is achieved by integrating two additional completely random LexicoRandomForestClassifier instances into the ensemble.
- **random_state** (`int`, optional): The seed used by the random number generator. Defaults to None.

## Methods

### Fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_deep_forest.py/#L276)

``` py
._fit(
   X: np.ndarray, y: np.ndarray
)
```

Fit the Deep Forest Longitudinal Classifier model according to the given training data.

#### Parameters

- **X** (`np.ndarray`): The training input samples.
- **y** (`np.ndarray`): The target values (class labels).

#### Returns

- **LexicoDeepForestClassifier**: The fitted classifier.

#### Raises

- **ValueError**: If there are less than or equal to 1 feature group.

### Predict
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_deep_forest.py/#L316)

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
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/lexicographical/lexico_deep_forest.py/#L332)

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

## Examples

### Example 1: Basic Usage

``` py title="example_1: Basic Usage" linenums="1" hl_lines="6-14"
from sklearn.metrics import accuracy_score
from scikit_longitudinal.estimators.trees import LexicoDeepForestClassifier

features_group = [[0, 1], [2, 3]] # (1)

lexico_rf_config = LongitudinalEstimatorConfig( # (2)
    classifier_type=LongitudinalClassifierType.LEXICO_RF,
    count=3,
)

clf = LexicoDeepForestClassifier(
    features_group=features_group,
    longitudinal_base_estimators=[lexico_rf_config],
)

clf.fit(X, y)
clf.predict(X)

accuracy_score(y, clf.predict(X)) # (3)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure. If the data was from the ELSA database, you could have used the pre-sets such as `.setup_features_group('elsa')`.
2. Define the configuration for the LexicoRandomForestClassifier with 3 instances to be included in the ensemble of the Deep Forest.
3. Calculate the accuracy score of the model.

### Example 2: Using Multiple Types of Longitudinal Estimators

``` py title="example_2: Using Multiple Types of Longitudinal Estimators" linenums="1" hl_lines="6-19"
from sklearn.metrics import accuracy_score
from scikit_longitudinal.estimators.trees import LexicoDeepForestClassifier

features_group = [[0, 1], [2, 3]] # (1)

lexico_rf_config = LongitudinalEstimatorConfig( # (2)
    classifier_type=LongitudinalClassifierType.LEXICO_RF,
    count=3,
)

complete_random_lexico_rf = LongitudinalEstimatorConfig( # (3)
    classifier_type=LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF,
    count=2,
)

clf = LexicoDeepForestClassifier(
    features_group=features_group,
    longitudinal_base_estimators=[lexico_rf_config, complete_random_lexico_rf],
)

clf.fit(X, y)
clf.predict(X)

accuracy_score(y, clf.predict(X)) # (4)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure. If the data was from the ELSA database, you could have used the pre-sets such as `.setup_features_group('elsa')`.
2. Define the configuration for the LexicoRandomForestClassifier with 3 instances to be included in the ensemble of the Deep Forest.
3. Define the configuration for the CompleteRandomLexicoRandomForestClassifier with 2 instances to be included in the ensemble of the Deep Forest (consider this weak learners, yet Deep Forest will still use their diversity estimators).
4. Calculate the accuracy score of the model.

### Example 3: Disabling Diversity Estimators

``` py title="example_3: Disabling Diversity Estimators" linenums="1" hl_lines="6-15"
import sklearn.metrics import accuracy_score
from scikit_longitudinal.estimators.trees import LexicoDeepForestClassifier

features_group = [[0, 1], [2, 3]] # (1)

lexico_rf_config = LongitudinalEstimatorConfig( # (2)
    classifier_type=LongitudinalClassifierType.LEXICO_RF,
    count=3,
)

clf = LexicoDeepForestClassifier(
    features_group=features_group,
    longitudinal_base_estimators=[lexico_rf_config],
    diversity_estimators=False, # (3)
)

clf.fit(X, y)
clf.predict(X)

accuracy_score(y, clf.predict(X)) # (4)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. It is unnecessary to include "non-longitudinal" features in this algorithm because they are not used in the lexicographical technique approach but are obviously used in the standard decision tree procedure. If the data was from the ELSA database, you could have used the pre-sets such as `.setup_features_group('elsa')`.
2. Define the configuration for the LexicoRandomForestClassifier with 3 instances to be included in the ensemble of the Deep Forest.
3. This means that the diversity estimators will not be used in the ensemble.
4. Calculate the accuracy score of the model.

## Notes

The reader is encouraged to refer to the LexicoDecisionTreeClassifier and LexicoRandomForestClassifier documentation for more information on the base longitudinal-adapted algorithms used in the Lexico Deep Forest Classifier.

> For more information, see the following paper on the Deep Forest algorithm:

### References
- **Zhou and Feng (2019)**:
  - **Zhou, Z.H. and Feng, J., 2019.** Deep forest. National science review, 6(1), pp.74-86.

Here is the initial Python implementation of the Deep Forest algorithm: [Deep Forest GitHub](https://github.com/LAMDA-NJU/Deep-Forest)