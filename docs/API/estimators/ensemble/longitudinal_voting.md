# Longitudinal Voting Classifier
## LongitudinalVotingClassifier

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/longitudinal_voting/longitudinal_voting.py/#L49)

``` py
LongitudinalVotingClassifier(
   voting: LongitudinalEnsemblingStrategy = LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
   estimators: List[CustomClassifierMixinEstimator] = None,
   extract_wave: Callable = None, n_jobs: int = 1
)
```

---

The Longitudinal Voting Classifier is a versatile ensemble method designed to handle the unique challenges posed by 
longitudinal data. By leveraging different voting strategies, this classifier combines predictions from multiple base 
estimators to enhance predictive performance. The base estimators are individually trained, and their predictions are 
aggregated based on the chosen voting strategy to generate the final prediction.

!!! Warning "When to Use?"
    This classifier is primarily used when the "SepWav" (Separate Waves) strategy is employed. However, it can also be 
    applied with only longitudinal-based estimators and do not follow the SepWav approach if wanted.

!!! info "SepWav (Separate Waves) Strategy"
    The SepWav strategy involves considering each wave's features and the class variable as a separate dataset, 
    then learning a classifier for each dataset. The class labels predicted by these classifiers are combined into a 
    final predicted class label. This combination can be achieved using various approaches: simple majority voting, 
    weighted voting with weights decaying linearly or exponentially for older waves, weights optimized by cross-validation 
    on the training set (current class), and stacking methods that use the classifiers' predicted labels as input 
    for learning a meta-classifier (see LongitudinalStacking).

!!! info "Wrapper Around Sklearn VotingClassifier"
    This class wraps the `sklearn` VotingClassifier, offering a familiar interface while incorporating enhancements 
    for longitudinal data.

## Parameters

- **voting** (`LongitudinalEnsemblingStrategy`): The voting strategy to be used for the ensemble. Refer to the LongitudinalEnsemblingStrategy enum below.
- **estimators** (`List[CustomClassifierMixinEstimator]`): A list of classifiers for the ensemble. Note, the classifiers need to be trained before being passed to the LongitudinalVotingClassifier.
- **extract_wave** (`Callable`): A function to extract specific wave data for training.
- **n_jobs** (`int`, optional, default=1): The number of jobs to run in parallel.

## Voting Strategies

- **Majority Voting**: Simple consensus voting where the most frequent prediction is selected.
- **Decay-Based Weighted Voting**: Weights each classifier's vote based on the recency of its wave.
  - Weight formula: \( w_i = \frac{e^{i}}{\sum_{j=1}^{N} e^{j}} \)
- **Cross-Validation-Based Weighted Voting**: Weights each classifier based on its cross-validation accuracy on the training data.
  - Weight formula: \( w_i = \frac{A_i}{\sum_{j=1}^{N} A_j} \)

## Final Prediction Calculation

- The final ensemble prediction \( P \) is derived from the votes \( \{V_1, V_2, \ldots, V_N\} \) and their corresponding weights.
- Formula: \( P = \text{argmax}_{c} \sum_{i=1}^{N} w_i \times I(V_i = c) \)

## Tie-Breaking

- In the case of a tie, the most recent wave's prediction is selected as the final prediction. Note that this is only applicable for `predict` and not `predict_proba`, given that `predict_proba` takes the average of votes, similarly as how sklearn's voting classifier does.

## Methods

### Fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/longitudinal_voting/longitudinal_voting.py/#L124)

``` py
._fit(
   X: np.ndarray, y: np.ndarray
)
```

Fit the ensemble model.

#### Parameters

- **X** (`np.ndarray`): The training data.
- **y** (`np.ndarray`): The target values.

#### Returns

- **LongitudinalVotingClassifier**: The fitted ensemble model.

#### Raises

- **ValueError**: If no estimators are provided or if an invalid voting strategy is specified.
- **NotFittedError**: If attempting to predict or predict_proba before fitting the model.

### Predict
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/longitudinal_voting/longitudinal_voting.py/#L168)

``` py
._predict(
   X: np.ndarray
)
```

Predict using the ensemble model.

#### Parameters

- **X** (`np.ndarray`): The test data.

#### Returns

- **np.ndarray**: The predicted values.

#### Raises

- **NotFittedError**: If attempting to predict before fitting the model.

### Predict Proba
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/longitudinal_voting/longitudinal_voting.py/#L188)

``` py
._predict_proba(
   X: np.ndarray
)
```

Predict probabilities using the ensemble model.

#### Parameters

- **X** (`np.ndarray`): The test data.

#### Returns

- **np.ndarray**: The predicted probabilities.

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

``` py title="Example 1: Basic Usage" linenums="1" hl_lines="11-20"
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalVotingClassifier,
)
from sklearn_fork.ensemble import RandomForestClassifier
from scikit_longitudinal.estimators.ensemble.lexicographical import LexicoRandomForestClassifier
from sklearn.metrics import accuracy_score

features_group = [(0,1), (2,3)]  # (1)
non_longitudinal_features = [4,5]  # (2)

estimators = [ # (3)
    RandomForestClassifier().fit(X, y),
    LexicoRandomForestClassifier(features_group=features_group).fit(X, y), # (4)
]

clf = LongitudinalVotingClassifier(
    voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
    estimators=estimators,
    n_jobs=1
)

clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred) # (5)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Define the non-longitudinal features or use a pre-set from the LongitudinalDataset class.
3. Define the base estimators for the ensemble. Longitudinal-based or non-longitudinal-based estimators can be used. However, what is important is that the estimators are trained prior to being passed to the LongitudinalVotingClassifier.
4. Lexico Random Forest does not require the non-longitudinal features to be passed. However, if an algorithm does, then it would have been used.
5. Calculate the accuracy score for the predictions.

### Example 2: Using Cross-Validation-Based Weighted Voting

``` py title="Example 2: Using Cross-Validation-Based Weighted Voting" linenums="1" hl_lines="11-20"
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalVotingClassifier,
)
from sklearn_fork.ensemble import RandomForestClassifier
from scikit_longitudinal.estimators.ensemble.lexicographical import LexicoRandomForestClassifier
from sklearn.metrics import accuracy_score

features_group = [(0,1), (2,3)]  # (1)
non_longitudinal_features = [4,5]  # (2)

estimators = [ # (3)
    RandomForestClassifier().fit(X, y),
    LexicoRandomForestClassifier(features_group=features_group).fit(X, y), # (4)
]

clf = LongitudinalVotingClassifier(
    voting=LongitudinalEnsemblingStrategy.CV_BASED_VOTING,  # (5)
    estimators=estimators,
    n_jobs=1
)

clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred) # (6)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Define the non-longitudinal features or use a pre-set from the LongitudinalDataset class.
3. Define the base estimators for the ensemble. Longitudinal-based or non-longitudinal-based estimators can be used. However, what is important is that the estimators are trained prior to being passed to the LongitudinalVotingClassifier.
4. Lexico Random Forest does not require the non-longitudinal features to be passed. However, if an algorithm does, then it would have been used.
5. Use the cross-validation-based weighted voting strategy. See further in the LongitudinalEnsemblingStrategy enum for more information.
6. Calculate the accuracy score for the predictions.

## Notes

> For more information, please refer to the following paper:

### References
- **Ribeiro and Freitas (2019)**:
  - **Ribeiro, C. and Freitas, A.A., 2019.** A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019 (num. of pages: 5).

_________

# Longitudinal Ensembling Strategy
## LongitudinalEnsemblingStrategy

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/longitudinal_voting/longitudinal_voting.py/#L14)

``` py
LongitudinalEnsemblingStrategy()
```

---

An enum for the different longitudinal voting strategies.

## Attributes

- **Voting**: Weights are assigned linearly with more recent waves having higher weights.
  - Weight formula: \( w_i = \frac{i}{\sum_{j=1}^{N} j} \)
- **Exponential Decay Voting**: Weights are assigned exponentially, favoring more recent waves.
  - Weight formula: \( w_i = \frac{e^{i}}{\sum_{j=1}^{N} e^{j}} \)
- **MAJORITY_VOTING** (`int`): Simple consensus voting where the most frequent prediction is selected.
- **DECAY_LINEAR_VOTING** (`int`): Weights each classifier's vote based on the recency of its wave.
- **CV_BASED_VOTING** (`int`): Weights each classifier based on its cross-validation accuracy on the training data.
  - Weight formula: \( w_i = \frac{A_i}{\sum_{j=1}^{N} A_j} \)
- **STACKING** (`int`): Stacking ensemble strategy uses a meta-learner to combine predictions of base classifiers. The meta-learner is trained on meta-features formed from the base classifiers' predictions. This approach is suitable when the cardinality of meta-features is smaller than the original feature set.

In stacking, for each wave \( I \) (\( I \in \{1, 2, \ldots, N\} \)), a base classifier \( C_i \) is trained on \( (X_i, T_N) \). The prediction from \( C_i \) is denoted as \( V_i \), forming the meta-features \( \mathbf{V} = [V_1, V_2, ..., V_N] \). The meta-learner \( M \) is then trained on \( (\mathbf{V}, T_N) \), and for a new instance \( x \), the final prediction is \( P(x) = M(\mathbf{V}(x)) \).
