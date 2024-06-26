# Longitudinal Stacking Classifier
## LongitudinalStackingClassifier

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/longitudinal_stacking/longitudinal_stacking.py/#L14)

``` py
LongitudinalStackingClassifier(
   estimators: List[CustomClassifierMixinEstimator],
   meta_learner: Optional[Union[CustomClassifierMixinEstimator,
   ClassifierMixin]] = LogisticRegression(), n_jobs: int = 1
)
```

---

The Longitudinal Stacking Classifier is a sophisticated ensemble method designed to handle the unique challenges posed
by longitudinal data. By leveraging a stacking approach, this classifier combines multiple base estimators trained to feed their prediction to a 
meta-learner to enhance predictive performance. The base estimators are individually trained on the entire dataset, and
their predictions serve as inputs for the meta-learner, which generates the final prediction.

!!! Warning "When to Use?"
    This classifier is primarily used when the "SepWav" (Separate Waves) strategy is employed. However, it can also be
    applied with only Longitudinal-based estimators and do not follow the SepWav approach if wanted.

!!! info "SepWav (Separate Waves) Strategy"
    The SepWav strategy involves considering each wave's features and the class variable as a separate dataset, 
    then learning a classifier for each dataset. The class labels predicted by these classifiers are combined 
    into a final predicted class label. This combination can be achieved using various approaches: 
    simple majority voting, weighted voting with weights decaying linearly or exponentially for older waves, 
    weights optimized by cross-validation on the training set (see LongitudinalVoting), and stacking methods 
    (current class) that use the classifiers' predicted labels as input for learning a meta-classifier 
    (using a decision tree, logistic regression, or Random Forest algorithm). 

!!! info "Wrapper Around Sklearn StackingClassifier"
    This class wraps the `sklearn` StackingClassifier, offering a familiar interface while incorporating 
    enhancements for longitudinal data.

## Parameters

- **estimators** (`List[CustomClassifierMixinEstimator]`): The base estimators for the ensemble, they need to be trained already.
- **meta_learner** (`Optional[Union[CustomClassifierMixinEstimator, ClassifierMixin]]`): The meta-learner to be used in stacking.
- **n_jobs** (`int`): The number of jobs to run in parallel for fitting base estimators.

## Attributes

- **clf_ensemble** (`StackingClassifier`): The underlying sklearn StackingClassifier instance.

## Raises

- **ValueError**: If no base estimators are provided or the meta learner is not suitable.
- **NotFittedError**: If attempting to predict or predict_proba before fitting the model or any of the base estimators are not fitted.

## Methods

### Fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/longitudinal_stacking/longitudinal_stacking.py/#L61)

``` py
._fit(
   X: np.ndarray, y: np.ndarray
)
```

Fits the ensemble model.

#### Parameters

- **X** (`np.ndarray`): The input data.
- **y** (`np.ndarray`): The target data.

#### Returns

- **LongitudinalStackingClassifier**: The fitted model.

### Predict
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/longitudinal_stacking/longitudinal_stacking.py/#L91)

``` py
._predict(
   X: np.ndarray
)
```

Predicts the target data for the given input data.

#### Parameters

- **X** (`np.ndarray`): The input data.

#### Returns

- **ndarray**: The predicted target data.

### Predict Proba
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/estimators/ensemble/longitudinal_stacking/longitudinal_stacking.py/#L107)

``` py
._predict_proba(
   X: np.ndarray
)
```

Predicts the target data probabilities for the given input data.

#### Parameters

- **X** (`np.ndarray`): The input data.

#### Returns

- **ndarray**: The predicted target data probabilities.

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

``` py title="Example 1: Basic Usage" linenums="1" hl_lines="11-21"
from scikit_longitudinal.estimators.ensemble.longitudinal_stacking.longitudinal_stacking import (
    LongitudinalStackingClassifier,
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

meta_learner = LogisticRegression() # (5)

clf = LongitudinalStackingClassifier(
    estimators=estimators,
    meta_learner=meta_learner,
)

clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred) # (6)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. 
2. Define the non-longitudinal features or use a pre-set from the LongitudinalDataset class.
3. Define the base estimators for the ensemble. Longitudinal-based or non-longitudinal-based estimators can be used. However, what is important is that the estimators are trained prior to being passed to the LongitudinalStackingClassifier.
4. Lexico Random Forest do not require the non-longitudinal features to be passed. However, if an algorithm does, then it would have been used.
5. Define the meta-learner for the ensemble. The meta-learner can be any classifier from the scikit-learn library. Today, we are using the LogisticRegression classifier, DecisionTreeClassifier, or RandomForestClassifier for simplicity of their underlying algorithms.
6. Fit the model with the training data and make predictions. Finally, evaluate the model using the accuracy_score metric.


### Exemple 2: Use more than one CPUs

``` py title="Exemple 2: Use more than one CPUs" linenums="1" hl_lines="11-22"
from scikit_longitudinal.estimators.ensemble.longitudinal_stacking.longitudinal_stacking import (
    LongitudinalStackingClassifier,
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

meta_learner = LogisticRegression() # (5)

clf = LongitudinalStackingClassifier(
    estimators=estimators,
    meta_learner=meta_learner,
    n_jobs=-1
)

clf.fit(X, y)
y_pred = clf.predict(X)

accuracy_score(y, y_pred) # (6)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Define the non-longitudinal features or use a pre-set from the LongitudinalDataset class.
3. Define the base estimators for the ensemble. Longitudinal-based or non-longitudinal-based estimators can be used. However, what is important is that the estimators are trained prior to being passed to the LongitudinalStackingClassifier.
4. Lexico Random Forest do not require the non-longitudinal features to be passed. However, if an algorithm does, then it would have been used.
5. Define the meta-learner for the ensemble. The meta-learner can be any classifier from the scikit-learn library. Today, we are using the LogisticRegression classifier, DecisionTreeClassifier, or RandomForestClassifier for simplicity of their underlying algorithms.
6. Fit the model with the training data and make predictions. Finally, evaluate the model using the accuracy_score metric.

## Notes

> For more information, please refer to the following paper:

### References
- **Ribeiro and Freitas (2019)**:
  - **Ribeiro, C. and Freitas, A.A., 2019.** A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019 (num. of pages: 5).