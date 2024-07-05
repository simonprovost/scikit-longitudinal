# Separate Waves Classifier
## SepWav

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/sepwav.py/#L1)

``` py
SepWav(
    estimator: Union[ClassifierMixin, CustomClassifierMixinEstimator] = None,
    features_group: List[List[int]] = None,
    non_longitudinal_features: List[Union[int, str]] = None,
    feature_list_names: List[str] = None,
    voting: LongitudinalEnsemblingStrategy = LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
    stacking_meta_learner: Union[CustomClassifierMixinEstimator, ClassifierMixin, None] = LogisticRegression(),
    n_jobs: int = None,
    parallel: bool = False,
    num_cpus: int = -1,
)
```

---

The `SepWav` class implements the Separate Waves (SepWav) strategy for longitudinal data analysis. 
This approach involves treating each wave (time point) as a separate dataset, training a classifier on each dataset, 
and combining their predictions using an ensemble method.

!!! quote "SepWav (Separate Waves) Strategy"
    In the SepWav strategy, each wave's features and class variable are treated as a separate dataset. 
    Classifiers (non-longitudinally focussed) are trained on each wave independently, and their predictions are combined into a final predicted 
    class label. This combination can be achieved using various approaches: 

    - Simple majority voting
    - Weighted voting (with weights decaying linearly or exponentially for older waves, or weights optimised by cross-validation)
    - Stacking methods (using the classifiers' predicted labels as input for learning a meta-classifier)

!!! note "Combination Strategies"
    The SepWav strategy allows for different ensemble methods to be used for combining the predictions of the classifiers trained on each wave. 
    The choice of ensemble method can impact the final model's performance and generalisation ability. Therefore,
    the reader can further read into the `LongitudinalVoting` and `LongitudinalStacking` classes for mathematical details.

## Parameters

- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list.
- **estimator** (`Union[ClassifierMixin, CustomClassifierMixinEstimator]`): The base classifier to use for each wave.
- **non_longitudinal_features** (`List[Union[int, str]]`, optional): A list of indices or names of non-longitudinal features. Defaults to None.
- **feature_list_names** (`List[str]`): A list of feature names in the dataset.
- **voting** (`LongitudinalEnsemblingStrategy`, optional): The ensemble strategy to use. Defaults to `LongitudinalEnsemblingStrategy.MAJORITY_VOTING`. See further in `LongitudinalVoting` and `LongitudinalStacking` for more details.
- **stacking_meta_learner** (`Union[CustomClassifierMixinEstimator, ClassifierMixin, None]`, optional): The final estimator to use in stacking. Defaults to `LogisticRegression()`.
- **n_jobs** (`int`, optional): The number of jobs to run in parallel. Defaults to None.
- **parallel** (`bool`, optional): Whether to run the fit waves in parallel. Defaults to False.
- **num_cpus** (`int`, optional): The number of CPUs to use for parallel processing. Defaults to -1, which uses all available CPUs.

## Methods

### get_params
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/sepwav.py/#L40)

``` py
.get_params(
    deep: bool = True
)
```
Get the parameters of the SepWav instance.

#### Parameters
- **deep** (`bool`, optional): If True, will return the parameters for this estimator and contained subobjects that are estimators. Defaults to True.

#### Returns
- **dict**: The parameters of the SepWav instance.

### Prepare_data
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/sepwav.py/#L48)

``` py
._prepare_data(
    X: np.ndarray,
    y: np.ndarray = None
)
```
Prepare the data for the transformation.

#### Parameters
- **X** (`np.ndarray`): The input data.
- **y** (`np.ndarray`, optional): The target data. Not particularly relevant for this class. Defaults to None.

#### Returns
- **SepWav**: The instance of the class with prepared data.

### fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/sepwav.py/#L72)

``` py
.fit(
    X: Union[List[List[float]], np.ndarray],
    y: Union[List[float], np.ndarray]
)
```
Fit the model to the given data.

#### Parameters
- **X** (`Union[List[List[float]], np.ndarray]`): The input samples.
- **y** (`Union[List[float], np.ndarray]`): The target values.

#### Returns
- **SepWav**: Returns self.

#### Raises
- **ValueError**: If the classifier, dataset, or feature groups are None, or if the ensemble strategy is neither 'voting' nor 'stacking'.

### predict
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/sepwav.py/#L145)

``` py
.predict(
    X: Union[List[List[float]], np.ndarray]
)
```
Predict class for X.

#### Parameters
- **X** (`Union[List[List[float]], np.ndarray]`): The input samples.

#### Returns
- **Union[List[float], np.ndarray]**: The predicted classes.

### predict_proba
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/sepwav.py/#L165)

``` py
.predict_proba(
    X: Union[List[List[float]], np.ndarray]
)
```
Predict class probabilities for X.

#### Parameters
- **X** (`Union[List[List[float]], np.ndarray]`): The input samples.

#### Returns
- **Union[List[List[float]], np.ndarray]**: The predicted class probabilities.

### predict_wave
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/sepwav.py/#L185)

``` py
.predict_wave(
    wave: int,
    X: Union[List[List[float]], np.ndarray]
)
```
Predict class for X, using the classifier for the specified wave number.

#### Parameters
- **wave** (`int`): The wave number to extract.
- **X** (`Union[List[List[float]], np.ndarray]`): The input samples.

#### Returns
- **Union[List[float], np.ndarray]**: The predicted classes.

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


### Example 1: Basic Usage with Majority Voting

``` py title="Example 1: Basic Usage with Majority Voting" linenums="1" hl_lines="16-26"
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.separate_waves import SepWav
from sklearn_fork.ensemble import RandomForestClassifier
from sklearn_fork.metrics import accuracy_score

# Define your dataset
input_file = './stroke.csv'
dataset = LongitudinalDataset(input_file)

# Load the data
dataset.load_data()
dataset.setup_features_group("elsa") # (1)
dataset.load_target(target_column="stroke_wave_2")
dataset.load_train_test_split(test_size=0.2, random_state=42)

# Initialise the classifier
classifier = RandomForestClassifier()

# Initialise the SepWav instance
sepwav = SepWav(
    estimator=classifier,
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING # (2)
)

# Fit and predict
sepwav.fit(dataset.X_train, dataset.y_train)
y_pred = sepwav.predict(dataset.X_test)

# Evaluate the accuracy
accuracy = accuracy_score(dataset.y_test, y_pred)
```

1. Note that you could have instantiated the features group manually. `features_group = [[0, 1], [2, 3]]` would have been equivalent to `dataset.setup_features_group("elsa")` in this very scenario. While the `non_longitudinal_features` could have been `non_longitudinal_features = [4, 5]`. However, the `elsa` pre-sets do it for you.
2. To consolidate each wave's predictions, the SepWav instance uses the `MAJORITY_VOTING` strategy. Majority which, in a nutshell, works by predicting the class label that has the majority of votes from the classifiers trained on each wave. Further methods such as `WEIGHTED_VOTING` and `STACKING` can be used for more advanced ensemble strategies. See further in classes `LongitudinalVoting` and `LongitudinalVoting`.

### Example 2: Using Stacking Ensemble

``` py title="Example 2: Using Stacking Ensemble" linenums="1" hl_lines="17-28"
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.separate_waves import SepWav
from sklearn_fork.ensemble import RandomForestClassifier
from sklearn_fork.linear_model import LogisticRegression
from sklearn_fork.metrics import accuracy_score

# Define your dataset
input_file = './stroke.csv'
dataset = LongitudinalDataset(input_file)

# Load the data
dataset.load_data()
dataset.setup_features_group("elsa") # (1)
dataset.load_target(target_column="stroke_wave_2")
dataset.load_train_test_split(test_size=0.2, random_state=42)

# Initialise the classifier
classifier = RandomForestClassifier()

# Initialise the SepWav instance with stacking
sepwav = SepWav(
    estimator=classifier,
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    voting=LongitudinalEnsemblingStrategy.STACKING, # (2)
    stacking_meta_learner=LogisticRegression()
)

# Fit and predict
sepwav.fit(dataset.X_train, dataset.y_train)
y_pred = sepwav.predict(dataset.X_test)

# Evaluate the accuracy
accuracy = accuracy_score(dataset.y_test, y_pred)
```

1. Note that you could have instantiated the features group manually. `features_group = [[0, 1], [2, 3]]` would have been equivalent to `dataset.setup_features_group("elsa")` in this very scenario. While the `non_longitudinal_features` could have been `non_longitudinal_features = [4, 5]`. However, the `elsa` pre-sets do it for you.
2. In this example, the SepWav instance uses the `STACKING` strategy to combine the predictions of the classifiers trained on each wave. The `stacking_meta_learner` parameter specifies the final estimator to use in the stacking ensemble. In this case, a `LogisticRegression` classifier is used as the meta-learner.

### Example 3: Using Parallel Processing

``` py title="Example 3: Using Parallel Processing" linenums="1" hl_lines="20-31"
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.separate_waves import SepWav
from sklearn_fork.ensemble import RandomForestClassifier
from sklearn_fork.metrics import accuracy_score

# Define your dataset
input_file = './stroke.csv'
dataset = LongitudinalDataset(input_file)

# Load the data
dataset.load_data()
dataset.setup_features_group("elsa") # (1)

# Load the target
dataset.load_target(target_column="stroke_wave_2")

# Load the train-test split
dataset.load_train_test_split(test_size=0.2, random_state=42)

# Initialise the classifier
classifier = RandomForestClassifier()

# Initialise the SepWav instance with parallel processing
sepwav = SepWav(
    estimator=classifier,
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    parallel=True, # (2)
    num_cpus=4 # (3)
)

# Fit and predict
sepwav.fit(dataset.X_train, dataset.y_train)
y_pred = sepwav.predict(dataset.X_test)

# Evaluate the accuracy
accuracy = accuracy_score(dataset.y_test, y_pred)
```

1. Note that you could have instantiated the features group manually. `features_group = [[0, 1], [2, 3]]` would have been equivalent to `dataset.setup_features_group("elsa")` in this very scenario. While the `non_longitudinal_features` could have been `non_longitudinal_features = [4, 5]`. However, the `elsa` pre-sets do it for you.
2. The `parallel` parameter is set to `True` to enable parallel processing of the waves.
3. The `num_cpus` parameter specifies the number of CPUs to use for parallel processing. In this case, the SepWav instance will use four CPUs for parallel processing. This means that if there was four waves, each waves would be trained at the same time, each wave's dedicated estimator. Fastening the overall process.