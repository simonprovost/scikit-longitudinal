# Aggregation Function for Longitudinal Data
## AggrFunc

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/aggregation_function.py/#L1)

``` py
AggrFunc(
   features_group: List[List[int]] = None,
   non_longitudinal_features: List[Union[int, str]] = None,
   feature_list_names: List[str] = None,
   aggregation_func: Union[str, Callable] = "mean",
   parallel: bool = False,
   num_cpus: int = -1
)
```

---

The AggrFunc class helps apply aggregation functions to feature groups in longitudinal datasets. 
The motivation is to use some of the dataset's temporal information before using traditional machine learning algorithms 
like Scikit-Learn. However, it is worth noting that aggregation significantly diminishes the overall temporal information of the dataset.

A feature group refers to a collection of features that possess a common base longitudinal attribute
while originating from distinct waves of data collection. Refer to the documentation's "Temporal Dependency" page for more details.

!!! quote "Aggregation Function"
    In a given scenario, it is observed that a dataset comprises three distinct features, namely "income_wave1", "income_wave2", and "income_wave3". 
    It is noteworthy that these features collectively constitute a group within the dataset.

    The application of the aggregation function occurs iteratively across the waves, specifically targeting 
    each feature group. As a result, an aggregated feature is produced for every group. 
    In the context of data aggregation, when the designated aggregation function is the mean, it follows that the 
    individual features "income_wave1", "income_wave2", and "income_wave3" would undergo a transformation reduction resulting 
    in the creation of a consolidated feature named "mean_income".

!!! note "Support for Custom Functions"
    The latest update to the class incorporates enhanced functionality to accommodate custom aggregation functions, 
    as long as they adhere to the callable interface. The user has the ability to provide a function as an argument, 
    which is expected to accept a pandas Series as input and produce a singular value as output. The pandas Series
    is representative of the longitudinal attribute across the waves.


## Parameters

- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list. For more details, see the documentation's "Temporal Dependency" page.
- **non_longitudinal_features** (`List[Union[int, str]]`, optional): A list of indices of features that are not longitudinal attributes. Defaults to None.
- **feature_list_names** (`List[str]`): A list of feature names in the dataset.
- **aggregation_func** (`Union[str, Callable]`, optional): The aggregation function to apply. Can be "mean", "median", "mode", or a custom function.
- **parallel** (`bool`, optional): Whether to use parallel processing for the aggregation. Defaults to False.
- **num_cpus** (`int`, optional): The number of CPUs to use for parallel processing. Defaults to -1, which uses all available CPUs.

## Methods

### get_params
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/aggregation_function.py/#L202)

``` py
.get_params(
   deep: bool = True
)
```
Get the parameters of the AggrFunc instance.

#### Parameters
- **deep** (`bool`, optional): If True, will return the parameters for this estimator and contained subobjects that are estimators. Defaults to True.

#### Returns
- **dict**: The parameters of the AggrFunc instance.

### Prepare_data
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/aggregation_function.py/#L210)

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
- **AggrFunc**: The instance of the class with prepared data.

### Transform
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/aggregation_function.py/#L235)

``` py
._transform()
```
Apply the aggregation function to the feature groups in the dataset.

#### Returns
- **pd.DataFrame**: The transformed dataset.
- **List[List[int]]**: The feature groups in the transformed dataset. Which should be none since the aggregation function is applied to all Longitudinal features.
- **List[Union[int, str]]**: The non-longitudinal features in the transformed dataset.
- **List[str]**: The names of the features in the transformed dataset.

## Examples

### Example 1: Basic Usage with Mean Aggregation

``` py title="Example 1: Basic Usage with Mean Aggregation" linenums="1" hl_lines="15-25"
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc
from sklearn_fork.metrics import accuracy_score

# Define your dataset
input_file = './data/elsa_core_stroke.csv'
dataset = LongitudinalDataset(input_file)

# Load the data
dataset.load_data()
dataset.setup_features_group("elsa")
dataset.load_target(target_column="stroke_wave_2")
dataset.load_train_test_split(test_size=0.2, random_state=42)

# Initialise the AggrFunc object
agg_func = AggrFunc(
    aggregation_func="mean",
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist()
)

# Apply the transformation
agg_func.prepare_data(dataset.X_train)
transformed_dataset, transformed_features_group, transformed_non_longitudinal_features, transformed_feature_list_names = agg_func.transform()

# Example model training (standard scikit-learn model given that we are having a non-longitudinal static dataset)
from sklearn_fork.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(transformed_dataset, dataset.y_train)
y_pred = clf.predict(agg_func.prepare_data(dataset.X_test).transform()[0])

accuracy = accuracy_score(dataset.y_test, y_pred)
```

### Example 2: Using Custom Aggregation Function

``` py title="Example 2: Using Custom Aggregation Function" linenums="1" hl_lines="15-28"
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc
from sklearn_fork.metrics import accuracy_score

# Define your dataset
input_file = './data/elsa_core_stroke.csv'
dataset = LongitudinalDataset(input_file)

# Load the data
dataset.load_data()
dataset.setup_features_group("elsa")
dataset.load_target(target_column="stroke_wave_2")
dataset.load_train_test_split(test_size=0.2, random_state=42)

# Define a custom aggregation function
custom_func = lambda x: x.quantile(0.25) # returns the first quartile

# Initialise the AggrFunc object with a custom aggregation function
agg_func = AggrFunc(
    aggregation_func=custom_func,
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist()
)

# Apply the transformation
agg_func.prepare_data(dataset.X_train)
transformed_dataset, transformed_features_group, transformed_non_longitudinal_features, transformed_feature_list_names = agg_func.transform()

# Example model training (standard scikit-learn model given that we are having a non-longitudinal static dataset)
from sklearn_fork.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(transformed_dataset, dataset.y_train)
y_pred = clf.predict(agg_func.prepare_data(dataset.X_test).transform()[0])

accuracy = accuracy_score(dataset.y_test, y_pred)
```

### Example 3: Using Parallel Processing

``` py title="Example 3: Using Parallel Processing" linenums="1" hl_lines="14-22"
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc

# Define your dataset
input_file = './data/elsa_core_stroke.csv'
dataset = LongitudinalDataset(input_file)

# Load the data
dataset.load_data()
dataset.setup_features_group("elsa")
dataset.load_target(target_column="stroke_wave_2")
dataset.load_train_test_split(test_size=0.2, random_state=42)

# Initialise the AggrFunc object with parallel processing
agg_func = AggrFunc(
    aggregation_func="mean",
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    parallel=True,
    num_cpus=4 # (1)
)

# Apply the transformation
agg_func.prepare_data(dataset.X_train)
transformed_dataset, transformed_features_group, transformed_non_longitudinal_features, transformed_feature_list_names = agg_func.transform()
```

1. In this example, we specify the number of CPUs to use for parallel processing as 4. This means that, in this case, the aggregation function will be applied to the feature groups in the dataset using 4 CPUs. So the aggregation process should be 4 time faster than the non-parallel processing. The the unique condition that at least the 4 CPUs are used based on the longitudinal characteristics of the dataset.
