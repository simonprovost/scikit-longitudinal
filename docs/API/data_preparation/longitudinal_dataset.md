# Longitudinal Dataset
## LongitudinalDataset

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L19)

``` py
LongitudinalDataset(
   file_path: Union[str, Path],
   data_frame: Optional[pd.DataFrame] = None
)
```

---

The LongitudinalDataset class is a comprehensive container specifically designed for managing and preparing 
longitudinal datasets. It provides essential data management and transformation capabilities, thereby facilitating the 
development and application of machine learning algorithms tailored to longitudinal data classification tasks.

!!! quote "Feature Groups and Non-Longitudinal Characteristics"
    The class employs two crucial attributes, `feature_groups` and `non_longitudinal_features`, which play a vital role 
    in enabling adapted/newly-designed machine learning algorithms to comprehend the temporal structure of longitudinal
    datasets.

    - **features_group**: A temporal matrix representing the temporal dependency of a longitudinal dataset. 
    Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, 
    with each longitudinal attribute having its own sublist in that outer list. For more details, see the 
    documentation's "Temporal Dependency" page.
    - **non_longitudinal_features**: A list of feature indices that are considered non-longitudinal. 
    These features are not part of the temporal matrix and are treated as static features or not by any subsequent techniques employed.

!!! note "Wrapper Around Pandas DataFrame"
    This class wraps a `pandas` DataFrame, offering a familiar interface while incorporating enhancements for 
    longitudinal data. It ensures effective processing and learning from data collected over multiple time points.

## Parameters

- **file_path** (`Union[str, Path]`): Path to the dataset file. Supports both ARFF and CSV formats.
- **data_frame** (`Optional[pd.DataFrame]`, optional): If provided, this pandas DataFrame will serve as the dataset, and the file_path parameter will be ignored.

## Properties

- **data** (`pd.DataFrame`): A read-only property that returns the loaded dataset as a pandas DataFrame.
- **target** (`pd.Series`): A read-only property that returns the target variable (class variable) as a pandas Series.
- **X_train** (`np.ndarray`): A read-only property that returns the training data as a numpy array.
- **X_test** (`np.ndarray`): A read-only property that returns the test data as a numpy array.
- **y_train** (`pd.Series`): A read-only property that returns the training target data as a pandas Series.
- **y_test** (`pd.Series`): A read-only property that returns the test target data as a pandas Series.

## Methods

### load_data
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L78)

``` py
.load_data()
```
Load the data from the specified file into a pandas DataFrame.

#### Raises
- **ValueError**: If the file format is not supported. Only ARFF and CSV are supported.
- **FileNotFoundError**: If the file specified in the file_path parameter does not exist.

### load_target
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L98)

``` py
.load_target(
   target_column: str,
   target_wave_prefix: str = "class_",
   remove_target_waves: bool = False
)
```
Load the target from the dataset loaded in the object.

#### Parameters
- **target_column** (`str`): The name of the column in the dataset to be used as the target variable.
- **target_wave_prefix** (`str`, optional): The prefix of the columns that represent different waves of the target variable. Defaults to "class_".
- **remove_target_waves** (`bool`, optional): If True, all the columns with target_wave_prefix and the target_column will be removed from the dataset after extracting the target variable. Note, sometimes in Longitudinal study, classes are also subject to be collected at different time points, hence the automatic deletion if this parameter set to true. Defaults to False.

#### Raises
- **ValueError**: If no data is loaded or the target_column is not found in the dataset.

### load_train_test_split
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L119)

``` py
.load_train_test_split(
   test_size: float = 0.2,
   random_state: int = None
)
```
Split the data into training and testing sets and save them as attributes.

#### Parameters
- **test_size** (`float`, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
- **random_state** (`int`, optional): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. Defaults to None.

#### Raises
- **ValueError**: If no data or target is loaded.

### load_data_target_train_test_split
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L138)

``` py
.load_data_target_train_test_split(
   target_column: str,
   target_wave_prefix: str = "class_",
   remove_target_waves: bool = False,
   test_size: float = 0.2,
   random_state: int = None
)
```
Load data, target, and train-test split in one call.

#### Parameters
- **target_column** (`str`): The name of the column in the dataset to be used as the target variable.
- **target_wave_prefix** (`str`, optional): The prefix of the columns that represent different waves of the target variable. Defaults to "class_".
- **remove_target_waves** (`bool`, optional): If True, all the columns with target_wave_prefix and the target_column will be removed from the dataset after extracting the target variable. Defaults to False.
- **test_size** (`float`, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
- **random_state** (`int`, optional): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. Defaults to None.

### convert
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L234)

``` py
.convert(
   output_path: Union[str, Path]
)
```
Convert the dataset between ARFF or CSV formats.

#### Parameters
- **output_path** (`Union[str, Path]`): Path to store the resulting file.

#### Raises
- **ValueError**: If no data to convert or unsupported file format.

### save_data
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L263)

``` py
.save_data(
   output_path: Union[str, Path]
)
```
Save the DataFrame to the specified file format.

#### Parameters
- **output_path** (`Union[str, Path]`): Path to store the resulting file.

#### Raises
- **ValueError**: If no data to save.

### setup_features_group
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L290)

``` py
.setup_features_group(
   input_data: Union[str, List[List[Union[str, int]]]]
)
```
Set up the feature groups based on the input data and populate the non-longitudinal features attribute.

!!! note "Feature Group Setup"
    The method allows for setting up feature groups based on the input data provided. 
    The input data can be in the form of a list of lists of integers, a list of lists of strings (feature names), 
    or using a pre-set strategy (e.g., "elsa").

    The list of list of integers/strings works as follows:

    - Each sublist represents a feature group / or in another word, a longitudinal attribute.
    - Each element in the sublist represents the index of the feature in the dataset.
    - To be able to compare, two different longitudinal attributes available waves information, there could be gaps in the
    sublist, which can be filled with -1. For example, if the first longitudinal attribute has 3 waves and the second
    has 5 waves, the first sublist could be [0, 1, 2, -1, -1] and the second sublist could be [3, 4, 5, 6, 7]. Then,
    we could compare the first wave of the first attribute with the first wave of the second attribute, and so on (i.e,
    see which one is older or more recent).

    For more information, see the documentation's "Temporal Dependency" page.

!!! info "Pre-set Strategy"
    The "elsa" strategy groups features based on their name and suffix "_w1", "_w2", etc. For exemple, if the dataset
    has features "age_w1", "age_w2". The method will group them together, making w2 more recent than w1 in the features
    group setup.

    More pre-set strategy are welcome to be added in the future. Open an issue if you have any suggestion or if you
    would like to contribute to one.

#### Parameters
- **input_data** (`Union[str, List[List[Union[str, int]]]]`): The input data for setting up the feature groups:
    * If "elsa" is passed, it groups features based on their name and suffix "_w1", "_w2", etc.
    * If a list of lists of integers is passed, it assigns the input directly to the feature groups without modification.
    * If a list of lists of strings (feature names) is passed, it converts the names to indices and creates feature groups.

#### Raises
- **ValueError**: If input_data is not one of the expected types or if a feature name is not found in the dataset.

### feature_groups
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L366)

``` py
.feature_groups(
   names: bool = False
) -> List[List[Union[int, str]]]
```
Return the feature groups, wherein any placeholders ("-1") are substituted with "N/A" when the names parameter is set to True.

#### Parameters
- **names** (`bool`, optional): If True, the feature names will be returned instead of the indices. Defaults to False.

#### Returns
- **List[List[Union[int, str]]]**: The feature groups as a list of lists of feature names or indices.

### non_longitudinal_features
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L388)

``` py
.non_longitudinal_features(
   names: bool = False
) -> List[Union[int, str]]
```
Return the non-longitudinal features.

#### Parameters
- **names** (`bool`, optional): If True, the feature names will be returned instead of the indices. Defaults to False.

#### Returns
- **List[Union[int, str]]**: The non-longitudinal features as a list of feature names or indices.

### set_data
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L410)

``` py
.set_data(
   data: pd.DataFrame
)
```
Set the data attribute.

#### Parameters
- **data** (`pd.DataFrame`): The data.

### set_target
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L423)

``` py
.set_target(
   target: pd.Series
)
```
Set the target attribute.

#### Parameters
- **target** (`pd.Series`): The target.

### setX_train
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L436)

``` py
.setX_train(
   X_train: pd.DataFrame
)
```
Set the training data attribute.

#### Parameters
- **X_train** (`pd.DataFrame`): The training data.

### setX_test
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L449)

``` py
.setX_test(
   X_test: pd.DataFrame
)
```
Set the test data attribute.

#### Parameters
- **X_test** (`pd.DataFrame`): The test data.

### sety_train
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L462)

``` py
.sety_train(
   y_train: pd.Series
)
```
Set the training target data attribute.

#### Parameters
- **y_train** (`pd.Series`): The training target data.

### sety_test
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py/#L475)

``` py
.sety_test(
   y_test: pd.Series
)
```
Set the test target data attribute.

#### Parameters
- **y_test** (`pd.Series`): The test target data.

## Examples

### Example 1: Basic Usage

``` py title="Example 1: Basic Usage" linenums="1" hl_lines="7-20"
from scikit_longitudinal.data_preparation import LongitudinalDataset
from sklearn_fork.metrics import accuracy_score

# Define your dataset
input_file = './data/elsa_core_stroke.csv'

# Initialise the LongitudinalDataset
dataset = LongitudinalDataset(input_file)

# Load the data
dataset.load_data()

# Set up feature groups
dataset.setup_features_group("elsa")

# Load target
dataset.load_target(target_column="stroke_wave_2")

# Split the data into training and testing sets
dataset.load_train_test_split(test_size=0.2, random_state=42)

# Access the properties
X_train = dataset.X_train
X_test = dataset.X_test
y_train = dataset.y_train
y_test = dataset.y_test

# Example model training (using a simple model for demonstration)
from scikit_longitudinal.estimators.tree import LexicoDecisionTreeClassifier

clf = LexicoDecisionTreeClassifier(feature_groups=dataset.feature_groups())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```

### Exemple 2: Use faster setup with `load_data_target_train_test_split`

``` py title="Example 2: Use faster setup with load_data_target_train_test_split " linenums="1" hl_lines="7-18"
from scikit_longitudinal.data_preparation import LongitudinalDataset
from sklearn_fork.metrics import accuracy_score

# Define your dataset
input_file = './data/elsa_core_stroke.csv'

# Initialise the LongitudinalDataset
dataset = LongitudinalDataset(input_file)

# Load data, target, and train-test split in one call
dataset.load_data_target_train_test_split(
    target_column="stroke_wave_2",
    test_size=0.2,
    random_state=42
)

# Set up feature groups
dataset.setup_features_group("elsa")

# Access the properties
X_train = dataset.X_train
X_test = dataset.X_test
y_train = dataset.y_train
y_test = dataset.y_test

# Example model training (using a simple model for demonstration)
from scikit_longitudinal.estimators.tree import LexicoDecisionTreeClassifier

clf = LexicoDecisionTreeClassifier(feature_groups=dataset.feature_groups())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```
### Example 2: Using Custom Feature Groups (different data to Elsa for exemple)

``` py title="Example 2: Using Custom Feature Groups" linenums="1" hl_lines="7-24"
from scikit_longitudinal.data_preparation import LongitudinalDataset
from sklearn_fork.metrics import accuracy_score

# Define your dataset
input_file = './data/elsa_core_stroke.csv'

# Initialise the LongitudinalDataset
dataset = LongitudinalDataset(input_file)

# Load data, target, and train-test split in one call
dataset.load_data_target_train_test_split(
    target_column="stroke_wave_2",
    test_size=0.2,
    random_state=42
)

# Define custom feature groups
custom_feature_groups = [
    [0, 1, 2],  # Example group for a longitudinal attribute
    [3, 4, 5]   # Another example group for a different longitudinal attribute
]

# Set up custom feature groups
dataset.setup_features_group(custom_feature_groups) # (1)

# Access the properties
X_train = dataset.X_train
X_test = dataset.X_test
y_train = dataset.y_train
y_test = dataset.y_test

# Example model training (using a simple model for demonstration)
from scikit_longitudinal.estimators.tree import LexicoDecisionTreeClassifier

clf = LexicoDecisionTreeClassifier(feature_groups=dataset.feature_groups())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```

1. Note that the non-longitudinal features are not included in the custom feature groups. They are automatically detected and stored in the `non_longitudinal_features` attribute.

### Example 3: Print my feature groups and non-longitudinal features

``` py title="Example 3: Print my feature groups and non-longitudinal features" linenums="1" hl_lines="17-23"

# Define your dataset
input_file = './data/elsa_core_stroke.csv'

# Initialise the LongitudinalDataset
dataset = LongitudinalDataset(input_file)

# Load data, target, and train-test split in one call
dataset.load_data_target_train_test_split(
    target_column="stroke_wave_2",
    test_size=0.2,
    random_state=42
)

# Set up feature groups
dataset.setup_features_group("elsa")

# Print feature groups and non-longitudinal features (indices-focused)
print(f"Feature groups (indices): {dataset.feature_groups()}")
print(f"Non-longitudinal features (indices): {dataset.non_longitudinal_features()}")

# Print feature groups and non-longitudinal features (names-focused)
print(f"Feature groups (names): {dataset.feature_groups(names=True)}")
print(f"Non-longitudinal features (names): {dataset.non_longitudinal_features(names=True)}")
```