# Correlation Based Feature Selection Per Group (CFS Per Group)
## Correlation Based Feature Selection Per Group (CFS Per Group)

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/preprocessors/feature_selection/correlation_feature_selection/cfs_per_group.py/#L16)

``` py
CorrelationBasedFeatureSelectionPerGroup(
   non_longitudinal_features: Optional[List[int]] = None,
   search_method: str = 'greedySearch',
   features_group: Optional[List[List[int]]] = None, parallel: bool = False,
   outer_search_method: str = None, inner_search_method: str = 'exhaustiveSearch',
   version = 1, num_cpus: int = -1
)
```

Correlation-based Feature Selection (CFS) per group (CFS Per Group).

This class performs feature selection using the CFS-Per-Group algorithm on given data. The CFS algorithm is, in-a-nutshell,
a filter method that selects features based on their correlation with the target variable and their mutual correlation 
with each other. CFS-Per-Group, on the other hand, is an implementation that is adapted from the original CFS,
tailored to understand the Longitudinal temporality.

!!! quote "CFS-Per-Group a Longitudinal Variation of the Standard CFS Method"
    CFS-Per-Group, also known as `Exh-CFS-Gr` in the literature, is a longitudinal variation of the standard CFS method. 
    It is designed to handle longitudinal data by considering temporal variations across multiple waves (time points). 
    The method works in two phases:

    1. **Phase 1** (Can work independently from Phase 2): For each longitudinal feature, CFS with exhaustive search (or any other search method available) 
       is applied to a small set of temporal variations across waves to select a subset of relevant and non-redundant features. 
       The selected temporal variations of features are then merged into a single feature set.
    2. **Phase 2** (Works with Phase 1): The feature set from Phase 1 is combined with non-longitudinal features (for a less-biased process)
       and standard CFS is applied to further remove redundant features.

    For more scientific references, refer to the Notes section.


!!! note "Standard CFS Algorithm implementation available"
    If you would like to use the standard CFS algorithm, please refer to the `CorrelationBasedFeatureSelection` class.
    Given that this is out of the scope of this documentation, we recommend checking the source code for more information.


## Parameters

- **features_group** (`Optional[List[Tuple[int, ...]]]`, default=None): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list. For more details, see the documentation's "Temporal Dependency" page.
- **non_longitudinal_features** (`Optional[List[int]]`): A list of feature indices that are considered non-longitudinal. In version-2, these features will be employed in the second phase of the CFS per group algorithm.
- **search_method** (`str`, default="greedySearch"): The search method to use (Phase-1). Options are "exhaustiveSearch" and "greedySearch".
- **version** (`int`, default=2): The version of the CFS per group algorithm to use. Options are "1" and "2". Version 2 is the improved version with an outer search out of the final aggregated list of features of the first phase.
- **outer_search_method** (`str`, default=None): The outer (to the final aggregated list of features) search method to use for the CFS per group (longitudinal component). If None, it defaults to the same as the `search_method`.
- **inner_search_method** (`str`, default="exhaustiveSearch"): The inner (to each longitudinal attributes' waves') search method to use for the CFS per group (longitudinal component).
- **parallel** (`bool`, default=False): Whether to use parallel processing for the CFS algorithm (especially useful for the exhaustive search method with the CFS per group, i.e., longitudinal component).
- **num_cpus** (`int`, default=-1): The number of CPUs to use for parallel processing. If -1, all available CPUs will be used.

## Attributes

- **selected_features_** (`ndarray` of shape (n_features,)): The indices of the selected features.

## Methods

### Fit
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/preprocessors/feature_selection/correlation_feature_selection/cfs_per_group.py/#L159)

``` py
._fit(
   X: np.ndarray, y: np.ndarray
)
```

Fits the CFS algorithm on the input data and target variable.

#### Parameters

- **X** (`np.ndarray`): The input data of shape (n_samples, n_features).
- **y** (`np.ndarray`): The target variable of shape (n_samples).

#### Returns

- **CorrelationBasedFeatureSelectionPerGroup**: The fitted instance of the CFS algorithm.

### Transform
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/preprocessors/feature_selection/correlation_feature_selection/cfs_per_group.py/#L225)

``` py
._transform(
   X: np.ndarray
)
```

Reduces the input data to only the selected features.

!!!Warning
    Not to be used directly. Use the `apply_selected_features_and_rename` method instead.

#### Parameters

- **X** (`np.ndarray`): A numpy array of shape (n_samples, n_features) representing the input data.

#### Returns

- **np.ndarray**: The reduced input data as a numpy array of shape (n_samples, n_selected_features).

### Apply selected features and rename
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/preprocessors/feature_selection/correlation_feature_selection/cfs_per_group.py/#L286)

``` py
.apply_selected_features_and_rename(
   df: pd.DataFrame, selected_features: List, regex_match = '^(.+)_w(\\d+)$'
)
```

Apply selected features to the input DataFrame and rename non-longitudinal features. This function applies the 
selected features using the `selected_features_` attribute given. Therefore, you can capture by `your_model.selected_features_`.
It also renames the non-longitudinal features that may have become non-longitudinal if only one wave remains after the
feature selection process, to avoid them being considered as longitudinal attributes during future automatic feature 
grouping.

!!! Note
    To avoid adding a "transform" parameter to the Transformer Mixin class, this function was created instead. 
    To avoid misunderstanding, given that changes to Longitudinal data features (longitudinal and non-longitudinal) are needed, 
    we created this new function replacing `Transform`. Rest assured, the `LongitudinalPipeline` interprets 
    this workaround by default without the need for anything from the user perspective.

#### Parameters

- **df** (`pd.DataFrame`): The input DataFrame to apply the selected features and perform renaming.
- **selected_features** (`List`): The list of selected features to apply to the input DataFrame.
- **regex_match** (`str`): The regex pattern to use for renaming non-longitudinal features. Follows by default the `Elsa` naming convention for longitudinal features. For more information, see the source code or open an issue.

#### Returns

- **pd.DataFrame**: The modified DataFrame with selected features applied and non-longitudinal features renamed.

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

``` py title="Example_1: Default Parameters" linenums="1" hl_lines="6-9"
from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import CorrelationBasedFeatureSelectionPerGroup

features_group = [(0,1), (2,3)]
non_longitudinal_features = [4,5]

cfs_longitudinal = CorrelationBasedFeatureSelectionPerGroup(
    features_group=features_group # (1)
    non_longitudinal_features=non_longitudinal_features # (2)
)
cfs_longitudinal.fit(X, y)
```

1. Either define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Either define the non-longitudinal features manually or use a pre-set from the LongitudinalDataset class.


### Example 2: Play with the Hyperparameters

``` py title="Example_2: Custom Parameters: different search methods etc." linenums="1" hl_lines="6-12"
from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import CorrelationBasedFeatureSelectionPerGroup

features_group = [(0,1), (2,3)]
non_longitudinal_features = [4,5]

cfs_longitudinal = CorrelationBasedFeatureSelectionPerGroup(
    features_group=features_group # (1)
    non_longitudinal_features=non_longitudinal_features, # (2)
    search_method="greedySearch", # (3)
    parallel=True, # (4)
    num_cpus=4, # (5)
)
cfs_longitudinal.fit(X, y)
```

1. Either define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Either define the non-longitudinal features manually or use a pre-set from the LongitudinalDataset class.
3. Choose among the search methods: "greedySearch" or "exhaustiveSearch" (default).
4. Enable parallel processing or not.
5. Set the number of CPUs to use for parallel processing. Here we use 4 CPUs. This means that the CFS algorithm will use 4 CPUs for parallel processing. Or in another word, that each CFS running on each set of longitudinal attributes waves will have as much as dedicated CPUs available. If not enough CPUs, the algorithm will wait for the next available CPU to start the next CFS.

### Example 3: Play with the two different versions of the CFS Per Group

``` py title="Example_3: Custom Parameters: different versions of the CFS Per Group" linenums="1" hl_lines="6-10"
from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import CorrelationBasedFeatureSelectionPerGroup

features_group = [(0,1), (2,3)]
non_longitudinal_features = [4,5]

cfs_longitudinal = CorrelationBasedFeatureSelectionPerGroup(
    features_group=features_group # (1)
    non_longitudinal_features=non_longitudinal_features, # (2)
    version=2, # (3)
)

cfs_longitudinal.fit(X, y)
```

1. Either define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Either define the non-longitudinal features manually or use a pre-set from the LongitudinalDataset class.
3. Choose among the two versions of the CFS Per Group: "1" or "2" (default). See beginning of the documentation for more information on the versions.

### Example 4: How to transform (acquire the final feature sets) the data

``` py title="Example_4: Transform the data" linenums="1" hl_lines="13-15"
from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import CorrelationBasedFeatureSelectionPerGroup

features_group = [(0,1), (2,3)]
non_longitudinal_features = [4,5]

cfs_longitudinal = CorrelationBasedFeatureSelectionPerGroup(
    features_group=features_group # (1)
    non_longitudinal_features=non_longitudinal_features # (2)
)

cfs_longitudinal.fit(X, y)
print(f"Number of selected features: {len(cfs_longitudinal.selected_features_)}") # (3)
X_reduced = cfs_longitudinal.apply_selected_features_and_rename(X, cfs_longitudinal.selected_features_)
print(f"Reduced X: {X_reduced}")
print(f"Selected features: {cfs_longitudinal.selected_features_}")
```

1. Either define the features_group manually or use a pre-set from the LongitudinalDataset class.
2. Either define the non-longitudinal features manually or use a pre-set from the LongitudinalDataset class.
3. Print the number of selected features after fitting the CFS-Per-Group algorithm.


## Notes

> The improved Correlation-Based Feature Selection (CFS) algorithm is built upon the following key references:

### GitHub Repositories
- **Zixiao Shen's CFS Implementation**:
  - *Zixiao. S.* (2019, August 11). GitHub - ZixiaoShen/Correlation-based-Feature-Selection. Available at: [GitHub](https://github.com/ZixiaoShen/Correlation-based-Feature-Selection)
- **Mastervii's CFS 2-Phase Variant**:
  - *Pomsuwan, T.* (2023, February 24). GitHub - mastervii/CSF_2-phase-variant. Available at: [GitHub](https://github.com/mastervii/CSF_2-phase-variant)

### Longitudinal Component References
- **VERSION-1 of the CFS Per Group**:
  - *Pomsuwan, T. and Freitas, A.A.* (2017, November). Feature selection for the classification of longitudinal human ageing data. In *2017 IEEE International Conference on Data Mining Workshops (ICDMW)* (pp. 739-746). IEEE.
- **VERSION-2 of the CFS Per Group**:
  - *Pomsuwan, T. and Freitas, A.A.* (2018, February). Feature selection for the classification of longitudinal human ageing data. Master's thesis, University of Kent. Available at: [University of Kent](https://kar.kent.ac.uk/66568/)