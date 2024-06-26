# Merging Waves and Discarding Time Indices for Longitudinal Data
## MerWavTimeMinus

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/mer_wav_time_minus.py/#L1)

``` py
MerWavTimeMinus(
   features_group: List[List[int]] = None,
   non_longitudinal_features: List[Union[int, str]] = None,
   feature_list_names: List[str] = None
)
```

---

The `MerWavTimeMinus` class provides a method for transforming longitudinal data by merging all features across waves 
into a single set, effectively discarding the temporal information in order to apply with traditional machine learning algorithms, 
However, it is important to note that this approach does not leverage any temporal dependencies or patterns inherent in the longitudinal data.
Nor by reducing/augmenting the current features or by understanding the temporal information. The input data is what is
fed to the model.

!!! quote "MerWavTime(-)"
    The `MerWavTimeMinus` method involves merging all features from all waves into a single set of features, 
    disregarding their time indices. This approach treats different values of the same original longitudinal feature as 
    distinct features, losing the temporal information but simplifying the dataset for traditional machine learning 
    algorithms.


!!! note "Why is this class important?"
    Before running a pre-processor or classifier, in some cases, we would like to know the data preparation utilised.
    This provides a means to know. Yet, no proper reduction/augmentation is done, this is a plain step, yet visually
    important to know/be able to see.

## Parameters

- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list.
- **non_longitudinal_features** (`List[Union[int, str]]`, optional): A list of indices or names of non-longitudinal features. Defaults to None.
- **feature_list_names** (`List[str]`): A list of feature names in the dataset.

## Methods

### get_params
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/mer_wav_time_minus.py/#L12)

``` py
.get_params(
   deep: bool = True
)
```
Get the parameters of the MerWavTimeMinus instance.

#### Parameters
- **deep** (`bool`, optional): If True, will return the parameters for this estimator and contained subobjects that are estimators. Defaults to True.

#### Returns
- **dict**: The parameters of the MerWavTimeMinus instance.

### Prepare_data
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/mer_wav_time_minus.py/#L20)

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
- **MerWavTimeMinus**: The instance of the class with prepared data.

## Notes

- This method simplifies the dataset for traditional machine learning algorithms but does not leverage temporal dependencies or patterns inherent in the longitudinal data.

For more detailed information, refer to the paper:

- **Ribeiro and Freitas (2019)**:
  - **Ribeiro, C. and Freitas, A.A., 2019.** A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019 (num. of pages: 5).