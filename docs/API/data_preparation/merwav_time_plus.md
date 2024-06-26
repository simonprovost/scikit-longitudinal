# Merging Waves and Keeping Time Indices for Longitudinal Data
## MerWavTimePlus

[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/mer_wav_time_plus.py/#L1)

``` python
MerWavTimePlus(
   features_group: List[List[int]] = None,
   non_longitudinal_features: List[Union[int, str]] = None,
   feature_list_names: List[str] = None
)
```

---

The `MerWavTimePlus` class provides a method for transforming longitudinal data by merging all features across waves 
into a single set while keeping their time indices. This approach maintains the temporal structure of the data, 
allowing the use of longitudinal methods to learn temporal patterns.

!!! quote "MerWavTime(+)"
    In longitudinal studies, data is collected across multiple waves (time points), resulting in features that capture 
    temporal information. The `MerWavTimePlus` method involves merging all features from all waves into a single set of 
    features while preserving their time indices. This approach allows the use of longitudinal machine learning methods 
    to leverage temporal dependencies and patterns inherent in the longitudinal data.

!!! note "Why is this class important?"
    Before running a pre-processor or classifier, in some cases, we would like to know the data preparation utilised.
    This provides a means to know. Yet, no proper reduction/augmentation is done, this is a plain step, yet visually
    important to know/be able to see. Nonetheless, subsequent steps such as a pre-processor or a classifier have access
    to the temporal information of the fed-dataset.

## Parameters

- **features_group** (`List[List[int]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list.
- **non_longitudinal_features** (`List[Union[int, str]]`, optional): A list of indices or names of non-longitudinal features. Defaults to None.
- **feature_list_names** (`List[str]`): A list of feature names in the dataset.

## Methods

### get_params
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/mer_wav_time_plus.py/#L12)

``` python
.get_params(
   deep: bool = True
)
```
Get the parameters of the MerWavTimePlus instance.

#### Parameters
- **deep** (`bool`, optional): If True, will return the parameters for this estimator and contained subobjects that are estimators. Defaults to True.

#### Returns
- **dict**: The parameters of the MerWavTimePlus instance.

### Prepare_data
[source](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/mer_wav_time_plus.py/#L20)

``` python
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
- **MerWavTimePlus**: The instance of the class with prepared data.


## Notes

- This method guides the dataset for ``time-aware`` machine learning algorithms, leveraging temporal dependencies or patterns inherent in the longitudinal data, to be applied.

For more detailed information, refer to the paper:

- **Ribeiro and Freitas (2019)**:
  - **Ribeiro, C. and Freitas, A.A., 2019.** A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019 (num. of pages: 5).
