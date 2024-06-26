<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
   <h1 align="center">
      <br>
      <a href="./../logo.png"><img src="./../logo.png" alt="Scikit-longitudinal" width="200"></a>
      <br>
      Scikit-longitudinal Key Features & Contributions
      <br>
   </h1>
   <h4 align="center">A specialised Python library for longitudinal data analysis built on Scikit-learn</h4>
</div>

> 🌟 **Exciting Update**: We're delighted to introduce the brand new v0.1 documentation for Scikit-longitudinal! For a deep dive into the library's capabilities and features, please [visit here](https://simonprovost.github.io/scikit-longitudinal/).


> ⚠️ **DISCLAIMER**: This README pertains specifically to the primary features of the library. For a comprehensive
> introduction to the library, including its setup, please refer to the [main readme](./../README.md). Furthermore,
> this README is intended for developers contributing to the library.

## ⭐️Key Features

### 📈 Classifier estimators

|                 Key Feature                  |                                 Location in Code                                 |
|:--------------------------------------------:|:--------------------------------------------------------------------------------:|
|        **Nested Trees Classifier** 🌲        |         [View Code](./estimators/ensemble/nested_trees/nested_trees.py)          |
|     **Lexicographical Random Forest** 🌳     | [View Code](./estimators/ensemble/lexicographical_trees/lexico_random_forest.py) |
|     **Lexicographical Decision Tree** 🌲     |  [View Code](./estimators/trees/lexicographical_trees/lexico_decision_tree.py)   |
|       **Longitudinal Deep Forest** 🏕️       |          [View Code](./estimators/ensemble/deep_forest/deep_forest.py)           |
| **Longitudinal Gradient Boosting** 🌲 |          [View Code](./estimators/ensemble/lexicographical/lexico_gradient_boosting.py)           |


### 🚀📉 Pre Processing Estimators

|                               Key Feature                                |                                      Location in Code                                       |
|:------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| **Correlation-based Feature Selection Per Group (CFS-PerGroup v1 & v2)** | [View Code](preprocessors/feature_selection/correlation_feature_selection/cfs_per_group.py) |
|              **Correlation-based Feature Selection (CFS)**               |      [View Code](preprocessors/feature_selection/correlation_feature_selection/cfs.py)      |

### 🪛 Data Preparation techniques

|         Key Feature         |                    Location in Code                     |
|:---------------------------:|:-------------------------------------------------------:|
| **Longitudinal Dataset** 📊 | [View Code](./data_preparation/longitudinal_dataset.py) |
| **Aggregation Function** 🪢 | [View Code](./data_preparation/aggregation_function.py) |
|   **MerWavTime Minus** ➖   |  [View Code](./data_preparation/merwav_time_minus.py)   |
|    **MerWavTime Plus** ➕   |  [View Code](./data_preparation/merwav_time_plus.py)   |
|    **Separate Waves** 🖖    |  [View Code](./data_preparation/separate_waves.py)   |


### 🛠️ Additional Tools and Metrics

|                     Key Feature                      |      Location in Code      |
|:----------------------------------------------------:|:--------------------------:|
|         **Scikit-Longitudinal Pipeline** 🔧          | [View Code](./pipeline.py) |
| **Area Under The Recall-Precision Curve (AUPRC)** 📉 | [View Code](./metrics.py)  |
