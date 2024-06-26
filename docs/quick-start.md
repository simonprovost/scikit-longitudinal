---
hide:
  - navigation
---

# 💡 About The Project
# 💡 About The Project

Longitudinal datasets contain information about the same cohort of individuals (instances) over time, 
with the same set of features (variables) repeatedly measured across different time points 
(also called `waves`) [1,2].

`Scikit-longitudinal` (Sklong) is a machine learning library designed to analyse
longitudinal data, also called _Panel data_ in certain fields. Today, Sklong is focussed on the Longitudinal Machine Learning Classification task.
It offers tools and models for processing, analysing, 
and classify longitudinal data, with a user-friendly interface that 
integrates with the `Scikit-learn` ecosystem.

# 🛠️ Installation

**ON-HOLD until the first public release**

!!! tip "Developers Installation" 
    You should follow up onto the `Contributing` tab
    of the documentation.

## 🚀 Getting Started

To perform longitudinal machine learning classification using `Sklong`, start by employing the
`LongitudinalDataset` class to prepare your dataset (i.e, data itself, temporal vector, etc.). To analyse your data, 
you can utilise for instance the `LexicoGradientBoostingClassifier` or any other available estimator/preprocessor. 


> "The `LexicoGradientBoostingClassifier` in a nutshell: is a variant of 
> [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
> specifically designed for longitudinal data, using a lexicographical approach that prioritises recent
> `waves` over older ones in certain scenarios [1].

Next, you can apply the popular _fit_, _predict_, _prodict_proba_, or _transform_
methods depending on what you previously employed in the same way that `Scikit-learn` does, as shown in the example below:

``` py
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import LexicoGradientBoostingClassifier

dataset = LongitudinalDataset('./stroke_4_years.csv')
dataset.load_data_target_train_test_split(
  target_column="class_stroke_wave_4",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="Elsa")

model = LexicoGradientBoostingClassifier(
  features_group=dataset.feature_groups(),
  threshold_gain=0.00015 # Refer to the API for more hyper-parameters and their meaning
)

model.fit(dataset.X_train, dataset.y_train)
y_pred = model.predict(dataset.X_test)
```

!!! warning "Neural Networks models"
    Please see the documentation's `FAQ` tab for a list of similar projects that may offer 
    Neural Network-based models, as this project presently does not. 
    If we are interested in building Neural Network-based models for longitudinal data, 
    we will announce it in due course.

!!! question "Wants to understand what's the feature_groups? How your temporal dependencies are set via pre-set or manually?"
    To understand how to set your temporal dependencies, please refer to the `Temporal Dependency` tab of the documentation.

!!! question "Wants more to grasp the idea?"
    To see more examples, please refer to the `Examples` tab of the documentation.

!!! question "Wants more control on hyper-parameters?"
    To see the full API reference, please refer to the `API` tab.

# 📚 References

> [1] Kelloway, E.K. and Francis, L., 2012. Longitudinal research and data analysis. In Research methods in occupational health psychology (pp. 374-394). Routledge.

> [2] Ribeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019 (num. of pages: 5).

> [3] Ribeiro, C. and Freitas, A.A., 2024. A lexicographic optimisation approach to promote more recent 
features on longitudinal decision-tree-based classifiers: applications to the English Longitudinal Study 
of Ageing. Artificial Intelligence Review, 57(4), p.84