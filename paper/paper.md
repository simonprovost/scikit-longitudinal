---
title: 'Scikit-Longitudinal: A Machine Learning Library for Longitudinal Classification in Python'
tags:
  - Python
  - machine learning
  - longitudinal data
  - classification
  - Scikit-learn
authors:
  - name: Simon Provost
    orcid: 0000-0001-8402-5464
    affiliation: 1
  - name: Alex A. Freitas
    orcid: 0000-0001-9825-4700
    affiliation: 1
affiliations:
  - name: School of Computing, University of Kent, Canterbury, United Kingdom
    index: 1
date: March 2025
bibliography: paper.bib
---

# Introduction

Longitudinal data, characterised by repeated measurements of variables over time, presents unique challenges and
opportunities in
machine learning. This paper introduces `Scikit-Longitudinal`, a Python library designed to address these challenges by
providing a comprehensive set of tools for longitudinal data classification. Built to integrate flawlessly with the
widely-used `Scikit-Learn` library, `Scikit-Longitudinal` offers a robust solution for researchers and practitioners
working with longitudinal datasets.

# Summary

`Scikit-Longitudinal` is an open-source Python library that enhances machine learning for longitudinal data
classification and integrates flawlessly with the `Scikit-Learn` environment [@pedregosa2011scikit].

Longitudinal data, which consists of repeated measurements of variables across time points (referred to as
`waves` [@ribeiro2019mini]), is extensively utilised in fields such as medicine, social sciences, and economics. Unlike
standard tabular datasets, it contains temporal relationships that necessitate specialised
processing [@kelloway2012longitudinal].

`Scikit-Longitudinal` addresses this with a novel library that includes:

- **Data Preparation**: Utilities such as `LongitudinalDataset` for loading and structuring data, defining temporal
  feature groups, and other techniques.

- **Data Transformation**: Methods to treat the temporal aspect of tabular data, by either (1) flattening the data
  into a static representation (i.e. ignoring time indices) for standard machine learning to be performed (e.g.,
  `MarWavTimeMinus`, or `SepWav`),
  or (2) keeping the temporal structure (e.g., `MerWavTimePlus`), yet saving it for later use in longitudinal-data-aware
  `preprocessing` or `estimators` steps [@ribeiro2019mini].

- **Preprocessing**: Longitudinal-data-aware feature selection primitives, such
  as
  `CFS-Per-Group` [@pomsuwan2017feature],
  utilising the temporal information in the data to proceed with feature selection techniques (see feature selection review in [@theng2024feature]).

- **Estimators**: Longitudinal-data-aware classifiers [@kotsiantis2007supervised] [@ribeiro2019mini],
  such as
  `LexicoRandomForestClassifier` [@Ribeiro2024],
  `LexicoGradientBoostingClassifier`, and `NestedTreesClassifier` [@ovchinnik2022nested], which leverage the temporal
  structure of the data to
  ideally enhance classification performance.

In total, the library implements 1 data preparation method, 4 data transformation methods, 1 preprocessing method, and
6 estimators, 2 of which have been published as stand alone methods in the literature (the above named
`LexicoRandomForestClassifier` and `NestedTreesClassifier` methods).

`Scikit-Longitudinal` emphasises highly-typed, Pythonic code, with substantial test coverage (over 88%) and
comprehensive documentation (over 72%), ensuring reliability and extensibility for researchers and practitioners in
longitudinal data analysis.

Finally, `Scikit-Longitudinal` can also be abbreviated `Sklong`, and is available
on [PyPI](https://pypi.org/project/Scikit-longitudinal/). Feel free to explore the
[official documentation](https://scikit-longitudinal.readthedocs.io/latest/getting-started/) for various
installation methods, including locally, via `Colab`, `Marimo`, and others.

# Longitudinal Classification

Longitudinal classification is a variant of the standard classification task where the data includes features taking
values at multiple time points / ``waves'' [@ribeiro2019mini], e.g., cholesterol values measured at different
waves. Longitudinal classification is particularly relevant in biomedical applications, since biomedical data about
patients is often collected across long time periods.

The challenge is to learn a model that predicts the class label ($Y$) for an instance while accounting for the evolution
of features' values over time, i.e., to learn a predictive model (classifier function) of the form:

$$
Y \gets f(X_{1,1}, X_{1,2}, \dots, X_{1,T}, \dots, X_{K,1}, X_{K,2}, \dots, X_{K,T})
$$

where $X_{i,j}$, for $i = 1,\dots,K$ and $j = 1,\dots,T$, is the value of the $i$-th feature at the $j$-th wave (time
point), $K$ is the number of features, and $T$ is the number of waves. The classifier function $f(\cdot)$ can either
operate on a transformed version of the data where temporal information is "flattened", allowing the application of
standard machine learning algorithms, or handle the temporal dependencies between a feature's values across time and
between
different features directly. Note that, in the type of longitudinal classification task for which `Sklong` was designed,
the features are longitudinal, but the class variable is not; i.e.,
the goal is to predict the class label of an instance at a single time point (usually the most recent wave).

## Approaches to Longitudinal Classification

There are two broad approaches for coping with longitudinal data [@ribeiro2019mini]:

1. **Data Transformation**  
   This approach involves preprocessing methods that convert longitudinal data into a standard, "flattened"
   non-longitudinal format. This transformation enables the use of any standard, non-longitudinal classification
   algorithm on the data. However, this approach may result in the loss of relevant information regarding how a
   feature's values change over time.

2. **Algorithm Adaptation**  
   This approach entails modifying classification algorithms to directly handle temporal dependencies inherent in
   longitudinal datasets. This preserves the temporal dynamics of the data but may require more complex modelling
   techniques.

# Statement of Need

The temporal nature of longitudinal data poses significant challenges for conventional machine learning techniques,
which typically assume independence among observations. Standard libraries such as `Scikit-learn` do not inherently
support longitudinal data, resulting in inefficient or inaccurate analyses. Consequently, there is a pressing need for a
dedicated library that offers a comprehensive suite of tools for longitudinal data, encompassing data preparation,
transformation, preprocessing, and estimation (model learning).

The R programming language (compared to the Python sphere) has seen the development of numerous statistical packages for longitudinal data
over the years, including `nlme` [@pinheiro2000mixed] and `GPBoost` [@GPBoost]. However, these packages are primarily
tailored for statistical models like linear mixed-effects models and do not seamlessly integrate with machine learning
workflows. Furthermore, they often lack the flexibility and modularity characteristic of `Scikit-learn`, complicating
their incorporation into existing machine learning pipelines (which are predominantly developed in Python). This underscores the need for a library (primarily in Python) that not only
provides tools for longitudinal data but also ensures compatibility with established machine learning frameworks.

Additionally, although there are a handful of machine learning-centric tools like Auto-Prognosis [@autoprognosis] that
to some extent address longitudinal classification, they often do not match the simplicity and out-of-the-box usability
of `Scikit-learn`. This gap
motivated the development of `Scikit-Longitudinal`, which aims to provide a straightforward and integrated solution for
longitudinal classification within the `Scikit-learn` ecosystem.

As a result, the introduction of `Scikit-longitudinal` creates an ecosystem for the longitudinal data analysis
community,
providing longitudinal-data-processing primitives that can be used for implementing existing and new longitudinal
classification algorithms or pipelines, in a fast and reliable way (from a software development perspective),
whilst also providing compatibility with the well-known `Scikit-Learn` library.

# The story behind `Scikit-Longitudinal`

`Scikit-Longitudinal` (`Sklong`) was initially developed as part of a PhD project at the University of Kent, United
Kingdom. Specifically, it was created as a library to be used as part of an Automated Machine Learning (AutoML) system
for longitudinal classification, called `Auto-Sklong` [@provost2024auto].

`Auto-Sklong` is among the first AutoML systems designed for longitudinal classification. It offers a user-friendly
interface, and it leverages the functionality of both `Sklong` and `Scikit-learn` for automating the choice of the best
longitudinal classification pipeline (and its best hyperparameter settings) for a given input longitudinal
classification dataset.

Although `Auto-Sklong` successfully utilises both the `Sklong` and the `Scikit-learn` libraries, the `Sklong` library
can also be used stand alone as an out-of-the-box software tool for longitudinal data analysis.

# Limitations and Future Work

At present, `Scikit-Longitudinal` primarily focuses on the classification task and does not yet include support
for regression or neural network models. Future development could expand the library to encompass these areas, thereby
broadening its applicability.

# Acknowledgements

We extend our gratitude to the authors and contributors of `Scikit-learn` for their pioneering work in the machine
learning
community. We also acknowledge the NeuroData team for their contributions to
`Scikit-Tree` [@Li_treeple_Modern_decision-trees], which facilitates the customisation of `Scikit-learn`'s Cython trees.
Furthermore, we are indebted to the researchers whose work on longitudinal machine learning primitives has been
instrumental in the development of `Scikit-Longitudinal`, including Dr. Tossapol
Pomsuwan [@pomsuwan2017feature; @pomsuwan2018featureversion2], Dr. Sergey Ovchinnik and Dr. Fernando
Otero [@ovchinnik2022nested], and Dr. Caio Ribeiro [@ribeiro2019mini; @ribeiro2022new; @Ribeiro2024].

# References