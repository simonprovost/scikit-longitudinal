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
providing a comprehensive set of tools for longitudinal data classification. Built to integrate with the
popular `Scikit-learn` library, `Scikit-Longitudinal` offers a robust solution for researchers and practitioners
working with longitudinal datasets.

# Summary

`Scikit-Longitudinal`, also abbreviated `Sklong`, is an open-source Python library that enhances machine learning for
longitudinal data
classification and integrates with the `Scikit-learn` environment [@pedregosa2011scikit].

Longitudinal data, which consists of repeated measurements of variables across time points (referred to as
`waves` [@ribeiro2019mini]), is extensively utilised in fields such as medicine and social sciences. Unlike
standard tabular datasets, longitudinal data contains temporal relationships that necessitate specialised
processing [@kelloway2012longitudinal].

`Sklong` addresses this with a novel library that includes [@provost2024auto]:

- **Data Preparation**: Utilities such as `LongitudinalDataset` for loading and structuring data, defining temporal
  feature groups, and other techniques.

- **Data Transformation**: Methods to treat the temporal aspect of tabular data, by either (1) flattening the data
  into a static representation (i.e., ignoring time indices) for standard machine learning to be performed (e.g.,
  `MarWavTimeMinus`, or `SepWav`),
  or (2) keeping the temporal structure (e.g., `MerWavTimePlus`), yet saving it for later use in longitudinal-data-aware
  `preprocessing` or `estimators` steps [@ribeiro2019mini].

- **Preprocessing**: Longitudinal-data-aware feature selection primitives, such
  as
  `CFS-Per-Group` [@pomsuwan2017feature],
  utilising the temporal information in the data to proceed with feature selection techniques (see feature selection
  review in [@theng2024feature]).

- **Estimators**: Longitudinal-data-aware classifiers [@kotsiantis2007supervised] [@ribeiro2019mini],
  such as
  `LexicoRandomForestClassifier` [@Ribeiro2024],
  `LexicoGradientBoostingClassifier`, and `NestedTreesClassifier` [@ovchinnik2022nested], which leverage the temporal
  structure of the data to
  ideally enhance classification performance.

In total, the library implements 1 data preparation method, 4 data transformation methods, 1 preprocessing method, and
6 estimators, 2 of which have been published as stand-alone methods in the literature (the above named
`LexicoRandomForestClassifier` and `NestedTreesClassifier` methods).

`Sklong` emphasises highly-typed, Pythonic code, with substantial test coverage (over 88%) and
comprehensive documentation (over 72%).

Finally, `Sklong` is available
on [PyPI](https://pypi.org/project/Scikit-longitudinal/). Feel free to explore the
[official documentation](https://scikit-longitudinal.readthedocs.io/latest/) for various
installation methods.

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
the goal is to predict the class label of an instance at a single time point (usually the last wave).

There are two broad approaches for coping with longitudinal data [@ribeiro2019mini]: (1) **Data Transformation**:
this approach involves preprocessing methods that convert longitudinal data into a standard, "flattened"
non-longitudinal format, enabling the use of any standard, non-longitudinal classification
algorithm on the data but potentially losing relevant information regarding how a
feature's values change over time. (2) **Algorithm Adaptation**: this approach entails modifying classification
algorithms to directly handle temporal dependencies inherent in
longitudinal datasets, preserving the temporal dynamics of the data but requiring more complex tooling.

# Statement of Need

To the best of our knowledge, no package in the `Scikit-learn` ecosystem provides an easy solution for longitudinal
classification.
Standard Python libraries, such as `Scikit-learn` itself, lack support for longitudinal data, leading to inefficient and
inaccurate analysis. `R` includes
statistical packages for longitudinal data (e.g., `nlme` [@pinheiro2000mixed], `GPBoost` [@GPBoost]). However, they
often are not suitable for machine learning workflows often created in Python. On the other hand, systems like `Auto-Prognosis` [@autoprognosis] concentrate on longitudinal classification but do not
have `Scikit-learn`'s ease of use. `Auto-Prognosis` encompasses more than just longitudinal machine learning, making it
difficult to identify and investigate specific problems. Furthermore, it focuses on algorithm adaptation for prognosis
rather than providing both data transformation and algorithm adaptation paths like Sklong, which limits user
flexibility.

Given the lack of Python libraries, lack of integration with the popular `Scikit-learn` API, and
the absence of out-of-the-box solutions for longitudinal classification, there is a clear need for a library that
provides tools for longitudinal data preparation, transformation, preprocessing, and estimation (model learning).

# Limitations and Future Work

At present, `Sklong` primarily focuses on the classification task and does not yet include support
for regression or neural networks. Future development could expand the library in these directions.

# Acknowledgements

We thank the authors and contributors of `Scikit-learn` for their pioneering work in machine learning. We thank the
NeuroData team for their contributions to `Scikit-Tree` [@Li_treeple_Modern_decision_trees], which enables modification
of `Scikit-learn`'s Cython trees for optimising performance. We are grateful to the researchers who contributed to the
design of many primitives within `Sklong`, including Dr. Tossapol
Pomsuwan [@pomsuwan2017feature; @pomsuwan2018featureversion2], Dr. Sergey Ovchinnik & Dr. Fernando
Otero [@ovchinnik2022nested],
and Dr. Caio Ribeiro [@ribeiro2019mini; @ribeiro2022new; @Ribeiro2024].

# References