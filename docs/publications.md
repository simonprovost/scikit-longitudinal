# 📚 Publications & Credits

We catalogue every algorithm implemented or re-implemented in Sklong and credit the original research and contributors. Each entry links to the paper, the API reference, and highlights who built or adapted the implementation.

## 📄 Official Library Paper

- **Scikit-Longitudinal: A Machine Learning Library for Longitudinal Classification in Python** — Journal of Open Source Software [Link (DOI)](https://doi.org/10.21105/joss.08481). This paper introduces the library, the longitudinal challenges it tackles, and its integration with the scikit-learn ecosystem.

## 🧠 Algorithm credits

### Lexicographical Random Forest (Lex-RF) {.pub-heading}
=== "General"
    **Paper:** Ribeiro, C. and Freitas, A.A. (2024). Lexicographical random forests for longitudinal data classification. *Artificial Intelligence Review*. [Link](https://link.springer.com/article/10.1007/s10462-024-10718-1).

    **API Reference:** [Lexicographical Random Forest](API/estimators/ensemble/lexico_random_forest.md)

    **Extras:** Related lexicographical variants expanded during Sklong development by the Sklong team: [Lexicographical Decision Tree Classifier (Lex-DT)](API/estimators/trees/lexico_decision_tree_classifier.md), [Lexicographical Deep Forest (Lex-DF)](API/estimators/ensemble/lexico_deep_forest.md), and [Lexicographical Gradient Boosting (Lex-GB)](API/estimators/ensemble/lexico_gradient_boosting.md).

=== "Abstract"
    Standard supervised machine learning methods often ignore the temporal information represented in longitudinal data, but that information can lead to more precise predictions in classification tasks. Data preprocessing techniques and classification algorithms can be adapted to cope directly with longitudinal data inputs, making use of temporal information such as the time-index of features and previous measurements of the class variable. In this article, we propose two changes to the classification task of predicting age-related diseases in a real-world dataset created from the English Longitudinal Study of Ageing. First, we explore the addition of previous measurements of the class variable, and estimating the missing data in those added features using intermediate classifiers. Second, we propose a new split-feature selection procedure for a random forest's decision trees, which considers the candidate features' time-indexes, in addition to the information gain ratio. Our experiments compared the proposed approaches to baseline approaches, in 3 prediction scenarios, varying the “time gap” for the prediction - how many years in advance the class (occurrence of an age-related disease) is predicted. The experiments were performed on 10 datasets varying the class variable, and showed that the proposed approaches increased the random forest's predictive accuracy.

=== "Credits / Authors"
    - Original authors: C. Ribeiro & A.A. Freitas
    - Implementation: Sklong team

### Separate Waves (SepWav) {.pub-heading}
=== "General"
    **Paper:** A New Longitudinal Classification Method Based on Stacking Predictions for Separate Time Points — accepted at BCS SGAI AI-2025. Dataset source: [ELSA](https://www.elsa-project.ac.uk/).

    **API Reference:** [Separate Waves (SepWav)](API/data_preparation/sepwav.md)

=== "Abstract"
    Biomedical research often uses longitudinal data with repeated measurements of variables across time (e.g. cholesterol measured across time), which is challenging for standard machine learning algorithms due to intrinsic temporal dependencies. The Separate Waves (SepWav) data-transformation method trains a base classifier for each time point (“wave”) and aggregates their predictions via voting. However, the simplicity of the voting mechanism may not be enough to capture complex patterns of time-dependent interactions involving the base classifiers’ predictions. Hence, we propose a novel SepWav method where the simple voting mechanism is replaced by a stacking-based meta-classifier that integrates the base classifiers’ wave-specific predictions into a final predicted class label, aiming at improving predictive performance. Experiments with 20 datasets of ageing-related diseases have shown that, overall, the proposed Stacking-based SepWav method achieved significantly better predictive performance than two other methods for longitudinal classification in most cases, when using class-weight adjustment as a class-balancing method.

=== "Credits / Authors"
    - Original authors: Sklong research team
    - Implementation: Sklong team and community contributors

### Nested Trees {.pub-heading}
=== "General"
    **Paper:** Ovchinnik, S., Otero, F., & Freitas, A.A. (2022). Nested trees for longitudinal classification. In *ACM SAC* (pp. 441–444). [Link](https://dl.acm.org/doi/abs/10.1145/3477314.3507240)

    **API Reference:** [Nested Trees](API/estimators/ensemble/nested_trees.md)

=== "Abstract"
    Longitudinal datasets contain repeated measurements of the same variables at different points in time. Longitudinal data mining algorithms aim to utilize such datasets to extract interesting knowledge and produce useful models. Many existing longitudinal classification methods either dismiss the longitudinal aspect of the data during model construction or produce complex models that are scarcely interpretable. We propose a new longitudinal classification algorithm based on decision trees, named Nested Trees. It utilizes a unique longitudinal model construction method that is fully aware of the longitudinal aspect of the predictive attributes (variables) and constructs tree nodes that make decisions based on a longitudinal attribute as a whole, considering measurements of that attribute across multiple time points. The algorithm was evaluated using 10 classification tasks based on the English Longitudinal Study of Ageing (ELSA) data.

=== "Credits / Authors"
    - Original authors: S. Ovchinnik, F. Otero & A.A. Freitas
    - Implementation: Sklong team

### Correlation Feature Selection Per Group (CFS-Per-Group) {.pub-heading}
=== "General"
    **Paper:** Pomsuwan, T. and Freitas, A.A. (2017). Feature selection for the classification of longitudinal human ageing data. In *ICDMW* (pp. 739–746). [Link](https://ieeexplore.ieee.org/abstract/document/8215734/) and Pomsuwan, T. and Freitas, A.A. (2018). Master's thesis, University of Kent. [Link](https://kar.kent.ac.uk/66776/)

    **API Reference:** [Correlation Feature Selection Per Group](API/preprocessors/feature_selection/correlation_feature_selection_per_group.md)

=== "Abstract"
    We propose a new variant of the Correlation-based Feature Selection (CFS) method for coping with longitudinal data - where variables are repeatedly measured across different time points. The proposed CFS variant is evaluated on ten datasets created using data from the English Longitudinal Study of Ageing (ELSA), with different age-related diseases used as the class variables to be predicted. The results show that, overall, the proposed CFS variant leads to better predictive performance than the standard CFS and the baseline approach of no feature selection, when using Naïve Bayes and J48 decision tree induction as classification algorithms (although the difference in performance is very small in the results for J4.8). We also report the most relevant features selected by J48 across the datasets.

=== "Credits / Authors"
    - Original authors: T. Pomsuwan & A.A. Freitas
    - Implementation: Sklong team

## 🚧 Ongoing Development Algorithms

### Time-penalised Trees (TpT) {.pub-heading}
=== "General"
    **Paper:** Time-penalised trees (TpT): introducing a new tree-based data mining algorithm for time-varying covariates. (Real-world application in life insurance; additional comparative studies forthcoming.) [Link (DOI)](https://doi.org/10.1007/s10472-024-09950-w)

    **API Reference:** Time-penalised Trees (TpT) — integration planned for a future release.

    **Discussion:** [Issue #65](https://github.com/simonprovost/scikit-longitudinal/issues/65)

=== "Abstract"
    This article introduces a new decision tree algorithm that accounts for time-varying covariates in the decision-making process. Traditional decision tree algorithms assume that the covariates are static and do not change over time, which can lead to inaccurate predictions in dynamic environments. Other existing methods suggest workaround solutions such as the pseudo-subject approach. The proposed algorithm utilises a different structure and a time-penalised splitting criterion that allows a recursive partitioning of both the covariates space and time. Relevant historical trends are then inherently involved in the construction of a tree, and are visible and interpretable once it is fit. This approach allows for innovative and highly interpretable analysis in settings where the covariates are subject to change over time. The effectiveness of the algorithm is demonstrated through a real-world data application in life insurance. The results presented in this article can be seen as an introduction or proof-of-concept of the time-penalised approach, and the algorithm’s theoretical properties and comparison against existing approaches on datasets from various fields will be explored in forthcoming work.

=== "Credits / Authors"
    - Original authors: Mathias Valla
    - Implementation: Under active development by [Mathias Valla](https://github.com/MathiasValla)

### Clustering-based KNN Regression for Longitudinal Data (CKNNRLD) {.pub-heading}
=== "General"
    **Paper:** Boosting K-nearest neighbor regression performance for longitudinal data through a novel learning approach. [Link (DOI)](https://doi.org/10.1186/s12859-025-06205-1)

    **Discussion:** [Issue #72](https://github.com/simonprovost/scikit-longitudinal/issues/72)

=== "Abstract"
    Longitudinal studies often require flexible methodologies for predicting response trajectories based on time-dependent and time-independent covariates. CKNNRLD first clusters data using the KML algorithm (K-means for longitudinal data), then searches for nearest neighbors within the relevant cluster rather than across the entire dataset. Simulation studies show improved prediction accuracy, shorter execution time, and reduced computational burden compared to standard KNN—e.g., execution time was roughly 3.7× faster for N=2000, T=5, D=2, C=4, E=1, R=1. This cluster-aware search scales better for larger longitudinal datasets where traditional KNN slows as sample counts grow.

=== "Credits / Authors"
    - Original authors: Mohammad Sadegh Loeloe, Seyyed Mohammad Tabatabaei, Reyhane Sefidkar, Amir Houshang Mehrparvar & Sara Jambarsang
    - Implementation: Collaborative prototype with the Sklong team
