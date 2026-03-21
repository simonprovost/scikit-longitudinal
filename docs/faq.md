---
icon: lucide/circle-help
---

# Frequently Asked Questions

??? example "What is longitudinal data?"
    Longitudinal data records the same subjects across multiple time points, usually called `waves`, whilst tracking multiple variables for each subject. In biomedical research, this could mean measuring blood pressure, cholesterol, and smoking status for the same patient across repeated visits. Its main value is that it captures change over time rather than a single snapshot.

??? example "How is longitudinal data different from time-series data?"
    Time-series data and longitudinal data both involve observations made over time, but they differ in several important ways:

    - **Focus**: Time-series data usually follows one variable, or a small set of variables, across many time points. Longitudinal data usually follows multiple variables for the same subjects across repeated waves or visits.
    - **Nature**: Time-series workflows often focus on ordered temporal sequences, frequently numerical, whereas longitudinal data commonly includes both numerical and categorical variables collected repeatedly for each subject.
    - **Time structure**: Time-series data often involves many closely spaced observations, whilst longitudinal data usually involves fewer observations spread across longer periods such as months or years.
    - **Irregularity**: Time-series data is often regularly spaced, whereas longitudinal data more commonly includes irregular gaps between observations or missing visits.
    - **Machine-learning goal**: Time-series methods are often used to predict future values or detect temporal patterns in sequences. Longitudinal classification usually aims to predict a class observed at a single reference wave from features measured across several waves.

    In summary, the main differences lie in how time is represented, how subjects and variables are organised, and which prediction task is being addressed. This is why methods designed for long numerical sequences do not transfer directly to the repeated-measures classification setting targeted by `Sklong`.

??? example "What is longitudinal classification in `Scikit-Longitudinal`?"
    In `Sklong`, longitudinal classification means supervised learning where features are observed across several waves and the goal is to predict a categorical outcome. More precisely, the main setting is one where the features are observed across multiple time points, whilst the class variable is observed at a single reference wave.

    The library is therefore designed around longitudinal classification workflows rather than generic forecasting or regression workflows, keeping the tooling focused on repeated-measures tabular data.

??? example "What is `Auto-Sklong`?"
    `Auto-Sklong` is the companion AutoML project for longitudinal classification. `AutoML` means *Automated Machine Learning*: instead of choosing a pipeline and tuning its hyperparameters by hand, the system searches automatically across candidate longitudinal pipelines and their hyperparameters.

    In practice, this is a combinatorial-optimisation problem, because the system must explore many possible combinations of data-preparation methods, feature-selection options, class-balancing choices, classifiers, and hyperparameter settings.

    In other words, `Sklong` is the library for building and running longitudinal workflows, whilst `Auto-Sklong` is the separate project for automating model and pipeline selection on top of that ecosystem.

    Since this FAQ belongs to the `Sklong` repository, the full `Auto-Sklong` documentation and source code live separately:

    - [Auto-Sklong documentation](https://auto-sklong.readthedocs.io/en/latest/)
    - [Auto-Sklong on GitHub](https://github.com/simonprovost/auto-sklong)

??? example "Why not just flatten the dataset and use ordinary `scikit-learn`?"
    You can, and `MerWav-Time(-)` is exactly that baseline. The issue is that a standard learner will then treat repeated measurements as ordinary independent columns, without understanding that `cholesterol_w1` and `cholesterol_w4` are the same underlying variable observed at different time points.

    This matters because no single method dominates across all longitudinal datasets, and methods that preserve or explicitly exploit temporal structure often outperform plain flattening. `Sklong` exists so you can compare that baseline against more longitudinal-data-aware alternatives instead of assuming flattening is always enough.

??? example "What prediction setup does `Sklong` assume?"
    The main setup assumed in `Sklong` is this one: features are observed across multiple waves, whilst the class is observed at a single reference wave, often the most recent one. In other words, the library is primarily designed for predicting one target from repeated measurements collected earlier or alongside that final reference point.

    If your task has labels at every wave, is mainly sequence forecasting, or is centred on very long numerical series, that is a different problem family from the one `Sklong` currently targets.

??? example "What input format does `Sklong` expect: wide or long?"
    `Sklong` prefers the `wide` format: one row per subject, with wave-specific measurements represented as columns such as `smoke_w1`, `smoke_w2`, and `smoke_w3`. This makes train/test splitting safer, reduces leakage risk, and works naturally with the metadata used throughout the library.

    This also reflects the design of `Sklong`: rather than hard-coding dataset-specific column names, the library uses a wide-table representation together with lightweight metadata describing which columns form longitudinal groups and which columns are static.

    If your data is currently in `long` format, it is best to pivot it first. The [longitudinal data format tutorial](tutorials/sklong_longitudinal_data_format.md) and the [advanced uneven-wave setup tutorial](tutorials/advanced_temporal_setup.md) walk through that process.

??? example "What are waves, feature groups, and non-longitudinal features?"
    A `wave` is an ordered visit or time point in your study. A `features_group` is the ordered list of columns belonging to the same longitudinal variable across waves. `non_longitudinal_features` are the columns that do not follow that temporal pattern, such as static demographics.

    These metadata are central to `Sklong`. They let estimators and preprocessors reason about temporal structure and recency without hard-coding dataset-specific column names. This is consistent with the design choice in `Sklong`, where temporal grouping is represented explicitly through metadata rather than through bespoke code for each dataset. The [temporal dependency tutorial](tutorials/temporal_dependency.md) explains how to define them in practice.

??? example "How do I handle missing or uneven waves?"
    Keep the dataset in wide format, include the maximum number of waves you want to model, and leave missing visits as `NaN` when a subject simply does not have that observation. If an entire wave column does not exist for a feature, pad that position with `-1` inside `features_group` so the temporal ordering still remains explicit.

    In short:

    - Use `NaN` for missing values within an existing wave column.
    - Use `-1` only when a whole wave column is absent.
    - Keep one row per subject throughout.

    The [advanced temporal setup tutorial](tutorials/advanced_temporal_setup.md) covers this pattern in detail.

??? example "Should I use data transformation or algorithm adaptation?"
    Use the `data-transformation` pathway when you want to convert longitudinal data into a form that standard tabular learners can consume. This is usually the easiest path for baselines, quick experiments, or when you want to stay close to familiar `scikit-learn` workflows.

    Use the `algorithm-adaptation` pathway when you want the downstream model to use temporal structure directly, especially with longitudinal-data-aware feature selection and estimators. The comparison between both pathways is strongly motivated because neither one dominates across all datasets. If you are unsure, compare both manually or let [Auto-Sklong](https://github.com/simonprovost/Auto-Sklong) search across them for you.

??? example "What is the difference between `MerWav-Time(-)`, `AggrFunc`, `SepWav`, and `MerWav-Time(+)`?"
    These methods correspond to the main longitudinal pathways used in `Sklong` and `Auto-Sklong`.

    - `MerWav-Time(-)` flattens all wave-specific measurements into a standard tabular dataset and treats the multiple values of an original feature at different time points as distinct features.
    - `AggrFunc` aggregates each longitudinal feature across waves using a statistic such as the mean or median, yielding a single-wave representation.
    - `SepWav` trains one classifier per wave and combines those predictions afterwards.
    - `MerWav-Time(+)` keeps time indices explicit so that longitudinal-data-aware feature selection and estimators can operate on the temporal structure directly.

    The first three belong to the data-transformation family. `MerWav-Time(+)` belongs to the algorithm-adaptation family. See the API pages for [Merge Waves: Drop Indices](API/data_preparation/merwav_time_minus.md), [Aggregation Function](API/data_preparation/aggregation_function.md), [Separate Waves](API/data_preparation/sepwav.md), and [Merge Waves: Keep Indices](API/data_preparation/merwav_time_plus.md).

??? example "Does `Sklong` support binary and multiclass classification?"
    Yes. `Sklong` supports both binary and multiclass classification for its main classification workflows. That includes the lexicographic estimators, `NestedTrees`, and `SepWav` variants.

    The main differences are in how you interpret `predict_proba`, `classes_`, and evaluation metrics such as AUPRC. The [binary vs. multiclass tutorial](tutorials/binary_vs_multiclass.md) shows the same workflow in both settings.

??? example "When should I use class weighting?"
    Use class weighting when your target classes are imbalanced and missing the minority class would be costly. Class weighting is a simple way to rebalance learning pressure without changing the dataset structure itself.

??? example "What are the current limitations of `Scikit-Longitudinal`?"
    `Sklong` is currently focused on supervised longitudinal classification. It is not a general-purpose framework for every temporal machine-learning problem.

    The main current boundaries are:

    - no regression support yet, although it is wanted,
    - no longitudinal neural-network family integrated into the core workflow yet, although it is wanted,
    - no attempt to cover full time-series forecasting problems, since `aeon` and similar toolkits are already well shaped for that space,
    - an expectation that temporal structure is provided explicitly through wide-format data and feature-group metadata.

???+ bug "What if I have a question that isn't answered here?"
    If your question is not covered in this FAQ, please open an issue on
    [GitHub Issues](https://github.com/simonprovost/scikit-longitudinal/issues).
