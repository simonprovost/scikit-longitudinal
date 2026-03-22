<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
      <br>
      <a href="docs/assets/images/Sklong_Banner.avif">
         <img src="docs/assets/images/Sklong_Banner.avif" alt="Scikit-longitudinal banner">
      </a>
   </p>
   <h4 align="center">
      A Scikit-Learn-like Python library for Longitudinal Machine Learning —
      <a href="https://doi.org/10.21105/joss.08481">Paper</a> ·
      <a href="https://scikit-longitudinal.readthedocs.io/">Documentation</a> ·
      <a href="https://pypi.org/project/Scikit-longitudinal/">PyPi Index</a>
   </h4>
</div>

<div align="center">

<a href="https://pytest.org/">
   <img alt="pytest" src="https://img.shields.io/static/v1?label=Pytest&message=passing&color=472727&labelColor=FD706B&style=for-the-badge&logo=pytest&logoColor=white">
</a>

<a href="https://pre-commit.com/">
   <img alt="pre-commit" src="https://img.shields.io/static/v1?label=Pre-commit&message=checked&color=472727&labelColor=FD706B&style=for-the-badge&logo=pre-commit&logoColor=white">
</a>

<img src="https://img.shields.io/static/v1?label=Ruff&message=compliant&color=472727&labelColor=FD706B&style=for-the-badge&logo=ruff&logoColor=white" alt="Ruff compliant">

<img src="https://img.shields.io/static/v1?label=UV&message=managed&color=472727&labelColor=FD706B&style=for-the-badge&logo=uv&logoColor=white" alt="UV Managed">

<a href="https://codecov.io/gh/Scikit-Longitudinal/Scikit-Longitudinal">
   <img alt="Coverage" src="https://img.shields.io/static/v1?label=Coverage&message=88%25&color=472727&labelColor=FD706B&style=for-the-badge&logo=appveyor&logoColor=white">
</a>

<img src="https://img.shields.io/static/v1?label=Python&message=3.10%E2%80%933.13&color=472727&labelColor=FD706B&style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10–3.13">

</div>

## <a id="about-the-project"></a><img src="docs/assets/icons/lucide/github.svg" width="32" alt="" /> About The Project

`Bullshit` (Sklong) is a machine learning library tailored for Longitudinal machine (supervised) learning (Classification tasks focussed as of today). It offers tools and models for *processing, analysing, and predicting* longitudinal data, with a user-friendly interface that integrates with the `Scikit-learn` ecosystem.

**Wait, what is Longitudinal Data — In layman's terms?**

Longitudinal data is a "time-lapse" snapshot of the same subject, entity, or group tracked over time-periods,
similar to checking in on patients to see how they change. For instance, doctors may monitor a patient's blood pressure,
weight, and cholesterol every year for a decade to identify health trends or risk factors. This data is more useful for
predicting future results than a one-time (cross-sectional) survey because it captures evolution, patterns, and cause-effect throughout
time.

[See more in the documentation.](https://scikit-longitudinal.readthedocs.io/latest/)

## <a id="installation"></a><img src="docs/assets/icons/lucide/terminal.svg" width="32" alt="" /> Installation

To install Scikit-longitudinal:

```bash
pip install Scikit-longitudinal
```

To install a specific version:

```bash
pip install Scikit-longitudinal==0.1.0
```

> [!TIP]
> Want to use `Jupyter Notebook/Lab`, `Google Colab` or want to activate parallelism?
> Head to the [Getting Started](https://scikit-longitudinal.readthedocs.io/latest/getting-started/) section of the documentation, we explain it all! 🎉  

## <a id="getting-started"></a><img src="docs/assets/icons/lucide/square-code.svg" width="32" alt="" /> Getting Started

Let's run a simple Longitudinal machine learning classification task:

```py
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import LexicoGradientBoostingClassifier

dataset = LongitudinalDataset('./stroke.csv') # Note, this is a fictional dataset. Use yours!
dataset.load_data_target_train_test_split(
  target_column="class_stroke_wave_4",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="elsa")

model = LexicoGradientBoostingClassifier(
  features_group=dataset.feature_groups(),
  threshold_gain=0.00015 # Refer to the API for more hyper-parameters and their meaning
)

model.fit(dataset.X_train, dataset.y_train)
y_pred = model.predict(dataset.X_test)

# Classification report
print(classification_report(y_test, y_pred))
```

## <a id="citation"></a><img src="docs/assets/icons/lucide/graduation-cap.svg" width="32" alt="" /> How to Cite

If you use Sklong in your research, please cite our paper:

<a href="https://doi.org/10.21105/joss.08481">
   <img src="https://joss.theoj.org/papers/10.21105/joss.08481/status.svg" alt="JOSS DOI badge">
</a>

We would like to personally thank _Prof. Lengerich_ ([UW Madison](https://www.wisc.edu)—[@blengerich](https://github.com/blengerich) & [@AdaptInfer](https://github.com/AdaptInfer)), _&_ _Prof. Tahiri_ ([Université de Sherbrooke](https://www.usherbrooke.ca)—[@TahiriNadia](https://github.com/TahiriNadia) & [@tahiri-lab](https://github.com/tahiri-lab/)) for their amazing peer reviews!

```bibtex
@article{Provost2025,
    doi = {10.21105/joss.08481},
    url = {https://doi.org/10.21105/joss.08481},
    year = {2025},
    publisher = {The Open Journal},
    volume = {10},
    number = {112},
    pages = {8481},
    author = {Provost, Simon and Freitas, Alex A.},
    title = {Scikit-Longitudinal: A Machine Learning Library for Longitudinal Classification in Python},
    journal = {Journal of Open Source Software}
}
```

## <a id="license"></a><img src="docs/assets/icons/lucide/fingerprint-pattern.svg" width="32" alt="" /> License

Scikit-longitudinal is licensed under the [MIT License](./LICENSE).
