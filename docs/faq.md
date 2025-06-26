---
hide:
  - navigation
---

# 👀 Frequently Asked Questions
# 👀 Frequently Asked Questions

!!! info "What is a Longitudinal Data?"
    Longitudinal Data refers to observations made on multiple variables of interest for the same subject 
    over an extended period of time. This type of data is particularly valuable for studying 
    changes and trends, as well as making predictions about future outcomes. For example, in 
    a medical study, a patient's health measurements such as blood pressure, heart rate, 
    and weight might be recorded over several years to analyze the effectiveness of a treatment.

!!! question "What are the differences between Time-Series Data and Longitudinal Data?"
    Time-Series Data and Longitudinal Data both involve observations made over time, but they differ in several aspects:

    - **Focus**: _Originally_ Time-Series Data focuses on a single variable measured at regular intervals, while Longitudinal Data involves multiple variables observed over time for each subject.
    - **Nature**: Time-Series Data is usually employed for continuous data, whereas Longitudinal Data can handle both continuous and categorical data.
    - **Time gap**: Time-Series Data typically deals with shorter time periods (e.g., seconds, minutes, or hours), while Longitudinal Data often spans longer durations (e.g., months or years).
    - **Irregularity**: Time-Series Data is often regularly spaced, while Longitudinal Data can have irregular time intervals between observations.
    - **Machine Learning**: Time-Series Data are frequently used to predict future values, whereas Longitudinal Data are more frequently used to predict future outcomes. In addition, 
    the ML algorithms used for time-series are frequently distinct from those used for longitudinal data. For instance, Time-Series based techniques are based on time-windowing techniques, whereas Longitudinal based techniques frequently 
    use the current standard for machine learning classification for the prediction task. 
    Nevertheless, they will adapt (create variant of these standard classification based machine learning algorithm) 
    to comprehend the temporal nature of the data.

    In summary, the main differences between Time-Series and Longitudinal Data lie in the focus, nature, and the length of the time intervals considered.

!!! warning "Are there any limitations I should be aware of?"
    While Scikit-Longitudinal is a powerful tool for handling longitudinal data, it's important to be aware of its limitations. For example, as of today's date, `Scikit_Longitudinal` does not support regression tasks nor Neural-Networks adapted for Longitudinal Data. However, we are constantly working to improve the library and add new features, so be sure to check for updates regularly or contribute to the project on
    [GitHub](https://github.com/simonprovost/scikit-longitudinal).

!!! note "Where can I find more resources?"
    If you're looking for more information on how to use Scikit-Longitudinal, check out our [API Reference](API/index.md).

!!! question "What if I have a question that isn't answered here?"
    If you have a question that isn't answered here, feel free to reach out to us on 
    [GitHub Issues](https://github.com/simonprovost/scikit-longitudinal/issues).

# Related Projects

!!! quote "Auto-Sklong"
    -    _Open-source:_ ✅
    -    _Authors:_ [Simon Provost](https://github.com/simonprovost) & [Alex Freitas](https://www.kent.ac.uk/computing/people/3057/freitas-alex)
    -    _Github Link:_ [Auto-Sklong](https://github.com/simonprovost/Auto-Sklong)
    -   _Description:_ Auto-Sklong - An automated machine learning pipeline for longitudinal data. Leveraging Scikit-Longitudinal and Auto-Sklearn with a novel Search Space tailored to the Longitudinal ML classification task.
    -   _Note_: Sklong authors are Auto-Sklong authors.
!!! quote "Auto-prognosis"
    -    _Open-source:_ ✅
    -    _Authors:_ [VanderSchaar Lab](https://www.vanderschaar-lab.com/)
    -    _Github Link:_ [Auto-prognosis](https://github.com/vanderschaarlab/autoprognosis)
    -   _Description:_ AutoPrognosis - A system for automating the design of predictive modeling pipelines tailored for clinical prognosis.
    - _Note:_ Auto-prognosis is highly correlated with the [StepWise Model Selection Via Deep Kernel Learning (SMS-DKL)](https://proceedings.mlr.press/v108/zhang20f.html) and the Clairvoyance projects. Worth reading the paper, [yet the project despite being open source is very limited.](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/smsdkl)
!!! quote "Clairvoyance"
    -    _Open-source:_ ✅
    -    _Authors:_ [VanderSchaar Lab](https://www.vanderschaar-lab.com/)
    -    _Github Link:_ [Clairvoyance](https://github.com/vanderschaarlab/clairvoyance)
    -   _Description:_ Clairvoyance - A Pipeline Toolkit for Medical Time Series

!!! quote "LongiTools"
    -    _Open-source:_ ⏳
    -    _Authors:_ [LongiTools](https://longitools.org/)
    -    _Official Website:_ [LongiTools](https://longitools.org/)
    -   _Description:_ A European research project studying the interactions between environmental, lifestyle and biological factors to determine the risks of chronic cardiovascular and metabolic diseases.

!!! quote "Want your project to be featured here?"
    -    _Open-source:_ ✅ or ⏳ or ❌
    -    _Authors:_ ``Your Name(s)``
    -    _Github Link:_: ``Your Project Link``
    -   _Description:_ ``A brief description of your project``
    -   _How To?_: Open An Issue and We'll Add It Here!