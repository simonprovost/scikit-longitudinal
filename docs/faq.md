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
    If you're looking for more information on how to use Scikit-Longitudinal, check out our [API](API/index.md).

!!! question "What if I have a question that isn't answered here?"
    If you have a question that isn't answered here, feel free to reach out to us on
    [GitHub Issues](https://github.com/simonprovost/scikit-longitudinal/issues).

!!! quote "More projects to explore"
    Looking for complementary tooling? Visit our [Related Projects](related-projects.md) page for a curated list.