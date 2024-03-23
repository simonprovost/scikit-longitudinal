import os

import numpy as np

from scikit_longitudinal.data_preparation import LongitudinalDataset


def load_dataset(path: str, target_column: str) -> (np.ndarray, np.ndarray):
    """
    Loads a dataset from a given path and identifies the target column.

    Parameters:
        path (str): The file path to the dataset.
        target_column (str): The name of the target column in the dataset.

    Returns:
        DataFrame: The loaded dataset.
        Series: The target column.
    """
    longitudinal_dementia = LongitudinalDataset(file_path=path)
    longitudinal_dementia.load_data_target_train_test_split(
        target_column=target_column,
        remove_target_waves=True,
        random_state=42,
    )
    df = longitudinal_dementia.data
    target = longitudinal_dementia.target
    return df, target


def report_missing_values(df: np.ndarray) -> dict:
    """
    Checks for features with missing values in the dataset and reports the percentage of missing values for each feature.

    Parameters:
        df (DataFrame): The dataset to analyze.

    Returns:
        dict: A dictionary with feature names as keys and the percentage of missing values as values.
    """
    missing_percentage = df.isnull().mean() * 100
    return missing_percentage[missing_percentage > 0].to_dict()


def calculate_class_imbalance_ratio(target: np.ndarray) -> (dict, float):
    """
    Calculates and reports the class imbalance ratio alongside the minority/positive ratio
    for the given target class column, expressing the ratios as percentages and the imbalance ratio
    as the ratio of the most frequent class to the least frequent class.

    Parameters:
        target (Series): The target column containing class labels.

    Returns:
        dict: A dictionary with class labels as keys and their percentage of the total.
        float: The class imbalance ratio, defined as the ratio of the most frequent class to the least frequent class.
    """
    value_counts = target.value_counts()
    total = target.count()
    percentages = {label: (count / total) * 100 for label, count in value_counts.iteritems()}
    imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
    return percentages, imbalance_ratio


def check_normalization(df: np.ndarray) -> dict:
    """
    Checks if numerical features in the dataset are normalized to the range [0, 1].

    Parameters:
        df (DataFrame): The dataset containing numerical and categorical features.

    Returns:
        dict: A dictionary with feature names as keys and a boolean indicating whether it is normalized to [0, 1] as values.
    """
    features_status = {}
    numerical_features = df.select_dtypes(include=[np.number])
    for column in numerical_features.columns:
        min_val = numerical_features[column].min()
        max_val = numerical_features[column].max()
        is_normalized = np.isclose(min_val, 0, atol=0.01) and np.isclose(max_val, 1, atol=0.01)
        features_status[column] = is_normalized
    return features_status


def categorical_feature_analysis(df: np.ndarray) -> list:
    """
    Checks for categorical features and determines if they need one-hot encoding.

    Parameters:
        df (DataFrame): The dataset to analyze.

    Returns:
        list: A list of categorical features that need one-hot encoding or an indication that none are found.
    """
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return categorical_features if categorical_features else ["No categorical features found"]


def binary_feature_analysis(df: np.ndarray) -> dict:
    """
    Identifies binary features with less than 10 instances of one of the binary values.

    Parameters:
        df (DataFrame): The dataset to analyze.

    Returns:
        dict: A dictionary with binary features as keys and the count of the least frequent value as values.
    """
    binary_features = {}
    for column in df:
        if df[column].nunique() == 2:
            value_counts = df[column].value_counts()
            if value_counts.min() < 10:
                binary_features[column] = value_counts.to_dict()
    return binary_features


def report_analysis(
        dataset_scope: str,
        dataset_name: str,
        class_distributions: dict,
        class_imbalance_ratio: float,
        normalization_check: dict,
        categorical_features: list,
        binary_features_analysis: dict,
        missing_values_analysis: dict,
):
    """
    Generates a markdown report for the analysis of a dataset, including class imbalance ratio and missing values analysis.

    Parameters:
        dataset_scope (str):
            The scope of the dataset, either "core" or "nurse".
        dataset_name (str):
            The name of the dataset being analyzed.
        class_distributions (dict):
            The percentage of each class in the target column.
        class_imbalance_ratio (float):
            The ratio of the most frequent class to the least frequent class.
        normalization_check (dict):
            The normalization status of numerical features.
        categorical_features (list):
            List of categorical features that need encoding or indication of none.
        binary_features_analysis (dict):
            Binary features with less than 10 instances of one of the binary values.
        missing_values_analysis (dict):
            The percentage of missing values for each feature.

    Returns:
        str: A markdown formatted string representing the analysis report.
    """
    report = f"# Analysis Report for {dataset_name} – Scope: {dataset_scope}\n\n"
    report += "## Class Distribution and Imbalance\n"
    for label, percentage in class_distributions.items():
        report += f"- **Label {label}:** {percentage:.2f}%\n"
    report += f"\n**Class Imbalance Ratio:** {class_imbalance_ratio:.2f} (Most frequent class to least frequent class)\n"

    report += "\n## Normalization Check for Numerical Features\n"
    if all(is_normalized for is_normalized in normalization_check.values()):
        report += "- All numerical features are normalized to [0, 1].\n"
    else:
        for feature, is_normalized in normalization_check.items():
            if not is_normalized:
                report += f"- **{feature}:** Not Normalized\n"

    report += "\n## Categorical Features Analysis\n"
    if categorical_features == ["No categorical features found"]:
        report += "- No categorical features found that require encoding.\n"
    else:
        for feature in categorical_features:
            report += f"- **{feature}** needs one-hot encoding.\n"

    report += "\n## Binary Feature Analysis\n"
    if not binary_features_analysis:
        report += "- No binary features with bias concerns.\n"
    else:
        for feature, counts in binary_features_analysis.items():
            report += f"- **{feature}:** {counts}\n"

    report += "\n## Missing Values Analysis\n"
    if not missing_values_analysis:
        report += "- No features with missing values.\n"
    else:
        for feature, percentage in missing_values_analysis.items():
            report += f"- **{feature}:** {percentage:.2f}% missing values.\n"

    report += "\n---\n\n"
    return report


def main(dataset_info_list: list):
    """
    Main function to loop over datasets, perform analyses, and generate a markdown report.

    Parameters:
        dataset_info_list (list): A list of tuples with dataset path and target class column name.

    Returns:
        None: Generates and saves a markdown report.
    """
    full_report = ""
    for path, target_column in dataset_info_list:
        df, target = load_dataset(path, target_column)
        if "core" in path:
            dataset_scope = "core"
        else:
            dataset_scope = "nurse"
        dataset_name = path.split("/")[-1]

        normalization_check = check_normalization(df)
        categorical_features = categorical_feature_analysis(df)
        binary_features_analysis = binary_feature_analysis(df)
        missing_values = report_missing_values(df)
        class_imbalance, imbalance_ratio = calculate_class_imbalance_ratio(target)

        report = report_analysis(
            dataset_scope,
            dataset_name,
            class_imbalance,
            imbalance_ratio,
            normalization_check,
            categorical_features,
            binary_features_analysis,
            missing_values,
        )
        full_report += report

    current_directory = os.path.dirname(__file__)
    with open(f"{current_directory}/analysis_report.md", "w") as report_file:
        report_file.write(full_report)


if __name__ == "__main__":
    """ Run the main function with the ELSA datasets.
    
    The main function will loop over the datasets, perform analyses, and generate a markdown report. However,
    the data used in this script is not included in the repository. To run this script, you need to have access to the
    ELSA datasets and update the directories_to_check list with the correct paths to the datasets. Therefore,
    you can query the datasets directly to the author of this repository via opening a Github issue.
    
    """
    datasets = []
    current_directory = os.path.dirname(__file__)
    directories_to_check = [
        current_directory + "/../../../data/elsa/core/csv/",
        current_directory + "/../../../data/elsa/nurse/csv/",
    ]

    for directory in directories_to_check:
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                dataset_name = file.split("/")[-1]
                target_column = file.split("/")[-1].split("_")[0]
                datasets.append(
                    (
                        f"{directory}{file}",
                        f"class_{target_column}_w8",
                    )
                )
    main(datasets)
