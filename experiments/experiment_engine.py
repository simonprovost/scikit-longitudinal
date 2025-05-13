import csv
import os
import time
from typing import List, Any, Tuple, Callable

import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scikit_longitudinal.data_preparation import LongitudinalDataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold


def _load_dataset(
    dataset_file_path: str,
    target_column: str,
    random_state: int,
    remove_target_waves: bool = True,
    setup_features_group="elsa",
) -> (pd.DataFrame, pd.Series, List[List[int]], List[int], List[str]):
    """
    Loads the dataset from the specified file path, processes it, and prepares it for use in the experiment.

    Args:
        dataset_file_path (str): Path to the dataset file.
        target_column (str): The target column in the dataset.
        random_state (int): Seed for reproducibility.
        remove_target_waves (bool, optional): Whether to remove target waves from the dataset. Defaults to True.
        > To update the setup_data_parameters, to use one different from True.
        setup_features_group (str, optional): The method to set up the features group. Defaults to "elsa".
        > To update the setup_data_parameters, to use one different from "elsa".

    Returns:
        pd.DataFrame: The loaded dataset.
        pd.Series: The target column from the dataset.
        List[List[int]]: Grouped feature indices.
        List[int]: Non-longitudinal feature indices.
        List[str]: List of column names in the dataset.

    Raises:
        ValueError: If the dataset path does not exist.
    """
    if not os.path.exists(dataset_file_path):
        raise ValueError(f"The dataset path {dataset_file_path} does not exist.")

    longitudinal_dataset = LongitudinalDataset(file_path=dataset_file_path)
    longitudinal_dataset.load_data()
    # check if data has missing values or "?", if so, fill them with the mean
    if longitudinal_dataset.data.isnull().values.any() or any(
        [
            any([True for value in longitudinal_dataset.data[column] if value == "?"])
            for column in longitudinal_dataset.data.columns
        ]
    ):
        if longitudinal_dataset.data[target_column].isnull().values.any() or any(
            [True for value in longitudinal_dataset.data[target_column] if value == "?"]
        ):
            raise ValueError("Data has missing values in the target column.")
        print("Data has missing values. Filling them with the mean.")
        longitudinal_dataset.set_data(
            longitudinal_dataset.data.replace("?", pd.NA).apply(
                pd.to_numeric, errors="ignore"
            )
        )
        longitudinal_dataset.set_data(
            longitudinal_dataset.data.fillna(longitudinal_dataset.data.mean())
        )
        longitudinal_dataset.load_data()

    longitudinal_dataset.load_data_target_train_test_split(
        target_column=target_column,
        remove_target_waves=remove_target_waves,
        random_state=random_state,
    )
    longitudinal_dataset.setup_features_group(setup_features_group)

    return (
        longitudinal_dataset.data,
        longitudinal_dataset.target,
        longitudinal_dataset.feature_groups(),
        longitudinal_dataset.non_longitudinal_features(),
        longitudinal_dataset.data.columns.tolist(),
    )


def _split_dataset_by_kfold(
    dataset_file_path: str,
    n_outer_splits: int,
    shuffling: bool,
    fold_number: int,
    random_state: int,
    target_column: str,
) -> Tuple[str, str]:
    """
    Splits the dataset into training and testing sets based on K-Fold cross-validation.

    > To add one more parameters if different splitting strategy is neeed. E.g., StratifiedKFold.

    Args:
        dataset_file_path (str): Path to the dataset file.
        n_outer_splits (int): Number of outer splits for cross-validation.
        shuffling (bool): Whether to shuffle the data before splitting.
        fold_number (int): The current fold number.
        random_state (int): Seed for reproducibility.
        target_column (str): The target column in the dataset.

    Returns:
        Tuple[str, str]: Paths to the training and testing data files for the current fold.

    Raises:
        ValueError: If the dataset path does not exist.
    """
    if os.path.exists(dataset_file_path) and all(
        [
            os.path.exists(
                f"{os.path.splitext(dataset_file_path)[0]}_split_{i}_train.csv"
            )
            for i in range(1, n_outer_splits + 1)
        ]
    ):
        print(
            f"Split training and testing datasets already exist for the dataset: "
            f"{dataset_file_path} – Skipping the split."
        )
        return (
            f"{os.path.splitext(dataset_file_path)[0]}_split_{fold_number}_train.csv",
            f"{os.path.splitext(dataset_file_path)[0]}_split_{fold_number}_test.csv",
        )

    data = pd.read_csv(dataset_file_path)
    # folds = KFold(n_splits=n_outer_splits, shuffle=shuffling, random_state=random_state if shuffling else None)
    # Separate features and target column
    X = data.drop(columns=[target_column])
    y = data[target_column]

    folds = StratifiedKFold(
        n_splits=n_outer_splits,
        shuffle=shuffling,
        random_state=random_state if shuffling else None,
    )

    for i, (train_index, test_index) in enumerate(folds.split(X, y), 1):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        train_path = f"{os.path.splitext(dataset_file_path)[0]}_split_{i}_train.csv"
        test_path = f"{os.path.splitext(dataset_file_path)[0]}_split_{i}_test.csv"

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        print(
            f"Split {i} created - Train data saved to: {train_path}, Test data saved to: {test_path}"
        )

    return (
        f"{os.path.splitext(dataset_file_path)[0]}_split_{fold_number}_train.csv",
        f"{os.path.splitext(dataset_file_path)[0]}_split_{fold_number}_test.csv",
    )


class ExperimentEngine:
    """
    The ExperimentEngine class is responsible for running nested cross-validation experiments on a given dataset using
    an AutoML system or Scikit Longitudinal primitives.

    This class is designed to be flexible, allowing for the use of various AutoML systems or custom Sklong algorithms that
    adhere to the required interface.
    The results of each outer fold of the nested cross-validation are saved separately and can be manually merged
    afterward.

    Attributes:
        output_path (str): The directory where the output of each fold will be saved.
        fold_number (int): The current fold number in the nested cross-validation.
        setup_data_parameters (dict[str, Any]): Parameters required for setting up the dataset.
        system_reporter (Callable): Function or callable object used to report the results of the experiment.
        system_hyperparameters (dict[str, Any], Optional): Hyperparameters for configuring the AutoML system or Scikit Longitudinal primitives
        By Default, will be used AutoSklong with the provided hyperparameters.

    Methods:
        __init__(output_path, fold_number, setup_data_parameters, system_hyperparameters, system_reporter):
            Initializes the ExperimentEngine with the given parameters.
        _setup_output_directory():
            Sets up the directory structure for saving the output of the current fold.
        _setup_data():
            Prepares the training and testing datasets for the current fold.
        _setup_system():
            Configures the AutoML system or custom Sklong algorithm to be used in the experiment.
        run_experiment():
            Executes the training and evaluation of the configured system.
        report_experiment():
            Reports and saves the results of the experiment for the current fold.
    """

    def __init__(
        self,
        output_path: str,
        fold_number: int,
        setup_data_parameters: dict[str, Any],
        system_reporter: Callable,
        system_hyperparameters: dict[str, Any] = None,
        imbalanced_scenario: str = "vanilla",
    ):
        """
        Initialises the ExperimentEngine with the provided configuration.

        Args:
            output_path (str): The directory path where output files for each fold will be stored.
            fold_number (int): The fold number for the current run of nested cross-validation.
            setup_data_parameters (dict[str, Any]): A dictionary containing necessary parameters for data setup.
            Must include:
                - "dataset_file_path": Path to the dataset file.
                - "n_outer_splits": Number of outer splits for cross-validation.
                - "random_state": Random seed for reproducibility.
                - "target_column": The target column in the dataset.
                - "shuffling": Whether to shuffle the data before splitting.
            system_reporter (Callable,): A callable for reporting the results of the system.
                - The reporter function takes the system as input, as well as the test set, and return a dictionary of
                experiment results, where the object returned must contain:
                    - "predictions": The predicted labels for the test set.
                    - "probability_predictions": The predicted probabilities for each class.
                    - "best_pipeline": A dictionary containing the best pipeline configuration found by the system. The
                    object must look as follows:
                        - "data_preparation": The name of the data preparation technique used
                        - "preprocessor": The name of the preprocessor used
                        - "classifier": The name of the classifier used
                    - "metric_optimised": The name of the metric optimised by the system.
            system_hyperparameters (dict[str, Any]): A dictionary of hyperparameters for the AutoML system or or Scikit Longitudinal primitives. See further
            in `_setup_system`.

        Raises:
            ValueError: If any required setup_data_parameters are missing or invalid. As well as,
            if any required system_hyperparameters are missing or invalid.
        """
        required_data_parameters = [
            "dataset_file_path",
            "n_outer_splits",
            "random_state",
            "target_column",
            "shuffling",
        ]
        for param in required_data_parameters:
            if param not in setup_data_parameters:
                raise ValueError(
                    f"The {param} is required in setup_data_parameters. Missing: `{param}`"
                )

        print(
            f"system_hyperparameters.get(custom_system), {system_hyperparameters.get('custom_system', False)}"
        )
        print(f"Cond: {not system_hyperparameters.get('custom_system', False)}")
        if not system_hyperparameters.get("custom_system", False):
            if not system_hyperparameters:
                raise ValueError("System hyperparameters are required.")
            required_system_parameters = [
                "max_total_time",
                "max_eval_time",
                "search",
                "scoring",
                "store",
                "verbosity",
                "n_inner_jobs",
                "max_memory_mb",
            ]
            for param in required_system_parameters:
                if param not in system_hyperparameters:
                    raise ValueError(
                        f"The {param} is required in system_hyperparameters. Missing: `{param}`"
                    )

        required_imbalanced_scenario = [
            "vanilla",
            "random_under_sampling",
            "under_bagging",
            "easy_ensemble",
            "smote",
        ]

        if imbalanced_scenario not in required_imbalanced_scenario:
            raise ValueError(
                f"The imbalanced_scenario must be one of the following: {required_imbalanced_scenario}. "
                f"Current: {imbalanced_scenario}"
            )

        self.output_path = output_path
        self.fold_number = fold_number
        self.setup_data_parameters = setup_data_parameters
        self.system_hyperparameters = system_hyperparameters
        self._reporter = system_reporter
        self.imbalanced_scenario = imbalanced_scenario

        self._experiment = {}
        self._system = None
        self._train_data_path = ""
        self._test_data_path = ""
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._features_group = []
        self._non_longitudinal_features = []
        self._feature_list_names = []

        self._setup_output_directory()
        self._setup_data()
        self._setup_system()

    def _setup_output_directory(self) -> None:
        """
        Sets up the output directory for the current fold.

        Creates the main output directory if it doesn't exist and ensures that the directory for the current fold is
        created.

        Raises:
            ValueError: If the experiment has already been conducted (experiment_results.csv exists and is not empty).
        """
        current_directory = os.getcwd()
        directory_name = self.output_path
        directory_split_name = f"{directory_name}/fold_{self.fold_number}"
        fold_directory = os.path.join(current_directory, directory_split_name)

        # Ensure main output directory exists
        main_directory = os.path.join(current_directory, directory_name)
        if os.path.exists(main_directory):
            print(
                f"Output directory {main_directory} already exists. "
                f"Will only try to create the fold directory."
            )
        else:
            os.makedirs(main_directory, exist_ok=True)
            print(f"Output directory {main_directory} created.")

        # Now handle the fold directory
        if os.path.exists(fold_directory):
            # Check if experiment_results.csv exists
            results_file = os.path.join(fold_directory, "experiment_results.csv")
            if os.path.exists(results_file):
                # Check if experiment_results.csv is not empty
                if os.stat(results_file).st_size > 0:
                    raise ValueError(
                        f"Experiment for fold {self.fold_number} has already been conducted. "
                        f"Results file {results_file} is not empty."
                    )
                else:
                    print(
                        f"Experiment results file {results_file} is empty. Proceeding with the experiment, "
                        f"but removing the file."
                    )
                    os.remove(results_file)
            else:
                print(
                    f"Experiment results file {results_file} does not exist. Proceeding with the experiment."
                )
            # Directory exists, proceed
        else:
            # Create the fold directory
            os.makedirs(fold_directory, exist_ok=True)
            print(f"Output directory {fold_directory} created.")

        # Update self.output_path
        self.output_path = fold_directory

    def _setup_data(self) -> None:
        """
        Prepares the training and testing datasets for the current fold.

        Splits the dataset according to the specified parameters and loads the data into
        the appropriate attributes.

        Raises:
            ValueError: If required data parameters are missing or if data loading fails.
        """
        self._train_data_path, self._test_data_path = _split_dataset_by_kfold(
            dataset_file_path=self.setup_data_parameters["dataset_file_path"],
            n_outer_splits=self.setup_data_parameters["n_outer_splits"],
            shuffling=self.setup_data_parameters["shuffling"],
            fold_number=self.fold_number,
            random_state=self.setup_data_parameters["random_state"],
            target_column=self.setup_data_parameters["target_column"],
        )

        (
            self._X_train,
            self._y_train,
            self._features_group,
            self._non_longitudinal_features,
            self._feature_list_names,
        ) = _load_dataset(
            dataset_file_path=self._train_data_path,
            target_column=self.setup_data_parameters["target_column"],
            random_state=self.setup_data_parameters["random_state"],
        )
        self._X_test, self._y_test, _, _, _ = _load_dataset(
            dataset_file_path=self._test_data_path,
            target_column=self.setup_data_parameters["target_column"],
            random_state=self.setup_data_parameters["random_state"],
        )

        if any(
            [
                self._X_train.empty,
                self._X_test.empty,
                not self._features_group,
                not self._non_longitudinal_features,
                not self._feature_list_names,
            ]
        ):
            raise ValueError("Data not loaded correctly.")

    def _setup_system(self):
        """
        Configures the AutoML system or custom algorithm.

        If a `custom_system` is provided in the system_hyperparameters, it is used directly. Otherwise,
        the Auto-Sklong is configured with the provided hyperparameters.

        Raises:
            ValueError: If the system setup fails or if required methods are not implemented by the system.
        """
        if self.system_hyperparameters.get("custom_system", False):
            hyperparameters = self.system_hyperparameters.copy()
            hyperparameters.pop("custom_system")

            if hyperparameters.get("output_folder", False):
                hyperparameters[
                    hyperparameters.get("output_folder")
                ] = f"{self.output_path}/fold_{self.fold_number}_" + time.strftime(
                    "%Y%m%d-%H%M%S"
                )
                hyperparameters.pop("output_folder")
            elif hasattr(self.system_hyperparameters["custom_system"], "output_folder"):
                hyperparameters[
                    "output_folder"
                ] = f"{self.output_path}/fold_{self.fold_number}_" + time.strftime(
                    "%Y%m%d-%H%M%S"
                )
            else:
                print(
                    "No output folder provided for the custom system. If needed, please provide it through"
                    "the system_hyperparameters 'output_folder' key. E.g output_folder='<the name of the output "
                    "folder in your system>'"
                )

            if (
                "estimator" in hyperparameters
                and isinstance(hyperparameters["estimator"], Pipeline)
                and hasattr(hyperparameters["estimator"], "steps")
            ):
                for idx, step in enumerate(hyperparameters["estimator"].steps):
                    if step[0] == "classifier":
                        if hasattr(
                            hyperparameters["estimator"].steps[idx][1], "features_group"
                        ):
                            hyperparameters["estimator"].steps[idx][
                                1
                            ].features_group = self._features_group
                        if hasattr(
                            hyperparameters["estimator"].steps[idx][1],
                            "non_longitudinal_features",
                        ):
                            hyperparameters["estimator"].steps[idx][
                                1
                            ].non_longitudinal_features = (
                                self._non_longitudinal_features
                            )
                        if hasattr(
                            hyperparameters["estimator"].steps[idx][1],
                            "feature_list_names",
                        ):
                            hyperparameters["estimator"].steps[idx][
                                1
                            ].feature_list_names = self._feature_list_names

            self._system = self.system_hyperparameters["custom_system"](
                **hyperparameters
            )

            if hasattr(self._system, "features_group"):
                self._system.features_group = self._features_group
            if hasattr(self._system, "non_longitudinal_features"):
                self._system.non_longitudinal_features = self._non_longitudinal_features
            if hasattr(self._system, "feature_list_names"):
                self._system.feature_list_names = self._feature_list_names

            if self.imbalanced_scenario == "random_under_sampling":
                self._system = Pipeline(
                    steps=[
                        (
                            "balancing_class_distrib",
                            RandomUnderSampler(
                                random_state=self.setup_data_parameters["random_state"]
                            ),
                        ),
                        ("classifier", self._system),
                    ],
                )
            elif self.imbalanced_scenario == "under_bagging":
                self._system = Pipeline(
                    steps=[
                        (
                            "balancing_class_distrib",
                            BalancedBaggingClassifier(
                                base_estimator=self._system,
                                n_estimators=10,
                                random_state=self.setup_data_parameters["random_state"],
                            ),
                        )
                    ],
                )
            elif self.imbalanced_scenario == "easy_ensemble":
                self._system = Pipeline(
                    steps=[
                        (
                            "balancing_class_distrib",
                            EasyEnsembleClassifier(
                                base_estimator=self._system,
                                n_estimators=10,
                                random_state=self.setup_data_parameters["random_state"],
                            ),
                        )
                    ],
                )
            elif self.imbalanced_scenario == "smote":
                self._system = Pipeline(
                    steps=[
                        (
                            "balancing_class_distrib",
                            SMOTE(
                                random_state=self.setup_data_parameters["random_state"]
                            ),
                        ),
                        ("classifier", self._system),
                    ],
                )
            else:
                print(f"Imbalanced scenario: {self.imbalanced_scenario}")
                print(
                    "Will skip the imbalanced scenario setup given the system should already be setup with."
                )

            print(f"Custom system loaded: {self._system}")
        else:
            from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier

            self._system = GamaLongitudinalClassifier(
                features_group=self._features_group,
                non_longitudinal_features=self._non_longitudinal_features,
                feature_list_names=self._feature_list_names,
                max_total_time=self.system_hyperparameters["max_total_time"],
                max_eval_time=self.system_hyperparameters["max_eval_time"],
                scoring=self.system_hyperparameters["scoring"],
                verbosity=self.system_hyperparameters["verbosity"],
                search=self.system_hyperparameters["search"],
                n_jobs=self.system_hyperparameters["n_inner_jobs"],
                store=self.system_hyperparameters["store"],
                random_state=self.setup_data_parameters["random_state"],
                output_directory=f"{self.output_path}/fold_{self.fold_number}_"
                + time.strftime("%Y%m%d-%H%M%S"),
                max_memory_mb=self.system_hyperparameters["max_memory_mb"],
            )

        if type(self._system) is None:
            raise ValueError("System not loaded correctly.")

        required_methods = ["fit", "predict", "predict_proba"]
        for method in required_methods:
            if not hasattr(self._system, method):
                raise ValueError(f"The system does not have the method: {method}")

    def run_experiment(self) -> None:
        """
        Executes the experiment by fitting the configured system/algorithm to the training data.

        Raises:
            ValueError: If the system or reporter is not properly set up.
        """
        if type(self._system) is None:
            raise ValueError("System not loaded correctly.")
        if not self._reporter:
            raise ValueError("Reporter not loaded correctly.")

        self._system.fit(self._X_train, self._y_train)
        print("System's fitted.")

    def report_experiment(self) -> None:
        """
        Reports the results of the experiment for the current fold.

        Evaluates the system on the test data, calculates relevant metrics,
        and saves the results to a CSV file.

        Raises:
            ValueError: If the reporter or required experiment results are missing.
        """
        if not self._reporter:
            raise ValueError("Reporter not loaded correctly.")

        experiment_results = self._reporter(self._system, self._X_test)

        required_results = [
            "predictions",
            "probability_predictions",
            "best_pipeline",
            "metric_optimised",
        ]
        for result in required_results:
            if result not in experiment_results:
                raise ValueError(f"Experiment results missing: {result}")

        fold_number = self.fold_number
        y_test = self._y_test
        predictions = experiment_results.get("predictions")
        probability_predictions = experiment_results.get("probability_predictions")
        best_pipeline = experiment_results.get("best_pipeline")
        metric_optimised = experiment_results.get("metric_optimised")
        y_true = y_test
        y_pred = predictions
        y_prob = probability_predictions

        auroc = roc_auc_score(y_true, y_prob[:, 1])
        auprc = average_precision_score(y_true, y_prob[:, 1])
        (
            precision_macro_avg,
            recall_macro_avg,
            f1_macro_avg,
            _,
        ) = precision_recall_fscore_support(y_true, y_pred, average="macro")
        (
            precision_pos_class,
            recall_pos_class,
            f1_pos_class,
            _,
        ) = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1
        )

        geo_mean = geometric_mean_score(y_true, y_pred)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        csv_file_path = os.path.join(self.output_path, "experiment_results.csv")
        fieldnames = [
            "Fold",
            "AUPRC",
            "AUROC",
            "Macro AVG Precision",
            "Macro AVG Recall",
            "Macro AVG F1 Score",
            "Precision (Positive Class)",
            "Recall (Positive Class)",
            "F1 Score (Positive Class)",
            "Data Preparation",
            "Preprocessor",
            "Classifier",
            "Metric Optimised",
            "Geometric Mean",
        ]

        if not os.path.isfile(csv_file_path):
            with open(csv_file_path, mode="w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
        with open(csv_file_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(
                {
                    "Fold": fold_number,
                    "AUPRC": auprc,
                    "AUROC": auroc,
                    "Macro AVG Precision": precision_macro_avg,
                    "Macro AVG Recall": recall_macro_avg,
                    "Macro AVG F1 Score": f1_macro_avg,
                    "Precision (Positive Class)": precision_pos_class,
                    "Recall (Positive Class)": recall_pos_class,
                    "F1 Score (Positive Class)": f1_pos_class,
                    "Data Preparation": best_pipeline["data_preparation"],
                    "Preprocessor": best_pipeline["preprocessor"],
                    "Classifier": best_pipeline["classifier"],
                    "Metric Optimised": metric_optimised,
                    "Geometric Mean": geo_mean,
                }
            )

        print(f"Fold {fold_number} results saved to: {csv_file_path}")
