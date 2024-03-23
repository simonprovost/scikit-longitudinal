import os
import time
from typing import List

import numpy as np
import pandas as pd
import ray
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold

from scikit_longitudinal.data_preparation import LongitudinalDataset


class ScikitLongitudinalExperiment:
    """
    A class to explore time training for Scikit-Longitudinal algorithms using nested cross-validation

    Attributes:
        dataset (pd.DataFrame):
            The full input dataset.
        target (pd.Series):
            The target values.
        estimator (BaseEstimator):
            The candidate algorithm for training and evaluation.
        k_outer_folds (int):
            Number of outer folds for cross-validation.
        k_inner_folds (int):
            Number of inner folds for cross-validation.
        inner_fold_times (List[float]):
            Time taken for each inner fold of each outer fold during nested cross-validation.
        total_time (float):
            Total time taken for all inner folds combined during nested cross-validation.

    Methods:
        train():
            Performs nested cross-validation and records times.
        report():
            Generates a markdown report of the experiment results.
    """

    def __init__(self, dataset: pd.DataFrame, target: pd.Series, estimator: BaseEstimator,
                 k_outer_folds: int, k_inner_folds: int):
        """
        Initialises the experiment with the full dataset, target, estimator, and cross-validation settings.

        Parameters:
            dataset (pd.DataFrame):
                The full input dataset.
            target (pd.Series):
                The target values.
            estimator (BaseEstimator):
                The candidate algorithm for training and evaluation.
            k_outer_folds (int):
                Number of outer folds for cross-validation.
            k_inner_folds (int):
                Number of inner folds for cross-validation.
        """
        self.dataset = dataset
        self.target = target
        self.estimator = estimator
        self.k_outer_folds = k_outer_folds
        self.k_inner_folds = k_inner_folds
        self.inner_fold_times = []
        self.total_time = 0.0

    def train(self) -> None:
        """
        Performs nested cross-validation on the estimator using the provided data and settings.
        Records the time taken for each inner fold and the total time.
        """
        if not self.dataset.index.equals(self.target.index):
            raise ValueError("The dataset and target indices do not match.")
        if len(self.dataset) != len(self.target):
            raise ValueError("The dataset and target do not have the same number of samples.")
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The estimator does not have a fit method.")
        if not isinstance(self.k_outer_folds, int) or not isinstance(self.k_inner_folds, int):
            raise ValueError("The number of outer and inner folds must be integers.")
        dataset = self.dataset.copy()
        target = self.target.copy()
        outer_cv = KFold(n_splits=self.k_outer_folds)
        inner_cv = KFold(n_splits=self.k_inner_folds)

        start_time = time.time()
        for train_index, test_index in outer_cv.split(dataset):
            X_train_fold, X_test_fold = dataset.iloc[train_index], dataset.iloc[test_index]
            Y_train_fold, Y_test_fold = target.iloc[train_index], target.iloc[test_index]

            inner_fold_start_time = time.time()
            cross_val_score(self.estimator, X_train_fold, Y_train_fold, cv=inner_cv)
            inner_fold_end_time = time.time()

            self.inner_fold_times.append(inner_fold_end_time - inner_fold_start_time)

        self.total_time = time.time() - start_time

    def report(self) -> str:
        """
        Generates a markdown report of the experiment results.

        Returns:
            str: A markdown string with the experiment results.
        """
        report_md = f"""
        # Experiment Report For Algorithm: {self.estimator.__class__.__name__}

        - **Nested Cross-Validation Settings**:
            - Outer Folds: {self.k_outer_folds}
            - Inner Folds: {self.k_inner_folds}

        - **Timing**:
            - Inner Folds Times: {self.inner_fold_times}
            - Minimum Time for Inner Folds: {round(number=min(self.inner_fold_times), ndigits=3)} seconds
            - Maximum Time for Inner Folds: {round(number=max(self.inner_fold_times), ndigits=3)} seconds
            - Average Time for Inner Folds: {round(number=np.mean(self.inner_fold_times), ndigits=3)} seconds
            - Total Time for Inner Folds: {round(number=self.total_time, ndigits=3)} seconds
        """
        return report_md


@ray.remote
def ray_evaluate_estimator(estimator, dataset, target, k_outer_folds, k_inner_folds):
    experiment = ScikitLongitudinalExperiment(
        dataset=dataset,
        target=target,
        estimator=estimator,
        k_outer_folds=k_outer_folds,
        k_inner_folds=k_inner_folds
    )
    experiment.train()
    report = experiment.report()
    print(f"Finished evaluating {estimator.__class__.__name__}")
    print(f"Report: {report}")
    return report


class EmpiricalEvaluation:
    """
    A class to handle training time evaluation of a list of estimators with a given dataset.
    """

    def __init__(self, estimators: List[BaseEstimator], dataset_path: str, target_column: str, k_outer_folds: int,
                 k_inner_folds: int, n_jobs: int, export_name: str = ""):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.k_outer_folds = k_outer_folds
        self.k_inner_folds = k_inner_folds
        self.estimators = []
        self.reports = []
        self.n_jobs = n_jobs
        self.export_name = export_name

        self.dataset, self.target, self.feature_groups, self.non_longitudinal_features = self.load_dataset()

        for estimator in estimators:
            if hasattr(estimator, "features_group"):
                estimator.features_group = self.feature_groups
            if hasattr(estimator, "non_longitudinal_features"):
                estimator.non_longitudinal_features = self.non_longitudinal_features
            self.estimators.append(estimator)

    def load_dataset(self):
        """
            Loads a dataset from a given path and identifies the target column.

            Parameters:
                path (str): The file path to the dataset.
                target_column (str): The name of the target column in the dataset.

            Returns:
                DataFrame: The loaded dataset.
                Series: The target column.
                List[List[int]]: A list of lists containing the indices of the features in each group.
                List[Union[int, str]]: A list of indices or names of non-longitudinal features.
        """
        longitudinal_dataset = LongitudinalDataset(file_path=self.dataset_path)
        longitudinal_dataset.load_data_target_train_test_split(
            target_column=self.target_column,
            remove_target_waves=True,
            random_state=42,
        )
        longitudinal_dataset.setup_features_group("elsa")

        return (
            longitudinal_dataset.data,
            longitudinal_dataset.target,
            longitudinal_dataset.feature_groups(),
            longitudinal_dataset.non_longitudinal_features()
        )

    def evaluate_estimator(self, estimator: BaseEstimator) -> str:
        experiment = ScikitLongitudinalExperiment(
            dataset=self.dataset,
            target=self.target,
            estimator=estimator,
            k_outer_folds=self.k_outer_folds,
            k_inner_folds=self.k_inner_folds
        )
        experiment.train()
        return experiment.report()

    def start_empirical_evaluation(self):
        start_time = time.time()
        if self.n_jobs > 1:
            if ray.is_initialized():
                ray.shutdown()
            ray.init(ignore_reinit_error=True, num_cpus=self.n_jobs)

        dataset = self.dataset
        target = self.target
        k_outer_folds = self.k_outer_folds
        k_inner_folds = self.k_inner_folds

        if self.n_jobs > 1 and ray.available_resources().get('CPU', 0) > 1:
            evaluation_jobs = [
                ray_evaluate_estimator.remote(
                    estimator,
                    dataset,
                    target,
                    k_outer_folds,
                    k_inner_folds
                ) for estimator in self.estimators
            ]
            self.reports = ray.get(evaluation_jobs)
        else:
            for estimator in self.estimators:
                self.reports.append(self.evaluate_estimator(estimator))

        if ray.is_initialized():
            ray.shutdown()

        self.empirical_evaluation_time = time.time() - start_time

    def report_empirical_evaluation(self):
        current_directory = os.path.dirname(__file__)
        report_path = f"{current_directory}/empirical_evaluation_report_settings_{self.k_outer_folds}_{self.k_inner_folds}_{self.export_name}.md"
        num_samples = len(self.dataset)
        num_features = self.dataset.shape[1] if not self.dataset.empty else 0
        with open(report_path, 'w') as report_file:
            report_file.write(f"# Dataset Path: {self.dataset_path}\n")
            report_file.write(f"## Number of Samples: {num_samples}\n")
            report_file.write(f"## Number of Features: {num_features}\n\n")
            report_file.write(f"## Empirical Evaluation Time: {round(self.empirical_evaluation_time, 3)} seconds\n\n")
            for report in self.reports:
                report_file.write(report + "\n\n")
