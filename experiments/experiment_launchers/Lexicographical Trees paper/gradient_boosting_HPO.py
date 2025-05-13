import argparse
import os
from typing import List, Any

import pandas as pd
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scikit_longitudinal.data_preparation import LongitudinalDataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from experiments.experiment_engine import ExperimentEngine


def _load_dataset(
        dataset_file_path: str,
        target_column: str,
        random_state: int,
        remove_target_waves: bool = True,
        setup_features_group: str = "elsa"
) -> (pd.DataFrame, pd.Series, List[List[int]], List[int], List[str]):
    """
    Loads the dataset from the specified file path, processes it, and prepares it for use in the experiment.

    Args:
        dataset_file_path (str): Path to the dataset file.
        target_column (str): The target column in the dataset.
        random_state (int): Seed for reproducibility.
        remove_target_waves (bool, optional): Whether to remove target waves from the dataset. Defaults to True.
        setup_features_group (str, optional): The method to set up the features group. Defaults to "elsa".

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

    # Check for missing values or "?"
    if longitudinal_dataset.data.isnull().values.any() or any(
            [
                any([value == "?" for value in longitudinal_dataset.data[column]])
                for column in longitudinal_dataset.data.columns
            ]
    ):
        if longitudinal_dataset.data[target_column].isnull().values.any() or any(
                [
                    value == "?"
                    for value in longitudinal_dataset.data[target_column]
                ]
        ):
            raise ValueError("Data has missing values in the target column.")
        print("Data has missing values. Filling them with the mean.")
        longitudinal_dataset.set_data(
            longitudinal_dataset.data.replace("?", pd.NA).apply(pd.to_numeric, errors="ignore")
        )
        longitudinal_dataset.set_data(
            longitudinal_dataset.data.fillna(longitudinal_dataset.data.mean())
        )
        longitudinal_dataset.load_data()

    longitudinal_dataset.load_data_target_train_test_split(
        target_column=target_column,
        remove_target_waves=remove_target_waves,
        random_state=random_state
    )
    longitudinal_dataset.setup_features_group(setup_features_group)

    return (
        longitudinal_dataset.data,
        longitudinal_dataset.target,
        longitudinal_dataset.feature_groups(),
        longitudinal_dataset.non_longitudinal_features(),
        longitudinal_dataset.data.columns.tolist()
    )


def gmean_scorer(y_true, y_pred):
    """
    Custom scorer for Geometric Mean.
    """
    return geometric_mean_score(y_true, y_pred)


def _reporter_gradient_boosting_gridsearch(system: GridSearchCV, X_test: pd.DataFrame, opt_metric: str) -> dict[
    str, Any]:
    """
    Reporter function for GradientBoostingClassifier with GridSearchCV.
    Extracts predictions, probabilities, and best pipeline configuration.
    Also computes both AUROC and GMean for comprehensive evaluation.

    Args:
        system (GridSearchCV): The fitted GridSearchCV system.
        X_test (pd.DataFrame): The test dataset.
        opt_metric (str): The optimization metric used.

    Returns:
        dict[str, Any]: A dictionary containing predictions, probability predictions,
                        best pipeline configuration, and the optimized metric.
    """
    if not system:
        raise ValueError("System's (GridSearchCV) not fitted yet.")

    best_estimator = system.best_estimator_
    predictions = best_estimator.predict(X_test)
    probability_predictions = best_estimator.predict_proba(X_test)
    best_params = system.best_params_

    print(f"CV Results: \n{system.cv_results_}\n")

    sampler = best_estimator.named_steps['sampler']
    if sampler == 'passthrough':
        sampler_name = 'No Resampling'
    else:
        sampler_name = sampler.__class__.__name__

    return {
        "predictions": predictions,
        "probability_predictions": probability_predictions,
        "best_pipeline": {
            "data_preparation": "MerWavTimePlus",
            "preprocessor": "None",
            "classifier": f"{sampler_name} & GB with params {best_params}"
        },
        "metric_optimised": opt_metric,
    }


class Launcher:
    def __init__(self, args):
        self.args = args
        self._num_features = None
        self._X_train = None
        self._y_train = None
        self._features_group = None
        self._non_longitudinal_features = None
        self._feature_list_names = None

    def validate_parameters(self):
        """
        Validates the command-line arguments.
        """
        if not os.path.exists(self.args.dataset_path):
            raise ValueError("Dataset path does not exist.")
        if not isinstance(self.args.export_name, str):
            raise ValueError("Export name must be a string.")
        if not isinstance(self.args.fold_number, int) or self.args.fold_number <= 0:
            raise ValueError("Fold number must be a positive integer.")
        if not isinstance(self.args.n_outer_splits, int) or self.args.n_outer_splits <= 0:
            raise ValueError("n_outer_splits must be a positive integer.")
        if not isinstance(self.args.random_state, int) or self.args.random_state < 0:
            raise ValueError("Random state must be a non-negative integer.")
        if not isinstance(self.args.shuffling, bool):
            raise ValueError("Shuffling must be a boolean.")
        if self.args.sampler not in ['vanilla', 'undersample', 'oversample']:
            raise ValueError("Sampler must be one of: 'vanilla', 'undersample', 'oversample'.")
        if self.args.opt_metric not in ['AUROC', 'GMean']:
            raise ValueError("opt_metric must be either 'AUROC' or 'GMean'.")

    def launch_experiment(self):
        """
        Loads the data, sets up the pipeline and GridSearchCV, and initiates the experiment.
        """
        (
            self._X_train,
            self._y_train,
            self._features_group,
            self._non_longitudinal_features,
            self._feature_list_names
        ) = _load_dataset(
            dataset_file_path=self.args.dataset_path,
            target_column=self.args.target_column,
            random_state=self.args.random_state,
            remove_target_waves=True,
            setup_features_group="elsa"
        )

        if self.args.sampler == 'undersample':
            sampler = RandomUnderSampler(random_state=self.args.random_state)
        elif self.args.sampler == 'oversample':
            sampler = SMOTE(random_state=self.args.random_state)
        else:
            sampler = 'passthrough'

        pipeline = Pipeline([
            ('sampler', sampler),
            ('classifier', GradientBoostingClassifier(
                learning_rate=0.1,  # Will be overwritten by grid search
                max_depth=3,  # Will be overwritten by grid search
                random_state=self.args.random_state
            ))
        ])

        param_grid = {
            'classifier__learning_rate': [0.05, 0.1, 0.2, 0.3],
            'classifier__max_depth': [3, 4, 5, 6]
        }

        if self.args.opt_metric == 'AUROC':
            scoring = 'roc_auc'
        elif self.args.opt_metric == 'GMean':
            scoring = make_scorer(gmean_scorer)
        else:
            raise ValueError("Invalid opt_metric")

        experiment = ExperimentEngine(
            output_path=self.args.export_name,
            imbalanced_scenario="vanilla",
            fold_number=self.args.fold_number,
            setup_data_parameters={
                "dataset_file_path": self.args.dataset_path,
                "random_state": self.args.random_state,
                "target_column": self.args.target_column,
                "shuffling": self.args.shuffling,
                "n_outer_splits": self.args.n_outer_splits,
            },
            system_hyperparameters={
                "custom_system": GridSearchCV,
                "estimator": pipeline,
                "param_grid": param_grid,
                "scoring": scoring,
                "cv": StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=self.args.random_state
                ),
            },
            system_reporter=lambda system, X_test: _reporter_gradient_boosting_gridsearch(
                system,
                X_test,
                self.args.opt_metric
            ),
        )
        print("_" * 100)
        print("Print all combinations of HPs GridSearchCV will try (from experiment._system):")
        print(experiment._system.param_grid)
        print("_" * 100)
        experiment.run_experiment()
        experiment.report_experiment()

    def default_parameters(self):
        """
        Sets default parameters if not provided via command-line arguments.
        """
        default_parameters = {
            "n_outer_splits": 10,
            "random_state": 42,
            "shuffling": True,
            "sampler": 'vanilla',
            "opt_metric": 'AUROC'
        }
        for key, value in default_parameters.items():
            if getattr(self.args, key, None) is None:
                setattr(self.args, key, value)


def main():
    parser = argparse.ArgumentParser(
        description="Gradient Boosting with HPO and Class Imbalance Handling")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--export_name", type=str)
    parser.add_argument("--fold_number", type=int)
    parser.add_argument("--n_outer_splits", nargs='?', type=int)
    parser.add_argument("--random_state", nargs='?', type=int)
    parser.add_argument("--shuffling", nargs='?', type=bool)
    parser.add_argument("--target_column", type=str)
    parser.add_argument("--sampler", nargs='?', type=str)
    parser.add_argument("--opt_metric", nargs='?', type=str)

    args = parser.parse_args()

    runner = Launcher(args)
    runner.default_parameters()
    runner.validate_parameters()
    runner.launch_experiment()


if __name__ == "__main__":
    main()
