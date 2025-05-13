import argparse
import os
from typing import List, Union

import pandas as pd
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation import SepWav
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from experiments.experiment_engine import ExperimentEngine


def _load_dataset(
        dataset_file_path: str,
        target_column: str,
        random_state: int,
        remove_target_waves: bool = True,
        setup_features_group="elsa"
) -> (pd.DataFrame, pd.Series, List[List[int]], List[int], List[str]):
    """
    Loads the dataset from the specified file path, processes it, and prepares it for use in the experiment.
    """
    if not os.path.exists(dataset_file_path):
        raise ValueError(f"The dataset path {dataset_file_path} does not exist.")

    longitudinal_dataset = LongitudinalDataset(file_path=dataset_file_path)
    longitudinal_dataset.load_data()
    if longitudinal_dataset.data.isnull().values.any() or any(
            [
                any([True for value in longitudinal_dataset.data[column] if value == "?"])
                for column in longitudinal_dataset.data.columns
            ]
    ):
        if longitudinal_dataset.data[target_column].isnull().values.any() or any(
                [
                    True for value in longitudinal_dataset.data[target_column]
                    if value == "?"
                ]
        ):
            raise ValueError("Data has missing values in the target column.")
        print("Data has missing values. Filling them with the mean.")
        longitudinal_dataset.set_data(
            longitudinal_dataset.data.replace("?", pd.NA
                                              ).apply(pd.to_numeric, errors="ignore")
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


def _reporter_sepwav(system: Union[Pipeline, SepWav], X_test: pd.DataFrame) -> dict:
    """
    Reporter function for SepWav with stacking using Logistic Regression.
    Extracts predictions, probabilities, and best pipeline configuration.
    """
    if not system:
        raise ValueError("System's (SepWav, Pipeline) not fitted yet.")

    predictions = system.predict(X_test)
    probability_predictions = system.predict_proba(X_test)

    if isinstance(system, Pipeline):
        if sampler := system.named_steps.get('balancing_class_distrib', None):
            sampler_name = sampler.__class__.__name__
    else:
        sampler_name = 'No Resampling'

    return {
        "predictions": predictions,
        "probability_predictions": probability_predictions,
        "best_pipeline": {
            "data_preparation": "SepWav",
            "preprocessor": "None",
            "classifier": f"{sampler_name} & SepWav Random Forest with stacking Logistic Regression",
        },
        "metric_optimised": "None"
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
        if self.args.sampler not in ['vanilla', 'undersample', 'oversample', 'class_weight']:
            raise ValueError("Sampler must be one of: 'vanilla', 'undersample', 'oversample', 'class_weight'.")
        if self.args.opt_metric not in ['AUROC', 'GMean']:
            raise ValueError("opt_metric must be either 'AUROC' or 'GMean'.")

    def launch_experiment(self):
        """
        Loads the data, sets up the pipeline, and initiates the experiment.
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

        # Build the SepWav classifier
        base_estimator = RandomForestClassifier(class_weight='balanced' if self.args.sampler == 'class_weight' else None)
        stacking_meta_learner = LogisticRegression(
            class_weight='balanced' if self.args.sampler == 'class_weight' else None
        )

        if self.args.sampler == "undersample":
            imbalance_scenario = "random_under_sampling"
        elif self.args.sampler == "oversample":
            imbalance_scenario = "smote"
        else:
            imbalance_scenario = "vanilla"

        experiment = ExperimentEngine(
            output_path=self.args.export_name,
            imbalanced_scenario=imbalance_scenario,
            fold_number=self.args.fold_number,
            setup_data_parameters={
                "dataset_file_path": self.args.dataset_path,
                "random_state": self.args.random_state,
                "target_column": self.args.target_column,
                "shuffling": self.args.shuffling,
                "n_outer_splits": self.args.n_outer_splits,
            },
            system_hyperparameters={
                "custom_system": SepWav,
                "estimator": base_estimator,
                "voting": LongitudinalEnsemblingStrategy.STACKING,
                "stacking_meta_learner": stacking_meta_learner,
            },
            system_reporter=_reporter_sepwav,
        )
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
    parser = argparse.ArgumentParser(description="SepWav with Stacking Logistic Regression and Class Imbalance Handling")
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