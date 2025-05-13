import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest import LexicoDeepForestClassifier, \
    LongitudinalEstimatorConfig, LongitudinalClassifierType

from experiments.experiment_engine import ExperimentEngine


def _reporter_lexico_deep_forest(system: LexicoDeepForestClassifier, X_test: pd.DataFrame) -> dict[str, Any]:
    """
    Reports the results of the fitted LexicoDeepForestClassifier algorithm on the test data.

    Args:
        system (LexicoDeepForestClassifier): The fitted LexicoDeepForestClassifier algorithm.
        X_test (pd.DataFrame): The test dataset.

    Returns:
        dict[str, Any]: A dictionary containing the following keys:
            - "predictions": Predictions made by the system.
            - "probability_predictions": Probability predictions made by the system.
            - "best_pipeline": A dictionary with the names of the techniques used in the best pipeline for data preparation, preprocessing, and classification.
            - "metric_optimised": The name of the metric that was optimised during training.

    Raises:
        ValueError: If the system is not fitted or if required pipeline steps are not found.
    """

    if not system:
        raise ValueError("System's (LexicoDeepForestClassifier) not fitted yet.")
    return {
        "predictions": system.predict(X_test),
        "probability_predictions": system.predict_proba(X_test),
        "best_pipeline": {
            "data_preparation": "MerWavTimePlus",
            "preprocessor": "None",
            "classifier": "LexicoDeepForestClassifier"
        },
        "metric_optimised": "None"
    }


class Launcher:
    def __init__(self, args):
        self.args = args

    def validate_parameters(self):
        if not os.path.exists(self.args.dataset_path):
            raise ValueError("Dataset path does not exist.")
        if not isinstance(self.args.export_name, str):
            raise ValueError("Export name must be a string.")
        if not isinstance(self.args.fold_number, int) or self.args.fold_number <= 0:
            raise ValueError("Fold number must be a positive integer.")
        if not isinstance(self.args.n_outer_splits, int) or self.args.n_outer_splits <= 0:
            raise ValueError("n_outer_splits must be a positive integer.")
        if not isinstance(self.args.random_state, int) or self.args.random_state <= 0:
            raise ValueError("Random state must be a positive integer.")
        if not isinstance(self.args.shuffling, bool):
            raise ValueError("Shuffling must be a boolean.")

    def launch_experiment(self):
        lexico_rf_config = LongitudinalEstimatorConfig(
            classifier_type=LongitudinalClassifierType.LEXICO_RF,
            count=2,
            hyperparameters={
                "n_estimators": 100,
                "threshold_gain": 0.0015,
                "random_state": np.random.randint(0, 1000),
                "class_weight": "balanced",
            },
        )

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
                "custom_system": LexicoDeepForestClassifier,
                "longitudinal_base_estimators": [lexico_rf_config],
            },
            system_reporter=_reporter_lexico_deep_forest,
        )
        experiment.run_experiment()
        experiment.report_experiment()

    def default_parameters(self):
        default_parameters = {
            "n_outer_splits": 5,
            # "random_state": 64,
            "random_state": 42,
            "shuffling": True,
        }
        for key, value in default_parameters.items():
            if getattr(self.args, key) is None:
                setattr(self.args, key, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--export_name", type=str)
    parser.add_argument("--fold_number", type=int)
    parser.add_argument("--n_outer_splits", nargs='?', type=int)
    parser.add_argument("--random_state", nargs='?', type=int)
    parser.add_argument("--shuffling", nargs='?', type=bool)
    parser.add_argument("--target_column", type=str)

    args = parser.parse_args()

    runner = Launcher(args)
    runner.default_parameters()
    runner.validate_parameters()
    runner.launch_experiment()


if __name__ == "__main__":
    main()
