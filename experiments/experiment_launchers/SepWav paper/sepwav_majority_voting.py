import argparse
import os
from typing import Union

import pandas as pd
from imblearn.pipeline import Pipeline
from scikit_longitudinal.data_preparation import SepWav
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
)
from sklearn.ensemble import RandomForestClassifier

from experiments.experiment_engine import ExperimentEngine

class Launcher:
    def __init__(self, args):
        self.args = args

    def validate_parameters(self):
        """
        Validates the command-line arguments to ensure they are correct and usable.

        Raises:
            ValueError: If any parameter is invalid.
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
        Sets up and runs the experiment using SepWav with majority voting, incorporating the optimization metric.
        """
        opt_metric = self.args.opt_metric

        def _reporter_sepwav_majority_voting(system: Union[Pipeline, SepWav], X_test: pd.DataFrame) -> dict:
            """
            Reporter function for SepWav with majority voting using RandomForestClassifier as the base estimator.
            Extracts predictions, probabilities, and best pipeline configuration, including the optimization metric.

            Args:
                system (Union[Pipeline, SepWav]): The fitted SepWav system or pipeline.
                X_test (pd.DataFrame): The test dataset.

            Returns:
                dict: A dictionary containing predictions, probability predictions, and pipeline details.

            Raises:
                ValueError: If the system is not fitted.
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
            else:
                sampler_name = 'No Resampling'

            return {
                "predictions": predictions,
                "probability_predictions": probability_predictions,
                "best_pipeline": {
                    "data_preparation": "SepWav",
                    "preprocessor": "None",
                    "classifier": f"{sampler_name} & SepWav Random Forest with majority voting",
                },
                "metric_optimised": opt_metric
            }

        # Configure the base estimator
        base_estimator = RandomForestClassifier(
            class_weight='balanced' if self.args.sampler == 'class_weight' else None
        )

        # Map sampler argument to imbalanced scenario
        if self.args.sampler == "undersample":
            imbalance_scenario = "random_under_sampling"
        elif self.args.sampler == "oversample":
            imbalance_scenario = "smote"
        else:
            imbalance_scenario = "vanilla"

        # Set up the ExperimentEngine
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
                "voting": LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
            },
            system_reporter=_reporter_sepwav_majority_voting,
        )
        experiment.run_experiment()
        experiment.report_experiment()

    def default_parameters(self):
        """
        Sets default values for parameters if not provided via command-line arguments.
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
    parser = argparse.ArgumentParser(description="SepWav with Majority Voting and Class Imbalance Handling")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--export_name", type=str, required=True)
    parser.add_argument("--fold_number", type=int, required=True)
    parser.add_argument("--n_outer_splits", nargs='?', type=int)
    parser.add_argument("--random_state", nargs='?', type=int)
    parser.add_argument("--shuffling", nargs='?', type=bool)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--sampler", nargs='?', type=str)
    parser.add_argument("--opt_metric", nargs='?', type=str)

    args = parser.parse_args()

    runner = Launcher(args)
    runner.default_parameters()
    runner.validate_parameters()
    runner.launch_experiment()

if __name__ == "__main__":
    main()