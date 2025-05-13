import json
import os
from typing import List, Dict

import pandas as pd


class ExperimentResultsReporter:
    def __init__(self, config: Dict):
        if any(key not in config for key in ['root_placeholder', 'datasets', 'techniques', 'metric']):
            raise ValueError("Configuration must contain keys: 'root_placeholder', 'datasets', 'techniques', 'metric'")

        self.root_placeholder = config['root_placeholder']
        self.datasets = config['datasets']
        self.techniques = config['techniques']
        self.metric = config['metric']

    def merge_experiment_results(self, root_folders: List[str]) -> None:
        """
        Merge all experiment_results.csv files from the specified root folders into a single CSV file
        for each root folder.

        Args:
            root_folders (List[str]): List of root folders containing the experiment fold subfolders.

        Returns:
            None
        """
        for root_folder in root_folders:
            all_data = []

            for fold_name in os.listdir(root_folder):
                fold_path = os.path.join(root_folder, fold_name)
                if os.path.isdir(fold_path) and fold_name.startswith('fold_'):
                    csv_path = os.path.join(fold_path, 'experiment_results.csv')
                    if os.path.isfile(csv_path):
                        df = pd.read_csv(csv_path)
                        all_data.append(df)

            if all_data:
                merged_df = pd.concat(all_data, ignore_index=True)
                merged_df = merged_df.sort_values(by='Fold')
                output_path = os.path.join(root_folder, 'merged_experiment_results.csv')
                merged_df.to_csv(output_path, index=False)
                print(f"Merged CSV saved to {output_path}")

    def compute_average_metric(self, file_path: str) -> float:
        """Compute the average of the specified metric from a CSV file."""
        df = pd.read_csv(file_path)
        if self.metric not in df.columns:
            raise ValueError(f"Metric {self.metric} not found in the CSV file: {file_path}")
        return df[self.metric].mean()

    def gather_experiment_results(self) -> pd.DataFrame:
        """
        Gather experiment results from folders, compute average of the specified metric,
        and structure the results into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing datasets as rows and techniques as columns with the average metric.
        """
        results = {}

        for dataset in self.datasets:
            dataset_results = {}
            dataset_root = f"{self.root_placeholder}_{dataset}"

            for folder_name in os.listdir():
                if os.path.isdir(folder_name) and folder_name.startswith(dataset_root):
                    parts = folder_name.split('_')
                    if len(parts) > 2:
                        technique = '_'.join(parts[2:])
                        if "CORE" in technique:
                            technique = technique.replace("_CORE_", "_")
                        csv_path = os.path.join(folder_name, 'merged_experiment_results.csv')
                        if os.path.isfile(csv_path):
                            avg_metric = self.compute_average_metric(csv_path)
                            dataset_results[technique.split("_", 1)[1]] = avg_metric
                        else:
                            raise FileNotFoundError(f"CSV file not found: {csv_path}")
                    else:
                        raise ValueError(f"Invalid folder name: {folder_name}")

            dataset_name = dataset
            if "CORE" in dataset_name:
                dataset_name = dataset_name.replace("_CORE", "")
                dataset_name = f"EC_{dataset_name}"
            else:
                dataset_name = f"EN_{dataset_name}"
            results[dataset_name] = dataset_results

        return pd.DataFrame(results).T

    def run(self) -> None:
        """
        Run the experiment result merging and aggregation process.
        """
        root_folders = [
            f"{self.root_placeholder}_{dataset}_{technique}"
            for dataset in self.datasets
            for technique in self.techniques
        ]
        self.merge_experiment_results(root_folders)
        final_df = self.gather_experiment_results()
        final_df.to_csv(f'final_merged_experiment_results_{self.metric}.csv', index=True)
        print(f"Final merged CSV saved to final_merged_experiment_results_{self.metric}.csv")


def load_config(config_path: str) -> Dict:
    """
    Load the configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        Dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def main(config_path: str) -> None:
    """
    Main function to initialize and run the ExperimentResultsReporter.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        None
    """
    config = load_config(config_path)
    reporter = ExperimentResultsReporter(config)
    reporter.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Experiment Results Reporter')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()

    main(args.config)
