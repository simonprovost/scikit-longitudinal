from typing import Optional

from sklearn.base import BaseEstimator

from scikit_longitudinal.data_preparation import LongitudinalDataset


def print_message(message: str, title: Optional[str] = None, separator: str = "="):
    if title:
        print(separator * len(title))
        print(title)
    if message != "" and message is not None:
        print(message)
    if title:
        print(separator * len(title))
        print()


def extract_dataset_name(dataset: LongitudinalDataset, index: int) -> str:
    try:
        return dataset.file_path.split("/")[-1].split("_")[0]
    except AttributeError:
        return f"dataset_{index} (name not found)"


def get_type_name(obj: Optional[BaseEstimator]) -> str:
    return type(obj).__name__ if obj is not None else "None"
