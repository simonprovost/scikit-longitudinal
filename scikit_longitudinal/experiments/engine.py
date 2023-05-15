import os
import time
from collections import namedtuple
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.experiments.utils import get_type_name, print_message

DatasetInfo = namedtuple("DatasetInfo", ["file_path", "target_column"])


def check_data_loaded(func: Callable[..., Any]) -> Callable[..., Optional[Any]]:
    @wraps(func)
    def wrapper(self: "ExperimentEngine", *args: Any, **kwargs: Any) -> Optional[Any]:
        if not self.datasets:
            print("Error: Datasets have not been loaded yet. Please load datasets using create_datasets method.")
            return None
        return func(self, *args, **kwargs)

    return wrapper


def timing(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self: "ExperimentEngine", *args: Any, **kwargs: Any) -> Any:
        if not self.debug:
            return func(self, *args, **kwargs)
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        end = time.perf_counter()
        elapsed_time = end - start
        print(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


def validate_dataset_infos(func: Callable[[Any, List[DatasetInfo]], Any]) -> Callable[[Any, List[DatasetInfo]], Any]:
    @wraps(func)
    def wrapper(self: "ExperimentEngine", dataset_infos: List[DatasetInfo]) -> Any:
        if not all(isinstance(info, DatasetInfo) for info in dataset_infos):
            raise ValueError("All elements in dataset_infos must be instances of DatasetInfo")

        for info in dataset_infos:
            if not os.path.exists(info.file_path):
                raise FileNotFoundError(f"File not found: {info.file_path}")
            if not isinstance(info.target_column, str) or len(info.target_column) == 0:
                raise ValueError(f"Invalid target_column: {info.target_column}")

        return func(self, dataset_infos)

    return wrapper


class ExperimentEngine:
    def __init__(
        self,
        classifier: Union[Any],
        paper_results: Dict[str, float],
        preprocessor: Optional[Any] = None,
        postprocessor: Optional[Any] = None,
        debug: bool = False,
        random_state: int = 42,
    ) -> None:
        self.classifier: Union[Any] = classifier
        self.preprocessor: Optional[Any] = preprocessor
        self.postprocessor: Optional[Any] = postprocessor
        self.paper_results: Dict[str, float] = paper_results
        self.datasets: List[LongitudinalDataset] = []
        self.experiment_results: List["ExperimentResult"] = []  # noqa: F821
        self.compared_results: List[Tuple["ExperimentResult", bool, float]] = []  # noqa: F821
        self.debug: bool = debug
        self.random_state: int = random_state

    @timing
    @validate_dataset_infos
    def create_datasets(self, dataset_infos: List[DatasetInfo]) -> None:
        for dataset_info in dataset_infos:
            file_path = dataset_info.file_path
            target_column = dataset_info.target_column

            longitudinal_data = LongitudinalDataset(file_path)
            longitudinal_data.load_data_target_train_test_split(
                target_column=target_column,
                remove_target_waves=True,
                random_state=self.random_state,
            )
            longitudinal_data.setup_features_group(input_data="elsa")

            self.datasets.append(longitudinal_data)

    class ExperimentResult:
        def __init__(
            self,
            dataset_name: str,
            preprocessor_name: Optional[str],
            classifier_name: str,
            postprocessor_name: Optional[str],
            paper_result: float,
            implementation_result: float,
        ) -> None:
            self.dataset_name: str = dataset_name
            self.preprocessor_name: Optional[str] = preprocessor_name
            self.classifier_name: str = classifier_name
            self.postprocessor_name: Optional[str] = postprocessor_name
            self.paper_result: float = paper_result
            self.implementation_result: float = implementation_result

        def __str__(self) -> str:
            return (
                f"Dataset: {self.dataset_name}, "
                f"Preprocessor: {self.preprocessor_name}, "
                f"Classifier: {self.classifier_name}, "
                f"Postprocessor: {self.postprocessor_name}, "
                f"Paper result: {self.paper_result:.4f}, "
                f"Implementation result: {self.implementation_result:.4f}"
            )

    @timing
    @check_data_loaded
    def run_experiment(
        self,
        cv: int,
        preprocessor_pre_callback: Optional[Callable] = None,
        preprocessor_post_callback: Optional[Callable] = None,
        classifier_pre_callback: Optional[Callable] = None,
        classifier_post_callback: Optional[Callable] = None,
        postprocessor_pre_callback: Optional[Callable] = None,
        postprocessor_post_callback: Optional[Callable] = None,
    ) -> None:
        results = []
        f1_scorer = make_scorer(f1_score, average="macro")

        for i, dataset in enumerate(self.datasets):
            try:
                dataset_name = dataset.file_path.stem
            except AttributeError:
                dataset_name = f"dataset_{i} (name not found)"

            title = (
                f"Running experiment on {dataset_name} dataset with "
                f"[type={get_type_name(self.preprocessor)}] preprocessor and "
                f"[type={get_type_name(self.classifier)}] classifier and "
                f"[type={get_type_name(self.postprocessor)}] postprocessor"
            )
            print_message(message="", title=title)

            X_train, X_test, y_train, y_test = dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test

            if self.preprocessor:
                if preprocessor_pre_callback:
                    self.preprocessor = preprocessor_pre_callback(dataset, self.preprocessor)
                X_train = self.preprocessor.fit_transform(X_train, y_train)
                X_test = self.preprocessor.transform(X_test)
                if preprocessor_post_callback:
                    self.preprocessor = preprocessor_post_callback(dataset, self.preprocessor)

            if classifier_pre_callback:
                self.classifier = classifier_pre_callback(dataset, self.classifier)
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            cross_val_scores = cross_val_score(self.classifier, X_train, y_train, scoring=f1_scorer, cv=skf)
            results.append(np.mean(cross_val_scores))
            if classifier_post_callback:
                self.classifier = classifier_post_callback(dataset, self.classifier)

            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)

            if self.postprocessor:
                if postprocessor_pre_callback:
                    self.postprocessor = postprocessor_pre_callback(dataset, self.postprocessor)
                y_pred = self.postprocessor.fit_transform(y_pred)
                if postprocessor_post_callback:
                    self.postprocessor = postprocessor_post_callback(dataset, self.postprocessor)
            test_scorer = f1_score(y_test, y_pred, average="macro")

            print_message(message=f"{dataset_name} Classification Report:")
            print_message(message=classification_report(y_test, y_pred))
            print_message(message=f"Test Scorer: {test_scorer}")

            experiment_result = self.ExperimentResult(
                dataset_name=dataset_name,
                preprocessor_name=get_type_name(self.preprocessor),
                classifier_name=get_type_name(self.classifier),
                postprocessor_name=get_type_name(self.postprocessor),
                paper_result=self.paper_results[dataset_name],
                implementation_result=test_scorer,
            )

            self.experiment_results.append(experiment_result)

    @timing
    @check_data_loaded
    def run_comparison(self, window: float = 0.05) -> None:
        for experiment_result in self.experiment_results:
            our_result = experiment_result.implementation_result
            paper_result = experiment_result.paper_result

            is_within_range = paper_result - window <= our_result <= paper_result + window
            difference = our_result - paper_result

            self.compared_results.append((experiment_result, is_within_range, difference))

    @timing
    @check_data_loaded
    def print_comparison(self) -> None:
        print_message(message="", title="Comparison with the paper's results", separator="-")

        for experiment_result, is_within_range, difference in self.compared_results:
            dataset_name = experiment_result.dataset_name

            higher_or_lower = "higher" if difference > 0 else "lower"
            print_message(
                message=(
                    f"{dataset_name}: "
                    f"{'True' if is_within_range else 'False'} "
                    f"(Difference: {difference:.3f}, {higher_or_lower})"
                )
            )
        print_message(f"Number of true results: {len([x for x in self.compared_results if x[1]])}")
        print_message(f"Number of false results: {len([x for x in self.compared_results if not x[1]])}")
        print_message("")
