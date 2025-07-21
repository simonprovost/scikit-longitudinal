import re
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# pylint: disable=W0212, R0902, W1514, E1101, R0904
def clean_padding(features_group: List[List[int]]) -> List[List[int]]:
    """clean_padding is a function that removes the padding from the feature groups.

    The primary objective of this function is to facilitate the removal of padding from the feature groups. This
    function is called after the feature groups have been updated with padding. The function removes the padding
    from the feature groups, leaving only the actual features. This is necessary in some cases, such as when
    performing feature selection, as the padding "-1" may interfere with the process.

    Args:
        features_group (List[List[int]]):
            The feature groups to update without padding.

    Returns:
        List[List[int]]:
            The feature groups updated without padding.

    """
    if features_group is not None:
        features_group = [[idx for idx in group if idx != -1] for group in features_group]

    return features_group


def validate_feature_groups(func: Callable) -> Callable:  # pragma: no cover
    """Decorator to validate the feature_groups parameter.

    Args:
        func (Callable):
            Function to decorate.

    Returns:
        Callable: Decorated function.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        input_data = kwargs.get("input_data", None) or args[1]
        if input_data is not None and (isinstance(input_data, str) and input_data.lower() != "elsa"):
            if not isinstance(input_data, list) or not all(isinstance(group, list) for group in input_data):
                raise ValueError(
                    "Invalid input for input_data. Expected None, 'elsa', or a list of lists of integers ",
                    "or strings.",
                )
            for group in input_data:
                if not all(isinstance(item, (int, str)) for item in group):
                    raise ValueError("Invalid input_data type. Expected a list of lists of integers or strings.")
        return func(*args, **kwargs)

    return wrapper


def ensure_data_loaded(func: Callable) -> Callable:  # pragma: no cover
    """Decorator to ensure that data is loaded before performing an operation.

    Args:
        func (Callable):
            Function to decorate.

    Returns:
        Callable: Decorated function.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0]
        if instance._data is None:
            raise ValueError("No data is loaded. Load data first.")
        return func(*args, **kwargs)

    return wrapper


def check_extension(allowed_extensions: List[str]):  # pragma: no cover
    """Decorator to check the file extension of the output_path parameter.

    Args:
        allowed_extensions (List[str]):
            List of allowed file extensions.

    Returns:
        Callable: Decorator function.

    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs.get("data_frame", None) is not None:
                return func(*args, **kwargs)
            file_path = kwargs.get("file_path", None) or kwargs.get("output_path", None) or args[1]
            if Path(file_path).suffix.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported file format: {file_path}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


class LongitudinalDataset:
    """LongitudinalDataset is the entry-point to manage longitudinal datasets for machine learning tasks in `Sklong`.

    The `LongitudinalDataset` class handles longitudinal data, offering robust data management and transformation tools
    to support machine learning algorithms designed for longitudinal classification tasks. Recall,
    Longitudinal datasets are, yes tabular, but in a sense that temporal information exists and live throughout.
    Therefore, the class is designed to manage this temporal information and provide a clean interface for
    machine learning tasks throughout the `Sklong` library.

    !!! question "Feature Groups and Non-Longitudinal Features"
        Two key attributes, `feature_groups` and `non_longitudinal_features`, enable algorithms to interpret the temporal
        structure of longitudinal data, we try to build those as much as possible for users, while allowing
        users to also define their own feature groups if needed. As follows:

        - **feature_groups**: A list of lists where each sublist contains indices of a longitudinal attribute's waves,
          ordered from oldest to most recent. This captures temporal dependencies.
        - **non_longitudinal_features**: A list of indices for static, non-temporal features excluded from the temporal
          matrix.

        Proper setup of these attributes is critical for leveraging temporal patterns effectively, and effectively
        use the primitives that follow.

        To see more, we highly recommend visiting the `Temporal Dependency` page in the documentation.
        [Temporal Dependency Guide :fontawesome-solid-timeline:](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/){ .md-button }

    Args:
        file_path (Union[str, Path]): Path to the dataset file (supports ARFF and CSV formats).
        data_frame (Optional[pd.DataFrame], optional): If provided, uses this DataFrame as the dataset, ignoring
            `file_path`.

    Attributes:
        data (pd.DataFrame): Read-only access to the loaded dataset.
        target (pd.Series): Read-only access to the target variable.
        X_train (np.ndarray): Read-only access to the training data.
        X_test (np.ndarray): Read-only access to the test data.
        y_train (pd.Series): Read-only access to the training target.
        y_test (pd.Series): Read-only access to the test target.

    Examples:
        Below are examples illustrating the class's usage.

        !!! example "Loading and Preparing Data"
            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Initialize with a file path
            dataset = LongitudinalDataset('./data/stroke.csv') # Replace with your file path

            # Load the data
            dataset.load_data()

            # Load the target variable
            dataset.load_target(target_column="stroke_w2")

            # Set up feature groups with the "elsa" strategy
            dataset.setup_features_group("elsa")

            # Split into train and test sets –– Uses sklearn's train_test_split
            dataset.load_train_test_split(test_size=0.2, random_state=42)
            ```

        !!! example "Using Custom Feature Groups"
            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Initialize with a file path
            dataset = LongitudinalDataset('./data/stroke.csv')

            # Load data and target in one step
            dataset.load_data_target_train_test_split(target_column="stroke_w2", test_size=0.2, random_state=42)

            # Define custom feature groups
            custom_groups = [[0, 1], [2, 3]]  # Indices for smoke and cholesterol waves

            # Set up feature groups
            dataset.setup_features_group(custom_groups)
            ```

        !!! example "Converting File Formats"
            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Initialize and load an ARFF file
            dataset = LongitudinalDataset('./data/elsa_core_dd.arff')
            dataset.load_data()

            # Convert to CSV
            dataset.convert('./data/elsa_core_dd.csv')
            ```
    """

    @check_extension([".csv", ".arff"])
    def __init__(self, file_path: Union[str, Path], data_frame: Optional[pd.DataFrame] = None):
        if data_frame is not None:
            self._data = data_frame
            self.file_path = None  # type: ignore
        else:
            self.file_path = Path(file_path) if file_path is not None else None  # type: ignore
            self._data = None
            if self.file_path and not self.file_path.is_file():
                raise FileNotFoundError(f"File not found: {self.file_path}")

        self._target = None
        self._feature_groups = None
        self._non_longitudinal_features = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

        if self._data is None and self.file_path is None:
            raise ValueError("Either file_path or data_frame must be provided.")

    def load_data(self) -> None:
        """Load data from the file into a pandas DataFrame.

        Supports `ARFF` and `CSV` formats. If a DataFrame was provided at initialization, this method does nothing.

        Raises:
            ValueError: If the file format is unsupported (only ARFF and CSV are allowed).
            FileNotFoundError: If the file specified in `file_path` does not exist.
        """
        if self._data is not None:
            return

        file_ext = self.file_path.suffix.lower()

        if file_ext == ".arff":
            self._data = self._arff_to_csv(self.file_path)  # type: ignore
        elif file_ext == ".csv":
            self._data = pd.read_csv(self.file_path)  # type: ignore
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Only ARFF and CSV are supported.")

    def load_target(
        self,
        target_column: str,
        target_wave_prefix: str = "class_",
        remove_target_waves: bool = False,
    ) -> None:
        """Extract the target variable from the dataset.

        Optionally removes other target-related columns (e.g., from different waves) if specified.

        Args:
            target_column (str): Name of the target column in the dataset.
            target_wave_prefix (str, optional): Prefix for target columns across waves. Defaults to "class_".
            remove_target_waves (bool, optional): If True, removes all columns with `target_wave_prefix` except
                `target_column`. Defaults to False.

        Raises:
            ValueError: If no data is loaded or `target_column` is not in the dataset.
        """
        if self._data is None:
            raise ValueError("No data is loaded. Load data first.")

        if target_column not in self._data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        if remove_target_waves:
            self._data = self._data[
                [col for col in self._data.columns if not (col.startswith(target_wave_prefix) and col != target_column)]
            ]

        self._target = self._data[target_column]
        self._data.drop(columns=[target_column], inplace=True)

    def load_train_test_split(self, test_size: float = 0.2, random_state: int = None) -> None:
        """Split the dataset into training and testing sets.

        Utilises `sklearn.model_selection.train_test_split` for the split.

        Args:
            test_size (float, optional): Proportion of data for the test set. Defaults to 0.2.
            random_state (int, optional): Seed for reproducible splitting. Defaults to None.

        Raises:
            ValueError: If data or target is not loaded.
        """
        if self._data is None or self._target is None:
            raise ValueError("No data or target is loaded. Load them first.")

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._data, self._target, test_size=test_size, random_state=random_state
        )

    def load_data_target_train_test_split(
        self,
        target_column: str,
        target_wave_prefix: str = "class_",
        remove_target_waves: bool = False,
        test_size: float = 0.2,
        random_state: int = None,
    ) -> None:
        """Load data, extract target, and split into train/test sets in one call.

        Combines `load_data`, `load_target`, and `load_train_test_split` for streamlined setup.

        Args:
            target_column (str): Name of the target column.
            target_wave_prefix (str, optional): Prefix for target columns across waves. Defaults to "class_".
            remove_target_waves (bool, optional): If True, removes other target wave columns. Defaults to False.
            test_size (float, optional): Proportion of data for the test set. Defaults to 0.2.
            random_state (int, optional): Seed for reproducible splitting. Defaults to None.

        Raises:
            ValueError: If data or target loading fails due to invalid inputs.
        """
        self.load_data()
        self.load_target(target_column, target_wave_prefix, remove_target_waves)
        self.load_train_test_split(test_size, random_state)

    @property
    def data(self) -> pd.DataFrame:
        """Access the loaded dataset.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        return self._data

    @property
    def target(self) -> pd.Series:
        """Access the target variable.

        Returns:
            pd.Series: The target variable as a pandas Series.
        """
        return self._target

    @property
    def X_train(self) -> np.ndarray:
        """Access the training data.

        Returns:
            np.ndarray: The training data as a NumPy array.
        """
        return self._X_train

    @property
    def X_test(self) -> np.ndarray:
        """Access the test data.

        Returns:
            np.ndarray: The test data as a NumPy array.
        """
        return self._X_test

    @property
    def y_train(self) -> pd.Series:
        """Access the training target.

        Returns:
            pd.Series: The training target as a pandas Series.
        """
        return self._y_train

    @property
    def y_test(self) -> pd.Series:
        """Access the test target.

        Returns:
            pd.Series: The test target as a pandas Series.
        """
        return self._y_test

    @staticmethod
    def _arff_to_csv(input_path: Union[str, Path]) -> pd.DataFrame:
        """Convert an ARFF file to a pandas DataFrame.

        !!! note "Disclaimer: This is handmade"
            If new libraries handle such conversion, we highly recommend using them instead of this
            handmade conversion. This is a neat and quick solution, but it is not the most efficient one, in
            our humble opinion.

        Args:
            input_path (Union[str, Path]): Path to the ARFF file.

        Returns:
            pd.DataFrame: The converted DataFrame.
        """

        def parse_row(line: str, row_len: int) -> List[Any]:
            line = line.strip()  # Strip the newline character
            if "{" in line and "}" in line:
                # Sparse data row
                line = line.replace("{", "").replace("}", "")
                row = np.zeros(row_len, dtype=object)
                for data in line.split(","):
                    index, value = data.split()
                    try:
                        row[int(index)] = float(value)
                    except ValueError:
                        row[int(index)] = np.nan if value == "?" else value.strip("'")
            else:
                # Dense data row
                row = [
                    (
                        float(value)
                        if value.replace(".", "", 1).isdigit()
                        else (np.nan if value == "?" else value.strip("'"))
                    )
                    for value in line.split(",")
                ]

            return row

        def extract_columns_and_data_start_index(file_content: List[str]) -> Tuple[List[str], int]:
            columns = []
            len_attr = len("@attribute")

            for i, line in enumerate(file_content):
                if line.lower().startswith("@attribute "):
                    col_name = line[len_attr:].split()[0]
                    columns.append(col_name)
                elif line.lower().startswith("@data"):
                    return columns, i

            return columns, 0

        with open(input_path, "r") as fp:
            file_content = fp.readlines()

        columns, data_index = extract_columns_and_data_start_index(file_content)
        len_row = len(columns)
        rows = [parse_row(line, len_row) for line in file_content[data_index + 1 :]]
        return pd.DataFrame(data=rows, columns=columns)

    @staticmethod
    def _csv_to_arff(df: pd.DataFrame, relation_name: str) -> dict:  # pragma: no cover
        """Convert a DataFrame to an ARFF dictionary.

        !!! note "Disclaimer: This is handmade"
            If new libraries handle such conversion, we highly recommend using them instead of this
            handmade conversion. This is a neat and quick solution, but it is not the most efficient one, in
            our humble opinion.

        Args:
            df (pd.DataFrame): Input DataFrame.
            relation_name (str): Name for the ARFF relation.

        Returns:
            dict: ARFF dictionary with relation, attributes, and data.
        """
        df.fillna("?", inplace=True)

        return {
            "relation": relation_name,
            "attributes": [(col, df[col].dtype.name) for col in df.columns],
            "data": df.values.tolist(),
        }

    @ensure_data_loaded
    @check_extension([".csv", ".arff"])
    def convert(self, output_path: Union[str, Path]) -> None:  # pragma: no cover
        """Convert the dataset to another format (ARFF or CSV).

        !!! note "Disclaimer: This is handmade"
            If new libraries handle such conversion, we highly recommend using them instead of this
            handmade conversion. This is a neat and quick solution, but it is not the most efficient one, in
            our humble opinion.

        Args:
            output_path (Union[str, Path]): Path to save the converted file.

        Raises:
            ValueError: If no data is loaded or the output format is unsupported.
        """
        if self._data is None:
            raise ValueError("No data to convert. Load data first.")

        file_ext = Path(output_path).suffix.lower()

        if file_ext == ".arff":
            arff_data = self._csv_to_arff(self._data, self.file_path.stem)
            with open(output_path, 'w') as f:
                arff.dump({
                    'description': '',
                    'relation': arff_data['relation'],
                    'attributes': arff_data['attributes'],
                    'data': arff_data['data']
                }, f)
        elif file_ext == ".csv":
            self._data.to_csv(output_path, index=False, na_rep="")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @ensure_data_loaded
    def save_data(self, output_path: Union[str, Path]) -> None:  # pragma: no cover
        """Save the dataset to a file.

        Wraps the `convert` method to save in the specified format.

        Args:
            output_path (Union[str, Path]): Path to save the file.

        Raises:
            ValueError: If no data is loaded.
        """
        if self._data is None:
            raise ValueError("No data to save. Load or convert data first.")

        self.convert(output_path)

    @validate_feature_groups
    def setup_features_group(self, input_data: Union[str, List[List[Union[str, int]]]]) -> None:
        """Configure feature groups and non-longitudinal features for longitudinal analysis.

        !!! question "What is a feature group? What's the structure really?"
            In a nutshell, a feature group is a collection of features sharing a common base longitudinal attribute
            across different waves of data collection (e.g., "income_wave1", "income_wave2", "income_wave3"). Note that
            aggregation reduces the dataset's temporal information significantly.

            Each sublist in `feature_groups` represents a longitudinal attribute across waves, ordered oldest to most
            recent (e.g., `[index_w1, index_w2]`). Use -1 for missing waves to align groups.

            To see more, we highly recommend visiting the `Temporal Dependency` page in the documentation.

            [Temporal Dependency Guide :fontawesome-solid-timeline:](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/){ .md-button }

        This method defines how features are grouped to capture temporal dependencies across waves. It supports three
        distinct input types, each suited to different use cases, with detailed examples and explanations below.

        === "Using `elsa` for Automatic Grouping"

            Automatically groups features based on wave suffixes (e.g., "_w1", "_w2") found in column
            names. This is ideal for datasets like the English Longitudinal Study of Ageing (ELSA), where features are
            consistently named with wave indicators. The ELSA dataset, focused on individuals aged 50+ in England,
            includes longitudinal data such as "smoke_w1", "smoke_w2", etc., which this method organizes into groups.

            !!! tip "Where to find those datasets"
                To find those datasets, feel free to open an issue and question us!

                [Open An Issue! :fontawesome-brands-square-github:](https://scikit-longitudinal.readthedocs.io/latest//issues){ .md-button }

            How It Works:

            - [x] Identifies base feature names (e.g., "smoke") and their wave suffixes (e.g., "_w1", "_w2").
            - [x] Creates groups with indices ordered from oldest to most recent wave, padding with -1 for missing waves.
            - [x] Non-longitudinal features (e.g., "age_wave8") are excluded from groups unless explicitly renamed.

            Example:

            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Load an ELSA dataset
            dataset = LongitudinalDataset('./data/elsa_core.csv')
            dataset.load_data()
            dataset.load_target("stroke_w2")

            # Automatically group features by wave suffixes
            dataset.setup_features_group("elsa")

            # Resulting groups might look like:
            # [[0, 1], [2, 3, -1]]  # e.g., [smoke_w1, smoke_w2], [chol_w1, chol_w2, N/A]
            print(dataset.feature_groups())  # Indices
            print(dataset.feature_groups(names=True))  # Names
            ```

            Use Case:

            Best for ELSA or similarly structured datasets with clear wave-based naming conventions.

        === "List of Lists of Integers for Direct Indices"

            Allows manual specification of feature indices for each group. This provides precise control
            over which columns are grouped together, useful when wave patterns are irregular or known in advance.

            How It Works:

            - [x] Each sublist contains integer indices corresponding to columns in the DataFrame.
            - [x] Order matters: indices should reflect temporal sequence (oldest to newest).
            - [x] Use -1 to pad groups if waves are missing, ensuring alignment across groups.

            Example:

            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Load a dataset
            dataset = LongitudinalDataset('./data/health.csv')
            dataset.load_data()

            # Define groups manually with indices
            custom_groups = [[0, 1], [2, 3, -1]]  # e.g., [bp_w1, bp_w2], [weight_w1, weight_w2, N/A]
            dataset.setup_features_group(custom_groups)

            # Verify setup
            print(dataset.feature_groups())  # [[0, 1], [2, 3, -1]]
            ```

            Use Case:

            Ideal when you have specific knowledge of column indices and need fine-grained control.

        === "List of Lists of Strings for Feature Names"

            Specifies feature groups using column names, which are then converted to indices. This is
            intuitive for users familiar with dataset feature names, enhancing readability and reducing errors.

            How It Works:

            - [x] Each sublist contains strings matching DataFrame column names.
            - [x] Names are mapped to their respective indices internally.
            - [x] No padding is needed in the input; alignment is handled post-conversion.

            Example:

            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Load a dataset
            dataset = LongitudinalDataset('./data/stroke.csv')
            dataset.load_data()

            # Define groups with feature names
            custom_names = [['smoke_w1', 'smoke_w2'], ['chol_w1', 'chol_w2']]
            dataset.setup_features_group(custom_names)

            # Verify setup
            print(dataset.feature_groups(names=True))  # [['smoke_w1', 'smoke_w2'], ['chol_w1', 'chol_w2']]
            print(dataset.feature_groups())  # Corresponding indices, e.g., [[0, 1], [2, 3]]
            ```

            Use Case:

            Perfect for datasets where feature names are meaningful and users prefer working with names
            over indices.

        ––––––––––––––––––––––––––––

        !!! question "Want more automatic handlers?"
            If you want more automatic handlers, like for the ELSA databases, feel free to open an issue and
            question us!

            [Open An Issue! :fontawesome-brands-square-github:](https://scikit-longitudinal.readthedocs.io/latest//issues){ .md-button }

        Args:
            input_data (Union[str, List[List[Union[str, int]]]]): Input to define feature groups:

                - [x] "elsa": Auto-groups based on wave suffixes.
                - [x] List[List[int]]: Feature indices.
                - [x] List[List[str]]: Feature names.

        Raises:
            ValueError: If `input_data` is invalid, feature names are missing, or groups lack sufficient waves.
        """
        if isinstance(input_data, str) and input_data.lower() == "elsa":
            self._feature_groups = self._create_elsa_feature_groups()
        elif isinstance(input_data, list) and all(isinstance(group, list) for group in input_data):
            if all(isinstance(item, int) for group in input_data for item in group):
                self._feature_groups = input_data
            elif all(isinstance(item, str) for group in input_data for item in group):
                self._feature_groups = self._convert_feature_names_to_indices(input_data)

        if self._feature_groups is None:
            raise ValueError(f"Invalid input data: {input_data} or unknown error has occurred.")

        for group in self._feature_groups:
            if len(group) == 1 or (len(group) == 2 and -1 in group):
                raise ValueError(
                    "A longitudinally represented feature should be in at least two waves: ",
                    group,
                )
        # Populate non_longitudinal_features
        feature_group_names = [name for group in self.feature_groups(names=True) for name in group]
        non_longitudinal_feature_names = set(self._data.columns) - set(feature_group_names)
        self._non_longitudinal_features = [self._data.columns.get_loc(name) for name in non_longitudinal_feature_names]

    @validate_feature_groups
    def _convert_feature_names_to_indices(self, feature_groups: List[List[str]]) -> List[List[int]]:
        """Convert feature names to their column indices.

        Args:
            feature_groups (List[List[str]]): Feature groups as lists of feature names.

        Returns:
            List[List[int]]: Feature groups as lists of indices.

        Raises:
            ValueError: If a feature name is not in the dataset.
        """
        column_indices = {col: i for i, col in enumerate(self._data.columns)}
        index_groups = []
        for group in feature_groups:
            index_group = []
            for feature_name in group:
                if feature_name not in column_indices:
                    raise ValueError(f"Feature name not found in dataset: {feature_name}")
                index_group.append(column_indices[feature_name])
            index_groups.append(index_group)

        return index_groups

    def _create_elsa_feature_groups(self) -> List[List[int]]:
        """Generate feature groups for the "elsa" strategy.

        Groups features by base name and wave suffix (e.g., "_w1", "_w2"), padding with -1 for alignment.

        Returns:
            List[List[int]]: Feature groups as indices, padded where necessary.
        """
        wave_columns = {}
        wave_suffix_pattern = re.compile(r"_w(\d+)$")
        max_wave = 0

        for idx, col_name in enumerate(self._data.columns):
            if match := wave_suffix_pattern.search(col_name):
                wave_num = int(match[1])
                base_name = col_name[: match.start()]
                if base_name not in wave_columns:
                    wave_columns[base_name] = []
                wave_columns[base_name].append((wave_num, idx))
                if wave_num > max_wave:
                    max_wave = wave_num

        feature_groups = []
        for columns in wave_columns.values():
            sorted_columns = sorted(columns, key=lambda x: x[0])
            padded_group = [-1] * max_wave
            for wave_num, idx in sorted_columns:
                padded_group[wave_num - 1] = idx
            feature_groups.append(padded_group)

        return feature_groups

    def feature_groups(self, names: bool = False) -> List[List[Union[int, str]]]:
        """Retrieve the feature groups.

        Returns -1 placeholders as "N/A" when `names=True`.

        Args:
            names (bool, optional): If True, returns feature names instead of indices. Defaults to False.

        Returns:
            List[List[Union[int, str]]]: Feature groups as indices or names.
        """
        if names:
            return [[self._data.columns[i] if i != -1 else "N/A" for i in group] for group in self._feature_groups]
        return self._feature_groups

    def non_longitudinal_features(self, names: bool = False) -> List[Union[int, str]]:
        """Retrieve the non-longitudinal features.

        Args:
            names (bool, optional): If True, returns feature names instead of indices. Defaults to False.

        Returns:
            List[Union[int, str]]: Non-longitudinal features as indices or names.
        """
        if names:
            return [self._data.columns[i] for i in self._non_longitudinal_features]
        return self._non_longitudinal_features

    def set_data(self, data: pd.DataFrame) -> None:
        """Sets the data attribute.

        Args:
            data (pd.DataFrame):
                The data.

        """
        self._data = data

    def set_target(self, target: pd.Series) -> None:
        """Sets the target attribute.

        Args:
            target (pd.Series):
                The target.

        """
        self._target = target

    def setX_train(self, X_train: pd.DataFrame) -> None:
        """Set the training data attribute.

        Args:
            X_train (pd.DataFrame):
                The training data.

        """
        self._X_train = X_train

    def setX_test(self, X_test: pd.DataFrame) -> None:
        """Set the test data attribute.

        Args:
            X_test (pd.DataFrame):
                The test data.

        """
        self._X_test = X_test

    def sety_train(self, y_train: pd.Series) -> None:
        """Set the training target data attribute.

        Args:
            y_train (pd.Series):
                The training target data.

        """
        self._y_train = y_train

    def sety_test(self, y_test: pd.Series) -> None:
        """Set the test target data attribute.

        Args:
            y_test (pd.Series):
                The test target data.

        """
        self._y_test = y_test
