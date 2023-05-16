import re
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# pylint: disable=W0212, R0902, W1514, E1101, R0904


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
            file_path = kwargs.get("output_path", None) or args[1]
            if Path(file_path).suffix.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported file format: {file_path}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


class LongitudinalDataset:
    """LongitudinalDataset handle longitudinal datasets.

    It supports the importation, conversion, and storage of data in the ARFF and CSV formats.
    In addition, it permits the management of feature groups for longitudinal analysis and the separation of the
    dataset into train and test sets.

    Args:
        file_path (Union[str, Path]):
            Path to the dataset file. Could be ARFF or CSV the class handles both.

    Properties:
        data (pd.DataFrame):
            A read-only property that returns the loaded dataset as a pandas DataFrame.
        feature_groups (List[List[int]]):
            A list of lists containing the indices of the feature groups. This attribute should be set using the
            setup_features_group method.
        target (pd.Series):
            A read-only property that returns the target variable (class variable) as a pandas Series.

    Example:
        >>> from scikit_longitudinal.utils import LongitudinalDataset
        >>> input_file = './data/elsa_core_dd.arff'
        >>> output_file = './data/elsa_core_dd.csv'
        >>> dataset = LongitudinalDataset(input_file)
        >>> dataset.load_data()
        >>> dataset.setup_features_group("Elsa")
        >>> dataset.convert(output_file)

    """

    @check_extension([".csv", ".arff"])
    def __init__(self, file_path: Union[str, Path]):
        """Initialise the LongitudinalDataset object.

        Args:
            file_path (Union[str, Path]):
                Path to the dataset file. Could be ARFF or CSV the class handles both.

        Raises:
            FileNotFoundError: If the file specified in the file_path parameter does not exist.

        """
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = Path(file_path)
        self._data = None
        self._target = None
        self._feature_groups = None
        self._non_longitudinal_features = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    def load_data(self) -> None:
        """Loads data from the specified file into a pandas DataFrame.

        Raises:
            ValueError: If the file format is not supported. Only ARFF and CSV are supported.
            FileNotFoundError: If the file specified in the file_path parameter does not exist.

        """
        file_ext = self.file_path.suffix.lower()

        if file_ext == ".arff":
            self._data = self._arff_to_csv(self.file_path)
        elif file_ext == ".csv":
            self._data = pd.read_csv(self.file_path)

    def load_target(
        self,
        target_column: str,
        target_wave_prefix: str = "class_",
        remove_target_waves: bool = False,
    ) -> None:
        """Load the target from the dataset loaded in the object.

        Args:
            target_column (str):
                The name of the column in the dataset to be used as the target variable.
            target_wave_prefix (str, optional):
                The prefix of the columns that represent different waves of the target variable. Defaults to "class_".
            remove_target_waves (bool, optional):
                If True, all the columns with target_wave_prefix and the target_column will be removed from the dataset
                after extracting the target variable. Defaults to False.

        Raises:
            ValueError: If no data is loaded or the target_column is not found in the dataset.

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
        """Splits the data into training and testing sets and saves them as attributes.

        Args:
            test_size (float, optional):
                The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional):
                Controls the shuffling applied to the data before applying the split. Pass an int for reproducible
                output across multiple function calls. Defaults to None.

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
        """Loads data, target, and train-test split in one call.

        Args:
            target_column (str):
                The name of the column in the dataset to be used as the target variable.
            target_wave_prefix (str, optional):
                The prefix of the columns that represent different waves of the target variable. Defaults to "class_".
            remove_target_waves (bool, optional):
                If True, all the columns with target_wave_prefix and the target_column will be removed from the
                dataset after extracting the target variable. Defaults to False.
            test_size (float, optional):
                The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional):
                Controls the shuffling applied to the data before applying the split. Pass an int for reproducible
                output across multiple function calls. Defaults to None.

        """
        self.load_data()
        self.load_target(target_column, target_wave_prefix, remove_target_waves)
        self.load_train_test_split(test_size, random_state)

    @property
    def data(self) -> pd.DataFrame:
        """Returns the loaded dataset as a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded dataset.

        """
        return self._data

    @property
    def target(self) -> pd.Series:
        """Returns the target.

        Returns:
            pd.Series: The target.

        """
        return self._target

    @property
    def X_train(self) -> pd.DataFrame:
        """Get the training data.

        Returns:
            pd.DataFrame: The training data.

        """
        return self._X_train

    @property
    def X_test(self) -> pd.DataFrame:
        """Get the test data.

        Returns:
            pd.DataFrame: The test data.

        """
        return self._X_test

    @property
    def y_train(self) -> pd.Series:
        """Get the training target data.

        Returns:
            pd.Series: The training target data.

        """
        return self._y_train

    @property
    def y_test(self) -> pd.Series:
        """Get the test target data.

        Returns:
            pd.Series: The test target data.

        """
        return self._y_test

    @staticmethod
    def _arff_to_csv(input_path: Union[str, Path]) -> pd.DataFrame:
        """Converts an ARFF file to a DataFrame.

        Args:
            input_path (Union[str, Path]): Path to the input ARFF file.

        Returns:
            pd.DataFrame: Converted DataFrame.

        """

        def parse_row(line: str, row_len: int) -> List[Any]:
            """Parses a row of data from an ARFF file.

            Args:
                line (str): A row from the ARFF file.
                row_len (int): Length of the row.

            Returns:
                List[Any]: Parsed row as a list of values.

            """
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
            """Extracts column names and the index of the @data line from ARFF file content.

            Args:
                file_content (List[str]): List of lines from the ARFF file.

            Returns:
                Tuple[List[str], int]: List of column names and the index of the @data line.

            """
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
        """Converts a DataFrame to an ARFF dictionary.

        Args:
            df (pd.DataFrame):
                Input DataFrame.
            relation_name (str):
                Relation name for the ARFF file.

        Returns:
            dict: ARFF dictionary.

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
        """Converts the dataset between ARFF or CSV formats.

        Args:
            output_path (Union[str, Path]):
                Path to store the resulting file.

        """
        if self._data is None:
            raise ValueError("No data to convert. Load data first.")

        file_ext = Path(output_path).suffix.lower()

        if file_ext == ".arff":
            arff_data = self._csv_to_arff(self._data, self.file_path.stem)
            arff.dump(
                output_path,
                arff_data["data"],
                relation=arff_data["relation"],
                names=arff_data["attributes"],
            )
        elif file_ext == ".csv":
            self._data.to_csv(output_path, index=False, na_rep="")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @ensure_data_loaded
    def save_data(self, output_path: Union[str, Path]) -> None:  # pragma: no cover
        """Saves the DataFrame to the specified file format.

        Args:
            output_path (Union[str, Path]):
                Path to store the resulting file.

        """
        if self._data is None:
            raise ValueError("No data to save. Load or convert data first.")

        self.convert(output_path)

    @validate_feature_groups
    def setup_features_group(self, input_data: Union[str, List[List[Union[str, int]]]]) -> None:
        """Sets up the feature groups based on the input data and populates the non-longitudinal features attribute.

        Args:
            input_data (Union[str, List[List[Union[str, int]]]]):
                The input data for setting up the feature groups:
                    * If "elsa" is passed, it groups features based on their name and suffix "_w1", "_w2", etc.
                    * If a list of lists of integers is passed, it assigns the input directly to the feature groups
                    without modification.
                    * If a list of lists of strings (feature names) is passed, it converts the names to
                    indices and creates feature groups.

        Raises:
            ValueError: If input_data is not one of the expected types or if a feature name is not found in the dataset.

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
            if len(group) == 1:
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
        """Converts feature names to indices in the dataset's columns and returns the feature groups as lists of
        indices.

        Args:
            feature_groups (List[List[str]]):
                A list of lists of feature names.

        Returns:
            List[List[int]]: The corresponding feature groups represented as lists of column indices.

        Raises:
            ValueError: If a feature name is not found in the dataset.

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
        """Create feature groups for the "Elsa" case, where features are grouped based on their name and suffixes s.t
        "_w1", "_w2", etc.

        Returns:
            List[List[int]]: Feature groups using column indices.

        """
        wave_columns = {}
        wave_suffix_pattern = re.compile(r"_w(\d+)$")

        for idx, col_name in enumerate(self._data.columns):
            if match := wave_suffix_pattern.search(col_name):
                wave_num = int(match[1])
                base_name = col_name[: match.start()]
                if base_name not in wave_columns:
                    wave_columns[base_name] = []
                wave_columns[base_name].append((wave_num, idx))

        feature_groups = [sorted(columns, key=lambda x: x[0]) for columns in wave_columns.values()]
        return [[idx for _, idx in group] for group in feature_groups]

    def feature_groups(self, names: bool = False) -> List[List[Union[int, str]]]:
        """Returns the feature groups.

        Args:
            names (bool, optional):
                If True, the feature names will be returned instead of the indices. Defaults to False.

        Returns:
            List[List[Union[int, str]]]: The feature groups as a list of lists of feature names or indices.

        """
        if names:
            return [[self.data.columns[i] for i in group] for group in self._feature_groups]
        return self._feature_groups

    def non_longitudinal_features(self, names: bool = False) -> List[Union[int, str]]:
        """Returns the non longitudinal features.

        Args:
            names (bool, optional):
                If True, the feature names will be returned instead of the indices. Defaults to False.

        Returns:
            List[Union[int, str]]: The non longitudinal features as a list of feature names or indices.

        """
        if names:
            return [self.data.columns[i] for i in self._non_longitudinal_features]
        return self._non_longitudinal_features

    def set_data(self, data: pd.DataFrame) -> None:
        """Sets the data.

        Args:
            data (pd.DataFrame): The data.

        """
        self._data = data

    def set_target(self, target: pd.Series) -> None:
        """Sets the target.

        Args:
            target (pd.Series): The target.

        """
        self._target = target

    def setX_train(self, X_train: pd.DataFrame) -> None:
        """Set the training data.

        Args:
            X_train (pd.DataFrame): The training data.

        """
        self._X_train = X_train

    def setX_test(self, X_test: pd.DataFrame) -> None:
        """Set the test data.

        Args:
            X_test (pd.DataFrame): The test data.

        """
        self._X_test = X_test

    def sety_train(self, y_train: pd.Series) -> None:
        """Set the training target data.

        Args:
            y_train (pd.Series): The training target data.

        """
        self._y_train = y_train

    def sety_test(self, y_test: pd.Series) -> None:
        """Set the test target data.

        Args:
            y_test (pd.Series): The test target data.

        """
        self._y_test = y_test
