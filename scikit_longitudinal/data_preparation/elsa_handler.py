# pylint: disable=E0401, W0621,

import argparse
import os

import arff
import pandas as pd


class ElsaDataHandler:
    """A class to handle ELSA (English Longitudinal Study of Ageing) data.

    The ELSA dataset is a comprehensive collection of longitudinal data that has been meticulously gathered from a
    carefully selected and representative sample of individuals within the English population. This dataset
    specifically focuses on individuals who are aged 50 years and older, ensuring that it captures a significant
    portion of the population that falls within this age range. The InputCSVReader class is responsible for reading
    the input CSV file and performing any necessary data preprocessing. It also handles the creation of datasets for
    each unique class found in the data. Finally, it saves the generated datasets in the specified file format and
    destination directory.

    Refer to the UK data service to obtain the ELSA dataset:
     https://beta.ukdataservice.ac.uk/datacatalogue/series/series?id=200011

    Attributes:
        df (pd.DataFrame):
            The dataframe containing the data.
        elsa_type (str):
            The type of ELSA dataset (core, Nurse).
        datasets (dict):
            A dictionary containing datasets for each unique class.

    Examples:
        # Initialize the handler with a CSV file path and ELSA dataset type - Note that Core here refers to the core
        ELSA dataset to distinguish it from other ELSA datasets such as Nurse, etc. hence, the preprocessing step is
        different for each dataset type.
        >>>> elsa_data_handler = ElsaDataHandler("path/to/csv_file.csv", "core")
        # Preprocess the data and create datasets for each unique class
        >>>> elsa_data_handler.create_datasets()
        # Save the datasets in the desired file format
        >>>> elsa_data_handler.save_datasets(dir_output="output/directory", file_format="csv")
        # Get a specific dataset by its class name
        >>>> dataset = elsa_data_handler.get_dataset("class_name")

    """

    def __init__(self, csv_path: str, elsa_type: str):
        """Initialises the ELSA_data_handler with a given CSV file path.

        Args:
            csv_path (str): The path to the CSV file to handle.

        """
        self.df = pd.read_csv(csv_path)

        if elsa_type.lower() not in ["core", "nurse"]:
            raise ValueError("Invalid ELSA dataset type. Valid types are: core, nurse")
        self.elsa_type = elsa_type
        self.datasets = {}

    def core_preprocessing(self) -> None:
        """Preprocesses the core dataset.

        The renaming of certain attributes from "longitudinal" to "non-longitudinal" has been implemented. Specifically,
        the attribute names have been updated from "wN" to "waveN" to reflect this change. The default behaviour of the
        LongitudinalDataset class is modified in order to prevent the automatic creation of a group for the features.
        Instead, a separate group is created for each of the non-longitudinal attributes that are specified.
        Furthermore, it is worth noting that certain attributes have been deemed superfluous for the purpose of
        classification. Specifically, the age attributes have been excluded, with particular emphasis on the final
        attribute (age_w8), as it pertains to the individual's age in the past and is therefore deemed irrelevant.

        """
        column_mapping = {
            "dicdnf_w7": "dicdnf_wave7",
            "dicdnm_w7": "dicdnm_wave7",
            "heiadlX-of-7_w3": "heiadlX-of-7_wave3",
            "memtot_w1": "memtot_wave1",
            "indager_w8": "indager_wave8",
        }
        self.df = self.df.rename(columns={col: column_mapping[col] for col in self.df.columns if col in column_mapping})

        columns_to_drop = [
            "idauniq",
            "indager_w1",
            "indager_w2",
            "indager_w3",
            "indager_w4",
            "indager_w5",
            "indager_w6",
            "indager_w7",
        ]
        self.df = self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns])

    def nurse_preprocessing(self) -> None:
        """Preprocesses the nurse dataset.

        The renaming of certain attributes from "longitudinal" to "non-longitudinal" has been implemented. Specifically,
        the attribute names have been updated from "wN" to "waveN" to reflect this change. The default behaviour of the
        LongitudinalDataset class is modified in order to prevent the automatic creation of a group for the features.
        Instead, a separate group is created for each of the non-longitudinal attributes that are specified.
        Furthermore, it is worth noting that certain attributes have been deemed superfluous for the purpose of
        classification. Specifically, the age attributes have been excluded, with particular emphasis on the final
        attribute (age_w8), as it pertains to the individual's age in the past and is therefore deemed irrelevant.

        """
        column_mapping = {
            "indager_w8": "indager_wave8",
            "apoe_w2": "apoe_wave2",
            "dheas_w4": "dheas_wave4",
        }

        self.df = self.df.rename(columns={col: column_mapping[col] for col in self.df.columns if col in column_mapping})

    def get_unique_classes(self) -> list:
        """Returns the unique classes from the dataframe.

        A class is denoted by the prefix "class_" followed by the class name. For example, "class_1" is a class name.

        Returns:
            list: A list of unique class names.

        """
        columns = self.df.columns
        unique_classes = []
        for col in columns:
            if col.startswith("class_"):
                class_name = col.split("_", 2)[1]
                if class_name not in unique_classes:
                    unique_classes.append(class_name)
        return unique_classes

    def create_datasets(self):
        """
        Creates datasets for each unique class and stores them in the datasets attribute.
        """
        unique_classes = self.get_unique_classes()

        if self.elsa_type.lower() == "core":
            self.core_preprocessing()
        elif self.elsa_type.lower() == "nurse":
            self.nurse_preprocessing()

        features = [col for col in self.df.columns if not col.startswith("class_")]
        for class_name in unique_classes:
            temp_columns = [col for col in self.df.columns if col.startswith(f"class_{class_name}")]
            dataset = self.df[features + temp_columns]
            self.datasets[class_name] = dataset

    def get_dataset(self, class_name: str) -> pd.DataFrame:
        """Returns the dataset corresponding to the given class name.

        Args:
            class_name (str):
                The name of the class.

        Returns:
            pd.DataFrame:
                The dataset corresponding to the class name, or None if the class does not exist.

        """
        return self.datasets.get(class_name, None)

    def save_datasets(self, dir_output: str = "tmp", file_format: str = "csv"):
        """Saves the datasets in the specified file format.

        Args:
            dir_output (str):
                The directory to save the datasets in.
            file_format (str):
                The file format to save the datasets in. Supported formats are "csv" and "arff".

        Raises:
            ValueError: If an unsupported file format is provided.

        """
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        for class_name, dataset in self.datasets.items():
            if file_format.lower() == "csv":
                dataset.to_csv(f"{dir_output}/{class_name}_dataset.csv", index=False)
            elif file_format.lower() == "arff":
                dataset.fillna("?", inplace=True)
                arff.dump(
                    f"{dir_output}/{class_name}_dataset.arff",
                    dataset.values,
                    relation=class_name,
                    names=dataset.columns,
                )
            else:
                raise ValueError(f"Unsupported file format: {file_format}")


def main(args):  # pragma: no cover
    """Main function for the ELSA data handler.

    Used by the makefile target rule named `make create_elsa_core_datasets`.

    """

    csv_path = args.csv_path
    file_format = args.file_format
    dir_output = args.dir_output
    elsa_type = args.elsa_type

    elsa_data_handler = ElsaDataHandler(csv_path, elsa_type)
    elsa_data_handler.create_datasets()
    elsa_data_handler.save_datasets(dir_output=dir_output, file_format=file_format)


def parse_arguments():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Process input arguments")
    parser.add_argument("--csv_path", required=True, help="Path to the input CSV file")
    parser.add_argument("--elsa_type", required=True, help="Type of Elsa dataset (core, Nurse, etc.)")
    parser.add_argument("--file_format", default="csv", help="Output file format (default: csv)")
    parser.add_argument("--dir_output", default="tmp", help="Output directory (default: tmp)")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = parse_arguments()
    main(args)
