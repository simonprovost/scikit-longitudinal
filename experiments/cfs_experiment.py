from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from scikit_longitudinal.experiments.engine import DatasetInfo, ExperimentEngine
from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import (
    CorrelationBasedFeatureSelectionPerGroup,
)


class CustomFeatureSelector:
    def __init__(self, longitudinal_data, cfs_type):
        self.longitudinal_data = longitudinal_data
        self.cfs_type = cfs_type
        self.clf = None

    def fit(self, X_train, y_train):
        if self.cfs_type == "Exh-CFS-Gr":
            clf = CorrelationBasedFeatureSelectionPerGroup(
                non_longitudinal_features=self.longitudinal_data.non_longitudinal_features(),
                features_group=self.longitudinal_data.feature_groups(),
                parallel=True,
                outer_search_method="greedySearch",
                version=1,
            )
        else:
            clf = CorrelationBasedFeatureSelectionPerGroup(
                non_longitudinal_features=None, features_group=None, parallel=False, outer_search_method=None
            )

        self.clf = clf.fit(X_train, y_train)
        return self

    def transform(self, X):
        return self.clf.apply_selected_features_and_rename(X, self.clf.selected_features_)

    def fit_transform(self, X_train, y_train):
        self.fit(X_train, y_train)
        return self.transform(X_train)


def update_longitudinal_data(dataset, preprocessor):
    preprocessor.longitudinal_data = dataset
    return preprocessor


if __name__ == "__main__":
    dataset_infos = [
        DatasetInfo("./data/elsa/core/csv/angina_dataset.csv", "class_angina_w8"),
        DatasetInfo("./data/elsa/core/csv/arthritis_dataset.csv", "class_arthritis_w8"),
        DatasetInfo("./data/elsa/core/csv/cataract_dataset.csv", "class_cataract_w8"),
        DatasetInfo("./data/elsa/core/csv/dementia_dataset.csv", "class_dementia_w8"),
        DatasetInfo("./data/elsa/core/csv/diabetes_dataset.csv", "class_diabetes_w8"),
        DatasetInfo("./data/elsa/core/csv/hbp_dataset.csv", "class_hbp_w8"),
        DatasetInfo("./data/elsa/core/csv/heartattack_dataset.csv", "class_heartattack_w8"),
        DatasetInfo("./data/elsa/core/csv/osteoporosis_dataset.csv", "class_osteoporosis_w8"),
        DatasetInfo("./data/elsa/core/csv/parkinsons_dataset.csv", "class_parkinsons_w8"),
        DatasetInfo("./data/elsa/core/csv/stroke_dataset.csv", "class_stroke_w8"),
    ]

    paper_results = {
        "Exh-CFS-Gr + NB": {
            "angina_dataset": 0.576,
            "arthritis_dataset": 0.629,
            "cataract_dataset": 0.658,
            "dementia_dataset": 0.603,
            "diabetes_dataset": 0.733,
            "hbp_dataset": 0.676,
            "heartattack_dataset": 0.619,
            "osteoporosis_dataset": 0.618,
            "parkinsons_dataset": 0.570,
            "stroke_dataset": 0.610,
        },
        "Standard CFS + NB": {
            "angina_dataset": 0.562,
            "arthritis_dataset": 0.625,
            "cataract_dataset": 0.677,
            "dementia_dataset": 0.589,
            "diabetes_dataset": 0.760,
            "hbp_dataset": 0.693,
            "heartattack_dataset": 0.620,
            "osteoporosis_dataset": 0.613,
            "parkinsons_dataset": 0.560,
            "stroke_dataset": 0.602,
        },
        "Exh-CFS-Gr + J48": {
            "angina_dataset": 0.550,
            "arthritis_dataset": 0.610,
            "cataract_dataset": 0.670,
            "dementia_dataset": 0.580,
            "diabetes_dataset": 0.750,
            "hbp_dataset": 0.670,
            "heartattack_dataset": 0.600,
            "osteoporosis_dataset": 0.620,
            "parkinsons_dataset": 0.580,
            "stroke_dataset": 0.600,
        },
        "Standard CFS + J48": {
            "angina_dataset": 0.540,
            "arthritis_dataset": 0.620,
            "cataract_dataset": 0.670,
            "dementia_dataset": 0.590,
            "diabetes_dataset": 0.760,
            "hbp_dataset": 0.660,
            "heartattack_dataset": 0.610,
            "osteoporosis_dataset": 0.610,
            "parkinsons_dataset": 0.580,
            "stroke_dataset": 0.590,
        },
    }

    nb_classifier = GaussianNB()
    dt_classifier = DecisionTreeClassifier()

    print("Exh-CFS-Gr + GaussianNB")
    engine_exh_cfs_gr_nb = ExperimentEngine(
        classifier=nb_classifier,
        paper_results=paper_results["Exh-CFS-Gr + NB"],
        preprocessor=CustomFeatureSelector(longitudinal_data=None, cfs_type="Exh-CFS-Gr"),
        debug=True,
    )
    engine_exh_cfs_gr_nb.create_datasets(dataset_infos)
    engine_exh_cfs_gr_nb.run_experiment(
        cv=10,
        preprocessor_pre_callback=update_longitudinal_data,
    )
    engine_exh_cfs_gr_nb.run_comparison(window=0.05)
    engine_exh_cfs_gr_nb.print_comparison()
    print("__" * 50)

    print("Standard CFS + GaussianNB")
    engine_standard_cfs_nb = ExperimentEngine(
        classifier=nb_classifier,
        paper_results=paper_results["Standard CFS + NB"],
        preprocessor=CustomFeatureSelector(longitudinal_data=None, cfs_type="Standard"),
        debug=True,
    )
    engine_standard_cfs_nb.create_datasets(dataset_infos)
    engine_standard_cfs_nb.run_experiment(
        cv=10,
        preprocessor_pre_callback=update_longitudinal_data,
    )
    engine_standard_cfs_nb.run_comparison(window=0.05)
    engine_standard_cfs_nb.print_comparison()
    print("__" * 50)

    print("Exh-CFS-Gr + J48")
    engine_exh_cfs_gr_dt = ExperimentEngine(
        classifier=dt_classifier,
        paper_results=paper_results["Exh-CFS-Gr + J48"],
        preprocessor=CustomFeatureSelector(longitudinal_data=None, cfs_type="Exh-CFS-Gr"),
        debug=True,
    )
    engine_exh_cfs_gr_dt.create_datasets(dataset_infos)
    engine_exh_cfs_gr_dt.run_experiment(
        cv=10,
        preprocessor_pre_callback=update_longitudinal_data,
    )
    engine_exh_cfs_gr_dt.run_comparison(window=0.05)
    engine_exh_cfs_gr_dt.print_comparison()
    print("__" * 50)

    print("Standard CFS + J48")
    engine_standard_cfs_dt = ExperimentEngine(
        classifier=dt_classifier,
        paper_results=paper_results["Standard CFS + J48"],
        preprocessor=CustomFeatureSelector(longitudinal_data=None, cfs_type="Standard"),
        debug=True,
    )
    engine_standard_cfs_dt.create_datasets(dataset_infos)
    engine_standard_cfs_dt.run_experiment(
        cv=10,
        preprocessor_pre_callback=update_longitudinal_data,
    )
    engine_standard_cfs_dt.run_comparison(window=0.05)
    engine_standard_cfs_dt.print_comparison()
    print("__" * 50)
