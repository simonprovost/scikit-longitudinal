from sklearn.tree import DecisionTreeClassifier

from scikit_longitudinal.estimators.tree import NestedTreesClassifier
from scikit_longitudinal.experiments.engine import DatasetInfo, ExperimentEngine


def update_features_group(dataset, processor):
    processor.features_group = dataset.feature_groups()
    return processor


if __name__ == "__main__":
    dataset_infos = [
        DatasetInfo("./data/elsa/nurse/csv/angina_dataset.csv", "class_angina_w8"),
        DatasetInfo("./data/elsa/nurse/csv/arthritis_dataset.csv", "class_arthritis_w8"),
        DatasetInfo("./data/elsa/nurse/csv/cataract_dataset.csv", "class_cataract_w8"),
        DatasetInfo("./data/elsa/nurse/csv/dementia_dataset.csv", "class_dementia_w8"),
        DatasetInfo("./data/elsa/nurse/csv/diabetes_dataset.csv", "class_diabetes_w8"),
        DatasetInfo("./data/elsa/nurse/csv/hbp_dataset.csv", "class_hbp_w8"),
        DatasetInfo("./data/elsa/nurse/csv/heartattack_dataset.csv", "class_heartattack_w8"),
        DatasetInfo("./data/elsa/nurse/csv/osteoporosis_dataset.csv", "class_osteoporosis_w8"),
        DatasetInfo("./data/elsa/nurse/csv/parkinsons_dataset.csv", "class_parkinsons_w8"),
        DatasetInfo("./data/elsa/nurse/csv/stroke_dataset.csv", "class_stroke_w8"),
    ]

    paper_results = {
        "Decision Tree": {
            "angina_dataset": 0.455,
            "arthritis_dataset": 0.560,
            "cataract_dataset": 0.620,
            "dementia_dataset": 0.420,
            "diabetes_dataset": 0.365,
            "hbp_dataset": 0.555,
            "heartattack_dataset": 0.382,
            "osteoporosis_dataset": 0.403,
            "parkinsons_dataset": 0.345,
            "stroke_dataset": 0.280,
        },
        "Nested Tree": {
            "angina_dataset": 0.515,
            "arthritis_dataset": 0.548,
            "cataract_dataset": 0.575,
            "dementia_dataset": 0.532,
            "diabetes_dataset": 0.584,
            "hbp_dataset": 0.602,
            "heartattack_dataset": 0.513,
            "osteoporosis_dataset": 0.541,
            "parkinsons_dataset": 0.500,
            "stroke_dataset": 0.527,
        },
    }

    dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    nt_classifier = NestedTreesClassifier(
        features_group=[[0, 1], [2, 3]],
        max_outer_depth=10,
        max_inner_depth=5,
        min_outer_samples=2,
        inner_estimator_hyperparameters={"min_samples_split": 2},
        parallel=True,
    )

    print("Decision Tree")
    engine_dt = ExperimentEngine(
        classifier=dt_classifier,
        paper_results=paper_results["Decision Tree"],
        preprocessor=None,
        debug=True,
    )
    engine_dt.create_datasets(dataset_infos)
    engine_dt.run_experiment(
        cv=10,
    )
    engine_dt.run_comparison(window=0.05)
    engine_dt.print_comparison()
    print("__" * 50)

    print("Nested Tree")
    engine_nt = ExperimentEngine(
        classifier=nt_classifier,
        paper_results=paper_results["Nested Tree"],
        preprocessor=None,
        debug=True,
    )
    engine_nt.create_datasets(dataset_infos)
    engine_nt.run_experiment(
        cv=10,
        classifier_pre_callback=update_features_group,
    )
    engine_nt.run_comparison(window=0.05)
    engine_nt.print_comparison()
    print("__" * 50)
