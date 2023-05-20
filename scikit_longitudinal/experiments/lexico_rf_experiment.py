from sklearn.ensemble import RandomForestClassifier

from scikit_longitudinal.estimators.tree import LexcioRFClassifier
from scikit_longitudinal.experiments.engine import DatasetInfo, ExperimentEngine


def update_features_group(dataset, processor):
    processor.features_group = dataset.feature_groups()
    return processor


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
        "Random Forest": {
            "angina_dataset": 0.711,
            "arthritis_dataset": 0.749,
            "cataract_dataset": 0.609,
            "dementia_dataset": 0.764,
            "diabetes_dataset": 0.674,
            "hbp_dataset": 0.641,
            "heartattack_dataset": 0.678,
            "osteoporosis_dataset": 0.7,
            "parkinsons_dataset": 0.697,
            "stroke_dataset": 0.694,
        },
        "Lexico RF": {
            "angina_dataset": 0.709,
            "arthritis_dataset": 0.752,
            "cataract_dataset": 0.625,
            "dementia_dataset": 0.768,
            "diabetes_dataset": 0.672,
            "hbp_dataset": 0.633,
            "heartattack_dataset": 0.683,
            "osteoporosis_dataset": 0.701,
            "parkinsons_dataset": 0.701,
            "stroke_dataset": 0.697,
        },
    }

    dt_classifier = RandomForestClassifier()
    nt_classifier = LexcioRFClassifier(
        features_group=[[0, 1], [2, 3]],
    )

    print("Random Forest")
    engine_dt = ExperimentEngine(
        classifier=dt_classifier,
        paper_results=paper_results["Random Forest"],
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

    print("Lexico RF")
    engine_nt = ExperimentEngine(
        classifier=nt_classifier,
        paper_results=paper_results["Lexico RF"],
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
