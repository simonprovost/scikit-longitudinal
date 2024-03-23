from typing import List

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier as KNNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from experiments.pre_analysis_scikit_learn_longitudinal_algorithms.empirical_scikit_algorithms_evaluation import \
    EmpiricalEvaluation


def empirical_evaluation_template(
        path: str,
        target_column: str,
        estimators: List[BaseEstimator],
        k_outer_folds: int,
        k_inner_folds: int,
        n_jobs: int
):
    evaluation = EmpiricalEvaluation(
        estimators=estimators,
        dataset_path=path,
        target_column=target_column,
        k_outer_folds=k_outer_folds,
        k_inner_folds=k_inner_folds,
        n_jobs=n_jobs,
        export_name="SCIKIT_LEARN_ALGORITHMS_5_5"
    )
    evaluation.start_empirical_evaluation()
    evaluation.report_empirical_evaluation()


if __name__ == "__main__":
    """ Run the empirical evaluation for the fifth scikit-learn setting.
    
    To run this script, you need to have access to the
    ELSA datasets and update the path's variable with the correct paths to the dataset. Therefore,
    you can query the datasets directly to the author of this repository via opening a Github issue.
    
    """
    path = "./data/elsa/nurse/csv/hbp_dataset.csv"
    target_column = "class_hbp_w8"

    estimators = [
        DecisionTreeClassifier(
            criterion="entropy"
        ),
        RandomForestClassifier(
            criterion="entropy"
        ),
        GradientBoostingClassifier(),
        ExtraTreesClassifier(),
        LinearSVC(),
        KNNeighborsClassifier(),
    ]

    empirical_evaluation_template(
        path=path,
        target_column=target_column,
        estimators=estimators,
        k_outer_folds=5,
        k_inner_folds=5,
        n_jobs=6
    )
