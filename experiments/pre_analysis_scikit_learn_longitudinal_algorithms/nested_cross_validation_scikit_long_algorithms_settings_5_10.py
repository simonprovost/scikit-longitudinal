from typing import List

from deepforest import CascadeForestClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from experiments.pre_analysis_scikit_learn_longitudinal_algorithms.empirical_scikit_algorithms_evaluation import \
    EmpiricalEvaluation
from scikit_longitudinal.estimators.ensemble import LexicoDeepForestClassifier
from scikit_longitudinal.estimators.ensemble import NestedTreesClassifier
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest import LongitudinalEstimatorConfig, \
    LongitudinalClassifierType
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import \
    LexicoGradientBoostingClassifier
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_random_forest import LexicoRandomForestClassifier
from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier
from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import \
    CorrelationBasedFeatureSelection, CorrelationBasedFeatureSelectionPerGroup


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
        export_name="SCIKIT_LONG_ALGORITHMS_5_10"
    )
    evaluation.start_empirical_evaluation()
    evaluation.report_empirical_evaluation()


if __name__ == "__main__":
    """ Run the empirical evaluation for the first scikit-learn setting.
    
    To run this script, you need to have access to the
    ELSA datasets and update the path's variable with the correct paths to the dataset. Therefore,
    you can query the datasets directly to the author of this repository via opening a Github issue.
    
    """

    path = "./data/elsa/nurse/csv/hbp_dataset.csv"
    target_column = "class_hbp_w8"

    lexico_rf_config = LongitudinalEstimatorConfig(
        classifier_type=LongitudinalClassifierType.LEXICO_RF,
        count=2,
    )

    standard_deep_forest_five_max_layers = CascadeForestClassifier(
        max_layers=5
    )

    standard_deep_forest_four_max_layers = CascadeForestClassifier(
        max_layers=4
    )

    estimators = [
        RandomForestClassifier()
        for _ in range(2)
    ]
    estimators.extend(
        RandomForestClassifier(max_features=1)
        for _ in range(2)
    )

    standard_deep_forest_five_max_layers.set_estimator(estimators, n_splits=2)
    standard_deep_forest_four_max_layers.set_estimator(estimators, n_splits=2)

    estimators = [
        LexicoDecisionTreeClassifier(),
        LexicoRandomForestClassifier(),
        LexicoGradientBoostingClassifier(),
        CorrelationBasedFeatureSelection(),
        CorrelationBasedFeatureSelectionPerGroup(),
        CorrelationBasedFeatureSelectionPerGroup(version=2),
        NestedTreesClassifier(),
        LexicoDeepForestClassifier(
            longitudinal_base_estimators=[lexico_rf_config],
            max_layers=5
        ),
        LexicoDeepForestClassifier(
            longitudinal_base_estimators=[lexico_rf_config],
            max_layers=4
        ),
        standard_deep_forest_five_max_layers,
        standard_deep_forest_four_max_layers,
    ]

    empirical_evaluation_template(
        path=path,
        target_column=target_column,
        estimators=estimators,
        k_outer_folds=5,
        k_inner_folds=10,
        n_jobs=11
    )
