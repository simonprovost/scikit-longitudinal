import itertools
import os
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc
from scikit_longitudinal.data_preparation.merwav_time_minus import MerWavTimeMinus
from scikit_longitudinal.data_preparation.merwav_time_plus import MerWavTimePlus
from scikit_longitudinal.data_preparation.separate_waves import SepWav
from scikit_longitudinal.estimators.ensemble import (
    LexicoRandomForestClassifier,
    NestedTreesClassifier,
)
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
)
from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier
from scikit_longitudinal.pipeline import LongitudinalPipeline
from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import (
    CorrelationBasedFeatureSelection,
    CorrelationBasedFeatureSelectionPerGroup,
)


# SepWav(Voting) -> Standard CFS -> DecisionTreeClassifier
def SepWavStandardCFSDecisionTreeClassifierVoting(longitudinal_data):
    return [
        (
            "SepWav",
            SepWav(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
                voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
            ),
        ),
        (
            "CorrelationBasedFeatureSelection",
            CorrelationBasedFeatureSelection(),
        ),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4,
            ),
        ),
    ]


# SepWav(Stacking) -> Standard CFS -> RandomForest
def SepWavStandardCFSDecisionTreeClassifierStacking(longitudinal_data):
    return [
        (
            "SepWav",
            SepWav(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
                voting=LongitudinalEnsemblingStrategy.STACKING,
            ),
        ),
        (
            "CorrelationBasedFeatureSelection",
            CorrelationBasedFeatureSelection(),
        ),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4,
            ),
        ),
    ]


# SepWav(Voting) -> DecisionTreeClassifier
def SepWavDecisionTreeClassifierVoting(longitudinal_data):
    return [
        (
            "SepWav",
            SepWav(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
                voting=LongitudinalEnsemblingStrategy.DECAY_EXPONENTIAL_VOTING,
            ),
        ),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4,
            ),
        ),
    ]


# SepWav(Stacking) -> DecisionTreeClassifier
def SepWavDecisionTreeClassifierStacking(longitudinal_data):
    return [
        (
            "SepWav",
            SepWav(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
                voting=LongitudinalEnsemblingStrategy.STACKING,
                stacking_meta_learner=DecisionTreeClassifier(),
            ),
        ),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4,
            ),
        ),
    ]


# AggrFunc -> Standard CFS -> DecisionTree
def AggrFuncStandardCFSDecisionTreeClassifier(longitudinal_data):
    return [
        (
            "AggrFunc",
            AggrFunc(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
                aggregation_func="mean",
            ),
        ),
        (
            "CorrelationBasedFeatureSelection",
            CorrelationBasedFeatureSelection(),
        ),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4,
            ),
        ),
    ]


# AggrFunc -> DecisionTree
def AggrFuncDecisionTreeClassifier(longitudinal_data):
    return [
        (
            "AggrFunc",
            AggrFunc(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
                aggregation_func="mean",
            ),
        ),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4,
            ),
        ),
    ]


# MerWavTimeMinus -> Standard CFS -> DecisionTree
def MerWavTimeMinusStandardCFSDecisionTreeClassifier(longitudinal_data):
    return [
        (
            "MerWavTimeMinus",
            MerWavTimeMinus(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
            ),
        ),
        (
            "CorrelationBasedFeatureSelection",
            CorrelationBasedFeatureSelection(),
        ),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4,
            ),
        ),
    ]


# MerWavTimeMinus -> DT
def MerWavTimeMinusDecisionTreeClassifier(longitudinal_data):
    return [
        (
            "MerWavTimeMinus",
            MerWavTimeMinus(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
            ),
        ),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4,
            ),
        ),
    ]


# MerWavTimePlus -> Exhaustive CFS -> LexicoRF
def MerWavTimePlusExhaustiveCFSLexicoRFClassifier(longitudinal_data):
    return [
        (
            "MerWavTimePlus",
            MerWavTimePlus(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
            ),
        ),
        (
            "CorrelationBasedFeatureSelectionPerGroup",
            CorrelationBasedFeatureSelectionPerGroup(
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                features_group=longitudinal_data.feature_groups(),
                parallel=True,
                outer_search_method="greedySearch",
            ),
        ),
        (
            "LexicoRandomForestClassifier",
            LexicoRandomForestClassifier(
                n_estimators=5,
                features_group=longitudinal_data.feature_groups(),
                random_state=42,
            ),
        ),
    ]


# MerWavTimePlus -> LexicoRF
def MerWavTimePlusLexicoRFClassifier(longitudinal_data):
    return [
        (
            "MerWavTimePlus",
            MerWavTimePlus(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
            ),
        ),
        (
            "LexicoRandomForestClassifier",
            LexicoRandomForestClassifier(
                n_estimators=5,
                features_group=longitudinal_data.feature_groups(),
                random_state=42,
            ),
        ),
    ]


# MerWavTimePlus -> LexicoDecisionTree
def MerWavTimePlusLexicoDTClassifier(longitudinal_data):
    return [
        (
            "MerWavTimePlus",
            MerWavTimePlus(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
            ),
        ),
        (
            "LexicoDecisionTreeClassifier",
            LexicoDecisionTreeClassifier(
                threshold_gain=0.015,
                features_group=longitudinal_data.feature_groups(),
                random_state=42,
            ),
        ),
    ]


def MerWavTimePlusExhaustiveCFSNestedTree(longitudinal_data):
    return [
        (
            "MerWavTimePlus",
            MerWavTimePlus(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
            ),
        ),
        (
            "CorrelationBasedFeatureSelectionPerGroup",
            CorrelationBasedFeatureSelectionPerGroup(
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                features_group=longitudinal_data.feature_groups(),
                parallel=True,
                outer_search_method="greedySearch",
            ),
        ),
        (
            "NestedTreeClassifier",
            NestedTreesClassifier(
                features_group=longitudinal_data.feature_groups(),
                parallel=True,
                save_nested_trees=False,
            ),
        ),
    ]


# MerWavTimePlus -> NestedTree
def MerWavTimePlusNestedTree(longitudinal_data):
    return [
        (
            "MerWavTimePlus",
            MerWavTimePlus(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
            ),
        ),
        (
            "NestedTreeClassifier",
            NestedTreesClassifier(
                features_group=longitudinal_data.feature_groups(),
                parallel=True,
                save_nested_trees=False,
            ),
        ),
    ]


pipelines_dict = {
    "MerWavTimeMinusDecisionTreeClassifier": MerWavTimeMinusDecisionTreeClassifier,
    "AggrFuncDecisionTreeClassifier": AggrFuncDecisionTreeClassifier,
    "SepWavDecisionTreeClassifierVoting": SepWavDecisionTreeClassifierVoting,
    "SepWavDecisionTreeClassifierStacking": SepWavDecisionTreeClassifierStacking,
    "MerWavTimePlusLexicoRFClassifier": MerWavTimePlusLexicoRFClassifier,
    "MerWavTimePlusLexicoDTClassifier": MerWavTimePlusLexicoDTClassifier,
    "MerWavTimePlusNestedTree": MerWavTimePlusNestedTree,
    "MerWavTimeMinusStandardCFSDecisionTreeClassifier": MerWavTimeMinusStandardCFSDecisionTreeClassifier,
    "AggrFuncStandardCFSDecisionTreeClassifier": AggrFuncStandardCFSDecisionTreeClassifier,
    "SepWavStandardCFSDecisionTreeClassifierVoting": SepWavStandardCFSDecisionTreeClassifierVoting,
    "SepWavStandardCFSDecisionTreeClassifierStacking": SepWavStandardCFSDecisionTreeClassifierStacking,
    "MerWavTimePlusExhaustiveCFSNestedTree": MerWavTimePlusExhaustiveCFSNestedTree,
    "MerWavTimePlusExhaustiveCFSLexicoRFClassifier": MerWavTimePlusExhaustiveCFSLexicoRFClassifier,
}

non_ray_multiclass_pipelines_dict = {
    "MerWavTimeMinusDecisionTreeClassifier": MerWavTimeMinusDecisionTreeClassifier,
    "AggrFuncDecisionTreeClassifier": AggrFuncDecisionTreeClassifier,
    "SepWavDecisionTreeClassifierVoting": SepWavDecisionTreeClassifierVoting,
    "SepWavDecisionTreeClassifierStacking": SepWavDecisionTreeClassifierStacking,
    "MerWavTimePlusLexicoRFClassifier": MerWavTimePlusLexicoRFClassifier,
    "MerWavTimePlusLexicoDTClassifier": MerWavTimePlusLexicoDTClassifier,
    "MerWavTimeMinusStandardCFSDecisionTreeClassifier": MerWavTimeMinusStandardCFSDecisionTreeClassifier,
    "AggrFuncStandardCFSDecisionTreeClassifier": AggrFuncStandardCFSDecisionTreeClassifier,
    "SepWavStandardCFSDecisionTreeClassifierVoting": SepWavStandardCFSDecisionTreeClassifierVoting,
    "SepWavStandardCFSDecisionTreeClassifierStacking": SepWavStandardCFSDecisionTreeClassifierStacking,
}


class SyntheticMulticlassLongitudinalData:
    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_groups,
        non_longitudinal_features,
        feature_names,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self._feature_groups = feature_groups
        self._non_longitudinal_features = non_longitudinal_features
        self.data = pd.DataFrame(
            np.vstack([X_train, X_test]),
            columns=feature_names,
        )

    def feature_groups(self):
        return self._feature_groups

    def non_longitudinal_features(self):
        return self._non_longitudinal_features


class TestLongitudinalPipelines:
    @pytest.fixture
    def synthetic_multiclass_longitudinal_data(self):
        X, y = make_classification(
            n_samples=180,
            n_features=8,
            n_informative=8,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=1.75,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42
        )
        return SyntheticMulticlassLongitudinalData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_groups=[[0, 1], [2, 3], [4, 5]],
            non_longitudinal_features=[6, 7],
            feature_names=[
                "feat_a_w1",
                "feat_a_w2",
                "feat_b_w1",
                "feat_b_w2",
                "feat_c_w1",
                "feat_c_w2",
                "age",
                "bmi",
            ],
        )

    @pytest.fixture(
        params=list(
            itertools.product(
                ["core", "nurse"],
                [
                    "angina_dataset.csv",
                    # The following datasets are commented out because they take too long to run
                    # Feel free to uncomment them if you want to run them
                    # "arthritis_dataset.csv",
                    # "cataract_dataset.csv",
                    # "dementia_dataset.csv",
                    # "hbp_dataset.csv",
                    # "diabetes_dataset.csv",
                    # "osteoporosis_dataset.csv",
                    # "heartattack_dataset.csv",
                    # "parkinsons_dataset.csv",
                    # "stroke_dataset.csv",
                ],
            )
        )
    )
    def longitudinal_data(self, request):
        folder_name, dataset_name = request.param
        file_path = f"./data/elsa/{folder_name}/csv/{dataset_name}"
        if not os.path.exists(file_path):
            pytest.skip("CSV file not available - skipping test")
        longitudinal_data = LongitudinalDataset(file_path=file_path)
        longitudinal_data.load_data_target_train_test_split(
            target_column=f"class_{dataset_name.split('_')[0]}_w8",
            remove_target_waves=True,
            random_state=42,
        )
        longitudinal_data.setup_features_group(input_data="elsa")
        return longitudinal_data

    @pytest.mark.parametrize("pipeline_name, pipeline_function", pipelines_dict.items())
    def test_pipeline(self, longitudinal_data, pipeline_name, pipeline_function):
        longitudinal_pipeline = LongitudinalPipeline(
            steps=pipeline_function(longitudinal_data),
            features_group=longitudinal_data.feature_groups(),
            non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
            update_feature_groups_callback="default",
            feature_list_names=longitudinal_data.data.columns.tolist(),
        )
        longitudinal_pipeline.fit(
            longitudinal_data.X_train,
            longitudinal_data.y_train,
        )
        assert (
            len(longitudinal_pipeline.steps) > 0
        ), f"Pipeline {pipeline_name} does not contain any steps"

        y_pred = longitudinal_pipeline.predict(longitudinal_data.X_test)

        print(
            f"Classification report for {longitudinal_data.file_path} using {pipeline_name}"
        )
        print(classification_report(longitudinal_data.y_test, y_pred))

        assert y_pred is not None, "Predict method failed"
        assert len(y_pred) == len(
            longitudinal_data.y_test
        ), f"Size mismatch between predicted and actual values for {pipeline_name}"
        unique_y = longitudinal_data.y_train.unique()
        assert set(y_pred).issubset(
            unique_y
        ), f"Invalid class labels in predictions for {pipeline_name}"

        y_pred_proba = longitudinal_pipeline.predict_proba(longitudinal_data.X_test)
        assert y_pred_proba is not None, "Predict_proba method failed"
        assert len(y_pred_proba) == len(
            longitudinal_data.y_test
        ), f"Size mismatch between predicted and actual values for {pipeline_name}"
        assert y_pred_proba.shape[1] == len(
            unique_y
        ), f"Invalid number of classes in predictions for {pipeline_name}"
        assert np.allclose(
            np.sum(y_pred_proba, axis=1),
            np.ones(len(y_pred_proba)),
            atol=1e-5,
        ), f"Probabilities do not sum to 1 for {pipeline_name}"

    @pytest.mark.parametrize("pipeline_name, pipeline_function", pipelines_dict.items())
    def test_pipeline_without_default_callback(
        self, longitudinal_data, pipeline_name, pipeline_function
    ):
        def dummy_callable_wrong_parameters(
            step_idx, dummy_longitudinal_dataset, y, name
        ):
            return (
                dummy_longitudinal_dataset.data.to_numpy(),
                dummy_longitudinal_dataset.feature_groups(),
                dummy_longitudinal_dataset.non_longitudinal_features(),
                dummy_longitudinal_dataset.data.columns.tolist(),
            )

        with pytest.raises(
            ValueError,
            match="update_data_callback must be a callable function or a 'default' string value.",
        ):
            longitudinal_pipeline = LongitudinalPipeline(
                steps=pipeline_function(longitudinal_data),
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                update_feature_groups_callback="DUMMY_CALLBACK",
                feature_list_names=longitudinal_data.data.columns.tolist(),
            )

            longitudinal_pipeline.fit(
                longitudinal_data.X_train,
                longitudinal_data.y_train,
            )

        with pytest.raises(
            ValueError, match="update_data_callback must accept 5 parameters, got 4."
        ):
            longitudinal_pipeline = LongitudinalPipeline(
                steps=pipeline_function(longitudinal_data),
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                update_feature_groups_callback=dummy_callable_wrong_parameters,
                feature_list_names=longitudinal_data.data.columns.tolist(),
            )
            longitudinal_pipeline.fit(
                longitudinal_data.X_train,
                longitudinal_data.y_train,
            )

    @pytest.mark.parametrize(
        "pipeline_name, pipeline_function", non_ray_multiclass_pipelines_dict.items()
    )
    def test_non_ray_multiclass_pipeline_recipes(
        self,
        synthetic_multiclass_longitudinal_data,
        pipeline_name,
        pipeline_function,
    ):
        longitudinal_data = synthetic_multiclass_longitudinal_data
        longitudinal_pipeline = LongitudinalPipeline(
            steps=pipeline_function(longitudinal_data),
            features_group=longitudinal_data.feature_groups(),
            non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
            update_feature_groups_callback="default",
            feature_list_names=longitudinal_data.data.columns.tolist(),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The provided callable <function mean .*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but .* was fitted with feature names",
                category=UserWarning,
            )
            longitudinal_pipeline.fit(
                longitudinal_data.X_train,
                longitudinal_data.y_train,
            )
            y_pred = longitudinal_pipeline.predict(longitudinal_data.X_test)
            y_pred_proba = longitudinal_pipeline.predict_proba(longitudinal_data.X_test)

        unique_y = np.unique(longitudinal_data.y_train)

        assert (
            len(longitudinal_pipeline.steps) > 0
        ), f"Pipeline {pipeline_name} does not contain any steps"
        assert y_pred is not None, f"Predict method failed for {pipeline_name}"
        assert y_pred.shape == (len(longitudinal_data.y_test),)
        assert set(y_pred).issubset(
            unique_y
        ), f"Invalid class labels in predictions for {pipeline_name}"
        assert (
            y_pred_proba is not None
        ), f"Predict_proba method failed for {pipeline_name}"
        assert y_pred_proba.shape == (
            len(longitudinal_data.y_test),
            len(unique_y),
        ), f"Invalid number of classes in predictions for {pipeline_name}"
        assert np.allclose(
            np.sum(y_pred_proba, axis=1),
            np.ones(len(y_pred_proba)),
            atol=1e-5,
        ), f"Probabilities do not sum to 1 for {pipeline_name}"
