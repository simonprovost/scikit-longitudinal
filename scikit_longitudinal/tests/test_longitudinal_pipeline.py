import itertools
import os
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc
from scikit_longitudinal.data_preparation.merwav_time_minus import MerWavTimeMinus
from scikit_longitudinal.data_preparation.merwav_time_plus import MerWavTimePlus
from scikit_longitudinal.data_preparation.separate_waves import SepWav
from scikit_longitudinal.estimators.tree import LexicoRFClassifier, LexicoDecisionTreeClassifier, NestedTreesClassifier
from scikit_longitudinal.pipeline import LongitudinalPipeline
from scikit_longitudinal.preprocessing.feature_selection.cfs_per_group import CorrelationBasedFeatureSelectionPerGroup

import pytest

# SepWav(Voting) -> Standard CFS -> DecisionTreeClassifier
def SepWavStandardCFSDecisionTreeClassifierVoting(longitudinal_data):
    return [
        (
            "SepWav",
            SepWav(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
                ensemble_strategy="voting",
            ),
        ),
        (
            "CorrelationBasedFeatureSelection",
            CorrelationBasedFeatureSelectionPerGroup(
                cfs_type="cfs",
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


# SepWav(Stacking) -> Standard CFS -> RandomForest
def SepWavStandardCFSDecisionTreeClassifierStacking(longitudinal_data):
    return [
        (
            "SepWav",
            SepWav(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
                ensemble_strategy="stacking",
            ),
        ),
        (
            "CorrelationBasedFeatureSelection",
            CorrelationBasedFeatureSelectionPerGroup(
                cfs_type="cfs",
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


# SepWav(Voting) -> DecisionTreeClassifier
def SepWavDecisionTreeClassifierVoting(longitudinal_data):
    return [
        (
            "SepWav",
            SepWav(
                features_group=longitudinal_data.feature_groups(),
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                feature_list_names=longitudinal_data.data.columns.tolist(),
                ensemble_strategy="voting",
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
                ensemble_strategy="stacking",
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
            CorrelationBasedFeatureSelectionPerGroup(
                cfs_type="cfs",
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
            CorrelationBasedFeatureSelectionPerGroup(
                cfs_type="cfs",
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
                features_group=longitudinal_data.feature_groups(),
                cfs_longitudinal_outer_search_method="greedySearch",
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                parallel=True,
            ),
        ),
        (
            "LexicoRFClassifier",
            LexicoRFClassifier(
                n_estimators=5,
                features_group=longitudinal_data.feature_groups(),
                random_state=42,
            )
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
            "LexicoRFClassifier",
            LexicoRFClassifier(
                n_estimators=5,
                features_group=longitudinal_data.feature_groups(),
                random_state=42,
            )
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
            )
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
                features_group=longitudinal_data.feature_groups(),
                cfs_longitudinal_outer_search_method="greedySearch",
                non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
                parallel=True,
            ),
        ),
        (
            "NestedTreeClassifier",
            NestedTreesClassifier(
                features_group=longitudinal_data.feature_groups(),
                parallel=True,
                save_nested_trees=False,
            )
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
            )
        ),
    ]

pipelines_dict = {
    'MerWavTimeMinusDecisionTreeClassifier': MerWavTimeMinusDecisionTreeClassifier,
    'AggrFuncDecisionTreeClassifier': AggrFuncDecisionTreeClassifier,
    'SepWavDecisionTreeClassifierVoting': SepWavDecisionTreeClassifierVoting,
    'SepWavDecisionTreeClassifierStacking': SepWavDecisionTreeClassifierStacking,
    'MerWavTimePlusLexicoRFClassifier': MerWavTimePlusLexicoRFClassifier,
    'MerWavTimePlusLexicoDTClassifier': MerWavTimePlusLexicoDTClassifier,
    'MerWavTimePlusNestedTree': MerWavTimePlusNestedTree,

    'MerWavTimeMinusStandardCFSDecisionTreeClassifier': MerWavTimeMinusStandardCFSDecisionTreeClassifier,
    'AggrFuncStandardCFSDecisionTreeClassifier': AggrFuncStandardCFSDecisionTreeClassifier,
    'SepWavStandardCFSDecisionTreeClassifierVoting': SepWavStandardCFSDecisionTreeClassifierVoting,
    'SepWavStandardCFSDecisionTreeClassifierStacking': SepWavStandardCFSDecisionTreeClassifierStacking,

    'MerWavTimePlusExhaustiveCFSNestedTree': MerWavTimePlusExhaustiveCFSNestedTree,
    'MerWavTimePlusExhaustiveCFSLexicoRFClassifier': MerWavTimePlusExhaustiveCFSLexicoRFClassifier,
}


class TestLongitudinalPipelines:
    @pytest.fixture(
        params=list
            (itertools.product(
            [
                "core",
                "nurse"
            ],
            [
                "angina_dataset.csv",
                "arthritis_dataset.csv",
                "cataract_dataset.csv",
                "dementia_dataset.csv",
                "hbp_dataset.csv",
                "diabetes_dataset.csv",
                "osteoporosis_dataset.csv",
                "heartattack_dataset.csv",
                "parkinsons_dataset.csv",
                "stroke_dataset.csv"
            ]
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
            longitudinal_data.X_train, longitudinal_data.y_train,
        )
        assert len(longitudinal_pipeline.steps) > 0, f"Pipeline {pipeline_name} does not contain any steps"

        y_pred = longitudinal_pipeline.predict(longitudinal_data.X_test)

        print(f"Classification report for {longitudinal_data.file_path} using {pipeline_name}")
        print(classification_report(longitudinal_data.y_test, y_pred))

        assert y_pred is not None, "Predict method failed"
        assert len(y_pred) == len(
            longitudinal_data.y_test), f"Size mismatch between predicted and actual values for {pipeline_name}"
        unique_y = longitudinal_data.y_train.unique()
        assert set(y_pred).issubset(unique_y), f"Invalid class labels in predictions for {pipeline_name}"
