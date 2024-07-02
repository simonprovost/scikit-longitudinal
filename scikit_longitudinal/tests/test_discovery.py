import pytest
from sklearn.base import BaseEstimator

from scikit_longitudinal.discovery import all_scikit_longitudinal_estimators
from scikit_longitudinal.templates import (
    CustomClassifierMixinEstimator,
    CustomTransformerMixinEstimator,
    DataPreparationMixin,
)


class TestAllScikitLongitudinalEstimators:

    def test_no_filter(self):
        estimators = all_scikit_longitudinal_estimators()
        assert isinstance(estimators, list)
        assert all(isinstance(est, tuple) for est in estimators)
        assert all(
            issubclass(
                est[1],
                (BaseEstimator, CustomTransformerMixinEstimator, CustomClassifierMixinEstimator, DataPreparationMixin),
            )
            for est in estimators
        )

    def test_classifier_filter(self):
        estimators = all_scikit_longitudinal_estimators(type_filter="classifier")
        assert isinstance(estimators, list)
        assert all(isinstance(est, tuple) for est in estimators)
        assert all(
            issubclass(est[1], CustomClassifierMixinEstimator)
            or est[0]
            in [
                "LexicoGradientBoostingClassifier",
                "LexicoRandomForestClassifier",
                "LexicoDecisionTreeClassifier",
                "LexicoDecisionTreeRegressor",
                "LexicoDeepForestClassifier",
            ]
            for est in estimators
        )

    def test_transformer_filter(self):
        estimators = all_scikit_longitudinal_estimators(type_filter="transformer")
        assert isinstance(estimators, list)
        assert all(isinstance(est, tuple) for est in estimators)
        assert all(issubclass(est[1], CustomTransformerMixinEstimator) for est in estimators)

    def test_data_preparation_filter(self):
        estimators = all_scikit_longitudinal_estimators(type_filter="data_preparation")
        assert isinstance(estimators, list)
        assert all(isinstance(est, tuple) for est in estimators)
        assert all(issubclass(est[1], DataPreparationMixin) for est in estimators)

    def test_invalid_filter(self):
        with pytest.raises(
            ValueError,
            match="Parameter type_filter must be 'classifier', 'transformer', 'data_preparation' or None, got",
        ):
            all_scikit_longitudinal_estimators(type_filter="invalid_filter")

    def test_abstract_classes(self):
        estimators = all_scikit_longitudinal_estimators()
        abstract_classes = [
            est for est in estimators if hasattr(est[1], "__abstractmethods__") and est[1].__abstractmethods__
        ]
        assert len(abstract_classes) == 0

    def test_duplicates(self):
        estimators = all_scikit_longitudinal_estimators()
        names = [est[0] for est in estimators]
        assert len(names) == len(set(names)), "There are duplicate estimators in the list"
