from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

from scikit_longitudinal.data_preparation import SepWav
from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import (
    CorrelationBasedFeatureSelectionPerGroup,
)


class SpecialHandlerInterface(ABC):
    """Abstract base class for special handlers.

    Provides methods to handle special cases during both the transformation phase and the final estimator phase of a
    machine learning pipeline_managers.

    """

    @abstractmethod
    def handle_transform(
        self, transformer: Any, X: Any, y: Optional[Any] = None, **kwargs
    ) -> Tuple[Any, Any, Optional[Any]]:
        """Handle special cases during the transform phase.

        Args:
            transformer:
                The transformer being processed.
            X:
                Input data.
            y:
                Target data, optional.
            kwargs:
                Additional arguments.

        Returns:
            Tuple containing the potentially modified transformer, X, and y.

        """

    @abstractmethod
    def handle_final_estimator(
        self, final_estimator: Any, steps: List[Tuple[str, Any]], X: Any, y: Any, **kwargs
    ) -> Tuple[Any, List[Tuple[str, Any]], Any, Any]:
        """Handle special cases for the final estimator.

        Args:
            final_estimator:
                The final estimator in the pipeline.
            steps:
                The steps in the pipeline.
            X:
                Input data.
            y:
                Target data.
            kwargs:
                Additional arguments.

        Returns:
            Tuple containing the potentially modified final_estimator, steps, X, and y.

        """


class CorrelationBasedFeatureSelectionPerGroupHandler(SpecialHandlerInterface):
    """
    Special handler for the CorrelationBasedFeatureSelectionPerGroup transformer.
    """

    def handle_transform(
        self, transformer: Any, X: Any, y: Optional[Any] = None, **kwargs
    ) -> Tuple[Any, Any, Optional[Any]]:
        """Handle special cases during the transform phase for CorrelationBasedFeatureSelectionPerGroup.

        Modifies the input data X based on the transformer's selected features.

        """
        if max(transformer.selected_features_) < X.shape[1]:
            X = X[:, transformer.selected_features_]
        else:
            raise ValueError("Indices in transformer.selected_features_ exceed the number of columns in X.")
        return transformer, X, y

    def handle_final_estimator(
        self, final_estimator: Any, steps: List[Tuple[str, Any]], X: Any, y: Any, **kwargs
    ) -> Tuple[Any, List[Tuple[str, Any]], Any, Any]:
        """
        No special handling for the final estimator for this transformer.
        """
        return final_estimator, steps, X, y


class SepWavHandler(SpecialHandlerInterface):
    """
    Special handler for the SepWav transformer.
    """

    def handle_transform(
        self, transformer: Any, X: Any, y: Optional[Any] = None, **kwargs
    ) -> Tuple[Any, Any, Optional[Any]]:
        """
        No special handling during the transform phase for this transformer.
        """
        return transformer, X, y

    def handle_final_estimator(
        self, final_estimator: Any, steps: List[Tuple[str, Any]], X: Any, y: Any, **kwargs
    ) -> Tuple[Any, List[Tuple[str, Any]], Any, Any]:
        """Handle special cases for the SepWav transformer during the final estimator phase.

        Modifies the classifier attribute of the SepWav object in the pipeline's steps. While the SepWav is considered a
        transformer, it is also a classifier, and therefore needs to be handled differently than other transformers.
        Such that the SepWav become the actual final estimator in the pipeline. Refer to the documentation of SepWav for
        more information.

        """
        # TODO: This handler does not work if we have more than one preprocessor # pylint: disable=W0511
        #  / transformer in the  # pylint: disable=W0511
        #  pipeline_managers \ Find a way to make it work for any number of preprocessor # pylint: disable=W0511
        #  / transformers. # pylint: disable=W0511

        if len(steps) > 1 and isinstance(steps[-2][1], SepWav):
            if hasattr(steps[-2][1], "estimator"):
                steps[-2][1].estimator = final_estimator

                step_index = next((i for i, step in enumerate(steps) if step[0] == steps[-2][0]), None)

                if step_index is None:
                    raise ValueError(f"No step with name '{steps[-2][0]}' exists in the pipeline.")

                steps.append(steps.pop(step_index))
                final_estimator = steps[-1][1]
            else:
                raise ValueError("SepWav does not have an estimator attribute.")
        return final_estimator, steps, X, y


SPECIAL_HANDLERS: Dict[Type[Any], SpecialHandlerInterface] = {
    CorrelationBasedFeatureSelectionPerGroup: CorrelationBasedFeatureSelectionPerGroupHandler(),
    SepWav: SepWavHandler(),
}
