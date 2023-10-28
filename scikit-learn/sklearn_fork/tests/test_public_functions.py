from importlib import import_module
from inspect import signature
from numbers import Integral, Real

import pytest

from sklearn_fork.utils._param_validation import generate_invalid_param_val
from sklearn_fork.utils._param_validation import generate_valid_param
from sklearn_fork.utils._param_validation import make_constraint
from sklearn_fork.utils._param_validation import InvalidParameterError
from sklearn_fork.utils._param_validation import Interval


def _get_func_info(func_module):
    module_name, func_name = func_module.rsplit(".", 1)
    module = import_module(module_name)
    func = getattr(module, func_name)

    func_sig = signature(func)
    func_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    # The parameters `*args` and `**kwargs` are ignored since we cannot generate
    # constraints.
    required_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    return func, func_name, func_params, required_params


def _check_function_param_validation(
    func, func_name, func_params, required_params, parameter_constraints
):
    """Check that an informative error is raised when the value of a parameter does not
    have an appropriate type or value.
    """
    # generate valid values for the required parameters
    valid_required_params = {}
    for param_name in required_params:
        if parameter_constraints[param_name] == "no_validation":
            valid_required_params[param_name] = 1
        else:
            valid_required_params[param_name] = generate_valid_param(
                make_constraint(parameter_constraints[param_name][0])
            )

    # check that there is a constraint for each parameter
    if func_params:
        validation_params = parameter_constraints.keys()
        unexpected_params = set(validation_params) - set(func_params)
        missing_params = set(func_params) - set(validation_params)
        err_msg = (
            "Mismatch between _parameter_constraints and the parameters of"
            f" {func_name}.\nConsider the unexpected parameters {unexpected_params} and"
            f" expected but missing parameters {missing_params}\n"
        )
        assert set(validation_params) == set(func_params), err_msg

    # this object does not have a valid type for sure for all params
    param_with_bad_type = type("BadType", (), {})()

    for param_name in func_params:
        constraints = parameter_constraints[param_name]

        if constraints == "no_validation":
            # This parameter is not validated
            continue

        # Mixing an interval of reals and an interval of integers must be avoided.
        if any(
            isinstance(constraint, Interval) and constraint.type == Integral
            for constraint in constraints
        ) and any(
            isinstance(constraint, Interval) and constraint.type == Real
            for constraint in constraints
        ):
            raise ValueError(
                f"The constraint for parameter {param_name} of {func_name} can't have a"
                " mix of intervals of Integral and Real types. Use the type"
                " RealNotInt instead of Real."
            )

        match = (
            rf"The '{param_name}' parameter of {func_name} must be .* Got .* instead."
        )

        # First, check that the error is raised if param doesn't match any valid type.
        with pytest.raises(InvalidParameterError, match=match):
            func(**{**valid_required_params, param_name: param_with_bad_type})

        # Then, for constraints that are more than a type constraint, check that the
        # error is raised if param does match a valid type but does not match any valid
        # value for this type.
        constraints = [make_constraint(constraint) for constraint in constraints]

        for constraint in constraints:
            try:
                bad_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue

            with pytest.raises(InvalidParameterError, match=match):
                func(**{**valid_required_params, param_name: bad_value})


PARAM_VALIDATION_FUNCTION_LIST = [
    "sklearn_fork.calibration.calibration_curve",
    "sklearn_fork.cluster.cluster_optics_dbscan",
    "sklearn_fork.cluster.compute_optics_graph",
    "sklearn_fork.cluster.estimate_bandwidth",
    "sklearn_fork.cluster.kmeans_plusplus",
    "sklearn_fork.cluster.cluster_optics_xi",
    "sklearn_fork.cluster.ward_tree",
    "sklearn_fork.covariance.empirical_covariance",
    "sklearn_fork.covariance.ledoit_wolf_shrinkage",
    "sklearn_fork.covariance.shrunk_covariance",
    "sklearn_fork.datasets.clear_data_home",
    "sklearn_fork.datasets.dump_svmlight_file",
    "sklearn_fork.datasets.fetch_20newsgroups",
    "sklearn_fork.datasets.fetch_20newsgroups_vectorized",
    "sklearn_fork.datasets.fetch_california_housing",
    "sklearn_fork.datasets.fetch_covtype",
    "sklearn_fork.datasets.fetch_kddcup99",
    "sklearn_fork.datasets.fetch_lfw_pairs",
    "sklearn_fork.datasets.fetch_lfw_people",
    "sklearn_fork.datasets.fetch_olivetti_faces",
    "sklearn_fork.datasets.fetch_rcv1",
    "sklearn_fork.datasets.fetch_species_distributions",
    "sklearn_fork.datasets.get_data_home",
    "sklearn_fork.datasets.load_breast_cancer",
    "sklearn_fork.datasets.load_diabetes",
    "sklearn_fork.datasets.load_digits",
    "sklearn_fork.datasets.load_files",
    "sklearn_fork.datasets.load_iris",
    "sklearn_fork.datasets.load_linnerud",
    "sklearn_fork.datasets.load_sample_image",
    "sklearn_fork.datasets.load_svmlight_file",
    "sklearn_fork.datasets.load_svmlight_files",
    "sklearn_fork.datasets.load_wine",
    "sklearn_fork.datasets.make_biclusters",
    "sklearn_fork.datasets.make_blobs",
    "sklearn_fork.datasets.make_checkerboard",
    "sklearn_fork.datasets.make_circles",
    "sklearn_fork.datasets.make_classification",
    "sklearn_fork.datasets.make_friedman1",
    "sklearn_fork.datasets.make_friedman2",
    "sklearn_fork.datasets.make_friedman3",
    "sklearn_fork.datasets.make_gaussian_quantiles",
    "sklearn_fork.datasets.make_hastie_10_2",
    "sklearn_fork.datasets.make_low_rank_matrix",
    "sklearn_fork.datasets.make_moons",
    "sklearn_fork.datasets.make_multilabel_classification",
    "sklearn_fork.datasets.make_regression",
    "sklearn_fork.datasets.make_s_curve",
    "sklearn_fork.datasets.make_sparse_coded_signal",
    "sklearn_fork.datasets.make_sparse_spd_matrix",
    "sklearn_fork.datasets.make_sparse_uncorrelated",
    "sklearn_fork.datasets.make_spd_matrix",
    "sklearn_fork.datasets.make_swiss_roll",
    "sklearn_fork.decomposition.sparse_encode",
    "sklearn_fork.feature_extraction.grid_to_graph",
    "sklearn_fork.feature_extraction.img_to_graph",
    "sklearn_fork.feature_extraction.image.extract_patches_2d",
    "sklearn_fork.feature_extraction.image.reconstruct_from_patches_2d",
    "sklearn_fork.feature_selection.chi2",
    "sklearn_fork.feature_selection.f_classif",
    "sklearn_fork.feature_selection.f_regression",
    "sklearn_fork.feature_selection.mutual_info_classif",
    "sklearn_fork.feature_selection.mutual_info_regression",
    "sklearn_fork.feature_selection.r_regression",
    "sklearn_fork.inspection.partial_dependence",
    "sklearn_fork.inspection.permutation_importance",
    "sklearn_fork.linear_model.orthogonal_mp",
    "sklearn_fork.metrics.accuracy_score",
    "sklearn_fork.metrics.auc",
    "sklearn_fork.metrics.average_precision_score",
    "sklearn_fork.metrics.balanced_accuracy_score",
    "sklearn_fork.metrics.brier_score_loss",
    "sklearn_fork.metrics.calinski_harabasz_score",
    "sklearn_fork.metrics.check_scoring",
    "sklearn_fork.metrics.completeness_score",
    "sklearn_fork.metrics.class_likelihood_ratios",
    "sklearn_fork.metrics.classification_report",
    "sklearn_fork.metrics.cluster.adjusted_mutual_info_score",
    "sklearn_fork.metrics.cluster.contingency_matrix",
    "sklearn_fork.metrics.cluster.entropy",
    "sklearn_fork.metrics.cluster.fowlkes_mallows_score",
    "sklearn_fork.metrics.cluster.homogeneity_completeness_v_measure",
    "sklearn_fork.metrics.cluster.normalized_mutual_info_score",
    "sklearn_fork.metrics.cluster.silhouette_samples",
    "sklearn_fork.metrics.cluster.silhouette_score",
    "sklearn_fork.metrics.cohen_kappa_score",
    "sklearn_fork.metrics.confusion_matrix",
    "sklearn_fork.metrics.coverage_error",
    "sklearn_fork.metrics.d2_absolute_error_score",
    "sklearn_fork.metrics.d2_pinball_score",
    "sklearn_fork.metrics.d2_tweedie_score",
    "sklearn_fork.metrics.davies_bouldin_score",
    "sklearn_fork.metrics.dcg_score",
    "sklearn_fork.metrics.det_curve",
    "sklearn_fork.metrics.explained_variance_score",
    "sklearn_fork.metrics.f1_score",
    "sklearn_fork.metrics.fbeta_score",
    "sklearn_fork.metrics.get_scorer",
    "sklearn_fork.metrics.hamming_loss",
    "sklearn_fork.metrics.hinge_loss",
    "sklearn_fork.metrics.homogeneity_score",
    "sklearn_fork.metrics.jaccard_score",
    "sklearn_fork.metrics.label_ranking_average_precision_score",
    "sklearn_fork.metrics.label_ranking_loss",
    "sklearn_fork.metrics.log_loss",
    "sklearn_fork.metrics.make_scorer",
    "sklearn_fork.metrics.matthews_corrcoef",
    "sklearn_fork.metrics.max_error",
    "sklearn_fork.metrics.mean_absolute_error",
    "sklearn_fork.metrics.mean_absolute_percentage_error",
    "sklearn_fork.metrics.mean_gamma_deviance",
    "sklearn_fork.metrics.mean_pinball_loss",
    "sklearn_fork.metrics.mean_poisson_deviance",
    "sklearn_fork.metrics.mean_squared_error",
    "sklearn_fork.metrics.mean_squared_log_error",
    "sklearn_fork.metrics.mean_tweedie_deviance",
    "sklearn_fork.metrics.median_absolute_error",
    "sklearn_fork.metrics.multilabel_confusion_matrix",
    "sklearn_fork.metrics.mutual_info_score",
    "sklearn_fork.metrics.ndcg_score",
    "sklearn_fork.metrics.pair_confusion_matrix",
    "sklearn_fork.metrics.adjusted_rand_score",
    "sklearn_fork.metrics.pairwise.additive_chi2_kernel",
    "sklearn_fork.metrics.pairwise.cosine_distances",
    "sklearn_fork.metrics.pairwise.cosine_similarity",
    "sklearn_fork.metrics.pairwise.haversine_distances",
    "sklearn_fork.metrics.pairwise.laplacian_kernel",
    "sklearn_fork.metrics.pairwise.linear_kernel",
    "sklearn_fork.metrics.pairwise.manhattan_distances",
    "sklearn_fork.metrics.pairwise.nan_euclidean_distances",
    "sklearn_fork.metrics.pairwise.paired_cosine_distances",
    "sklearn_fork.metrics.pairwise.paired_euclidean_distances",
    "sklearn_fork.metrics.pairwise.paired_manhattan_distances",
    "sklearn_fork.metrics.pairwise.polynomial_kernel",
    "sklearn_fork.metrics.pairwise.rbf_kernel",
    "sklearn_fork.metrics.pairwise.sigmoid_kernel",
    "sklearn_fork.metrics.precision_recall_curve",
    "sklearn_fork.metrics.precision_recall_fscore_support",
    "sklearn_fork.metrics.precision_score",
    "sklearn_fork.metrics.r2_score",
    "sklearn_fork.metrics.rand_score",
    "sklearn_fork.metrics.recall_score",
    "sklearn_fork.metrics.roc_auc_score",
    "sklearn_fork.metrics.roc_curve",
    "sklearn_fork.metrics.top_k_accuracy_score",
    "sklearn_fork.metrics.v_measure_score",
    "sklearn_fork.metrics.zero_one_loss",
    "sklearn_fork.model_selection.cross_validate",
    "sklearn_fork.model_selection.learning_curve",
    "sklearn_fork.model_selection.permutation_test_score",
    "sklearn_fork.model_selection.train_test_split",
    "sklearn_fork.model_selection.validation_curve",
    "sklearn_fork.neighbors.sort_graph_by_row_values",
    "sklearn_fork.preprocessing.add_dummy_feature",
    "sklearn_fork.preprocessing.binarize",
    "sklearn_fork.preprocessing.label_binarize",
    "sklearn_fork.preprocessing.maxabs_scale",
    "sklearn_fork.preprocessing.normalize",
    "sklearn_fork.preprocessing.scale",
    "sklearn_fork.random_projection.johnson_lindenstrauss_min_dim",
    "sklearn_fork.svm.l1_min_c",
    "sklearn_fork.tree.export_text",
    "sklearn_fork.tree.plot_tree",
    "sklearn_fork.utils.gen_batches",
    "sklearn_fork.utils.resample",
]


@pytest.mark.parametrize("func_module", PARAM_VALIDATION_FUNCTION_LIST)
def test_function_param_validation(func_module):
    """Check param validation for public functions that are not wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)

    parameter_constraints = getattr(func, "_skl_parameter_constraints")

    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )


PARAM_VALIDATION_CLASS_WRAPPER_LIST = [
    ("sklearn_fork.cluster.affinity_propagation", "sklearn_fork.cluster.AffinityPropagation"),
    ("sklearn_fork.cluster.mean_shift", "sklearn_fork.cluster.MeanShift"),
    ("sklearn_fork.cluster.spectral_clustering", "sklearn_fork.cluster.SpectralClustering"),
    ("sklearn_fork.covariance.graphical_lasso", "sklearn_fork.covariance.GraphicalLasso"),
    ("sklearn_fork.covariance.ledoit_wolf", "sklearn_fork.covariance.LedoitWolf"),
    ("sklearn_fork.covariance.oas", "sklearn_fork.covariance.OAS"),
    ("sklearn_fork.decomposition.dict_learning", "sklearn_fork.decomposition.DictionaryLearning"),
    ("sklearn_fork.decomposition.fastica", "sklearn_fork.decomposition.FastICA"),
    ("sklearn_fork.decomposition.non_negative_factorization", "sklearn_fork.decomposition.NMF"),
    ("sklearn_fork.preprocessing.minmax_scale", "sklearn_fork.preprocessing.MinMaxScaler"),
    ("sklearn_fork.preprocessing.power_transform", "sklearn_fork.preprocessing.PowerTransformer"),
    (
        "sklearn_fork.preprocessing.quantile_transform",
        "sklearn_fork.preprocessing.QuantileTransformer",
    ),
    ("sklearn_fork.preprocessing.robust_scale", "sklearn_fork.preprocessing.RobustScaler"),
]


@pytest.mark.parametrize(
    "func_module, class_module", PARAM_VALIDATION_CLASS_WRAPPER_LIST
)
def test_class_wrapper_param_validation(func_module, class_module):
    """Check param validation for public functions that are wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)

    module_name, class_name = class_module.rsplit(".", 1)
    module = import_module(module_name)
    klass = getattr(module, class_name)

    parameter_constraints_func = getattr(func, "_skl_parameter_constraints")
    parameter_constraints_class = getattr(klass, "_parameter_constraints")
    parameter_constraints = {
        **parameter_constraints_class,
        **parameter_constraints_func,
    }
    parameter_constraints = {
        k: v for k, v in parameter_constraints.items() if k in func_params
    }

    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )
