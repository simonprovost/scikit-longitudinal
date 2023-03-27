# flake8: noqa
# pylint: skip-file

from math import log


def discrete_entropy(samples, base=2):
    """Calculate the discrete entropy of a list of samples, where each sample can be any hashable object.

    Notes:
        This function computes a discrete entropy estimator given a list of samples which can be any hashable
        object.

    Args:
        samples: A list of hashable objects.
        base: An integer that represents the logarithmic base to use in the entropy calculation.

    Returns:
        A float that represents the discrete entropy.
    """

    return entropy_from_probabilities(histogram(samples), base=base)


def discrete_mutual_information(x, y):
    """Calculate the discrete mutual information of two lists of samples, where each sample can be any hashable object.

    Notes:
        This function computes a discrete mutual information estimator given a list of samples which can be any
        hashable object.

    Args:
        x: A list of hashable objects.
        y: A list of hashable objects.

    Returns:
        A float that represents the discrete mutual information.
    """

    return -discrete_entropy(list(zip(x, y))) + discrete_entropy(x) + discrete_entropy(y)


def histogram(samples):
    """Compute a histogram from a list of samples.

    Args:
        samples: A list of hashable objects.

    Returns:
        A list of floats that represents the frequencies of each sample in the input list.
    """
    frequencies = {}
    for sample in samples:
        frequencies[sample] = frequencies.get(sample, 0) + 1
    return map(lambda freq: float(freq) / len(samples), frequencies.values())


def entropy_from_probabilities(probabilities, base=2):
    """Compute the entropy from a list of normalized probabilities.

    Notes:
        This function computes the entropy (base 2) from a list of normalized probabilities.
    Args:
        probabilities: A list of floats that represents the probabilities.
        base: An integer that represents the logarithmic base to use in the entropy calculation.

    Returns:
        A float that represents the entropy.
    """
    return -sum(map(elementwise_log, probabilities)) / log(base)


def elementwise_log(x):
    """Compute the element-wise logarithm for entropy calculation.

    Args:
        x: A float.

    Returns:
        A float that represents the element-wise logarithm.
    """
    return 0 if x <= 0.0 or x >= 1.0 else x * log(x)


def information_gain(feature1, feature2):
    """Calculate the information gain between two lists of samples.

    Args:
        feature1: A numpy array of shape (n_samples,).
        feature2: A numpy array of shape (n_samples,).

    Returns:
        The information gain as a float.
    """

    return discrete_entropy(feature1) - conditional_entropy(feature1, feature2)


def conditional_entropy(feature1, feature2):
    """Calculate the conditional entropy between two lists of samples.

    Args:
        feature1: A numpy array of shape (n_samples,).
        feature2: A numpy array of shape (n_samples,).

    Returns:
        The conditional entropy as a float.
    """

    return discrete_entropy(feature1) - discrete_mutual_information(feature1, feature2)


def symmetrical_uncertainty(feature1, feature2):
    """Calculate the symmetrical uncertainty between two lists of samples.

    Args:
        feature1: A numpy array of shape (n_samples,).
        feature2: A numpy array of shape (n_samples,).

    Returns:
        The symmetrical uncertainty as a float.
    """
    # calculate information gain of f1 and f2, t1 = ig(f1, f2)
    t1 = information_gain(feature1, feature2)
    # calculate entropy of f1
    t2 = discrete_entropy(feature1)
    # calculate entropy of f2
    t3 = discrete_entropy(feature2)

    return 2.0 * t1 / (t2 + t3)
