from setuptools import find_packages, setup  # pragma: no cover

setup(  # pragma: no cover
    name="Scikit-longitudinal",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.2.1",
        "pandas>=1.5.3",
        "matplotlib>=3.7.0",
        "jupyter>=1.0.0",
        "overrides>=7.3.1",
        "ray>=2.3.1",
        "statsmodels>=0.13.5",
    ],
    python_requires=">=3.10",
)
