from setuptools import find_packages, setup  # pragma: no cover

setup(  # pragma: no cover
    name="Scikit-longitudinal",
    version="0.0.2",
    description="Scikit-longitudinal, an open-source Python lib for longitudinal data analysis, builds on "
    "Scikit-learn's foundation. It offers specialized tools to tackle challenges of repeated measures "
    "data, ideal for researchers, data scientists, & analysts.",
    author="Simon Provost",
    author_email="sgp28@kent.ac.uk",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.3",
        "matplotlib>=3.7.0",
        "jupyter>=1.0.0",
        "overrides>=7.3.1",
        "ray>=2.3.1",
        "statsmodels>=0.13.5",
        "numpy>=1.21.2",
        "graphviz>=0.20.1",
        "autoflake>=2.1.1",
        "docformatter>=1.6.4",
        "arff>=0.9",
        "threadpoolctl>=3.1.0",
        "scikit-learn-tree @ ./scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=4.0.0",
            "flake8-docstrings>=1.7.0",
            "pydocstyle>=6.3.0",
            "genbadge>=1.1.0",
            "flake8>=6.0.0",
            "pylint>=2.17.1",
            "isort>=5.12.0",
            "autopep8>=2.0.2",
            "black>=23.1.0",
            "pre-commit>=3.2.0",
            "twine>=4.0.2",
        ]
    },
    python_requires=">=3.8.1,<3.11",
)
