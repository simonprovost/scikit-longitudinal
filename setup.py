from setuptools import find_packages, setup  # pragma: no cover

setup(  # pragma: no cover
    name="Scikit-longitudinal",
    version="0.0.2",
    description="Scikit-longitudinal, an open-source Python lib for longitudinal data analysis, builds on Scikit-learn's foundation. It offers specialized tools to tackle challenges of repeated measures data, ideal for researchers, data scientists, & analysts.",
    author="Provost Simon, Alex Freitas",
    author_email="simon.gilbert.provost@gmail.com, a.a.freitas@kent.ac.uk",
    packages=find_packages(),
    install_requires=[
        "deep-forest>=0.1.7",
        "starboost==0.0.2",
        "scipy>=1.5.0",
        "pandas<2.0.0,>=1.5.3",
        "matplotlib<4.0.0,>=3.7.0",
        "jupyter<2.0.0,>=1.0.0",
        "overrides<8.0.0,>=7.3.1",
        "ray<3.0.0,>=2.3.1",
        "statsmodels<1.0.0,>=0.14.0",
        "numpy==1.23.3",
        "graphviz<1.0.0,>=0.20.1",
        "arff<1.0,>=0.9",
        "threadpoolctl<4.0.0,>=3.1.0",
        "configspace<1.0.0,>=0.7.1",
        "stopit>=1.1.2",
        "scikit-lexicographical-trees==0.0.2",
        "rich>=13.6.0",
    ],
    extras_require={
        "dev": [
            "autoflake<3.0.0,>=2.1.1",
            "flake8-docstrings>=1.7.0",
            "pydocstyle>=6.3.0",
            "flake8>=6.0.0",
            "pylint>=2.17.1",
            "isort>=5.12.0",
            "black>=23.10.1",
            "autopep8>=2.0.2",
            "docformatter[tomli]<2.0.0,>=1.6.4",
            "pytest-cov>=4.0.0",
            "genbadge>=1.1.0",
            "mkdocs==1.6.0",
            "mkdocs-get-deps==0.2.0",
            "mkdocs-material==9.5.27",
            "mkdocs-material[imaging]",
            "mkdocs-material-extensions==1.3.1",
            "mkdocs-minify-plugin==0.8.0"
        ]
    },
    python_requires=">=3.9,<3.10",
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)