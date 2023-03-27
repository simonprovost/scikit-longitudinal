.PHONY: test clean_cache isort black flake8 pylint autopep8 precommit lint format docs

tests:
	@echo "Running Poetry Pytests..."
	poetry run pytest --cov=./ --cov-report=html --cov-config=.coveragerc --cov-report=html:htmlcov/scikit_longitudinal
	@echo "Done."

clean_cache:
	rm -rf __pycache__ htmlcov .pytest_cache .mypy_cache

isort:
	@echo "Running isort..."
	poetry run isort src
	@echo "Done."

black:
	@echo "Running black..."
	poetry run black --line-length 120 src
	@echo "Done."

flake8:
	@echo "Running flake8..."
	poetry run flake8 src
	@echo "Done."

pylint:
	@echo "Running pylint..."
	poetry run pylint --rcfile=pylintrc scikit_longitudinal/
	@echo "Done."

precommit:
	@echo "Running pre-commit hooks..."
	poetry run pre-commit run --all-files
	@echo "Done."

autopep8:
	@echo "Running autopep8..."
	poetry run autopep8 --in-place --aggressive --aggressive --recursive src
	@echo "Done."

docs:
	@echo "Generating Sphinx documentation..."
	cd docs && poetry run make html
	@echo "Done."

clean_docs:
	@echo "Cleaning Sphinx documentation..."
	rm -rm docs/_build
	@echo "Done."

lint: flake8 pylint

format: isort autopep8 black

all: format lint precommit tests docs
