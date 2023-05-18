.PHONY: test clean_cache isort black flake8 pylint autopep8 precommit lint format docformatter autoflake compile_scikit_tree

tests:
	@echo "Running Poetry Pytests..."
	poetry run pytest --cov=./ --cov-report=html --cov-config=.coveragerc --cov-report=html:htmlcov/scikit_longitudinal -s -vv --capture=no
	@echo "Done."

clean:
	rm -rf htmlcov .pytest_cache .*_cache
	find . -type d -name "__pycache__" -exec rm -r {} +

isort:
	@echo "Running isort..."
	poetry run isort --skip=scikit-learn scikit_longitudinal
	@echo "Done."

black:
	@echo "Running black..."
	poetry run black --line-length 120 --preview --exclude=./scikit-learn/* scikit_longitudinal
	@echo "Done."

flake8:
	@echo "Running flake8..."
	poetry run flake8 --exclude=./scikit-learn/* scikit_longitudinal
	@echo "Done."

autoflake:
	@echo "Running autoflake..."
	poetry run autoflake --in-place --remove-all-unused-imports --recursive --exclude=./scikit-learn/* scikit_longitudinal
	@echo "Done."

pylint:
	@echo "Running pylint..."
	poetry run pylint --rcfile=pylintrc scikit_longitudinal/
	@echo "Done."

precommit:
	@echo "Running pre-commit hooks..."
	git ls-files -- 'scikit_longitudinal/*.py' | xargs poetry run pre-commit run --files
	@echo "Done."

autopep8:
	@echo "Running autopep8..."
	poetry run autopep8 --in-place --aggressive --aggressive --max-line-length 120 --recursive --exclude=./scikit-learn/* scikit_longitudinal
	@echo "Done."

docformatter:
	@echo "Running docformatter..."
	poetry run docformatter --in-place --recursive --wrap-summaries 120 --wrap-descriptions 120 --blank --make-summary-multi-line --exclude=./scikit-learn/ scikit_longitudinal
	@echo "Done."

setup_git_hooks:
	cp scripts/pre-push .git/hooks/pre-push
	chmod +x .git/hooks/pre-push

compile_scikit_tree:
	@echo "Sourcing .env..."
	source .env ; \
	@echo "Removing poetry.lock..."
	rm -rf poetry.lock ; \
	@echo "Removing poetry environment..."
	poetry env remove $${POETRY_ENV_NAME} ; \
	@echo "Moving to scikit-learn directory..."
	cd scikit-learn ; \
	@echo "Running make clean..."
	make clean ; \
	@echo "Exporting LDFLAGS..."
	export LDFLAGS=$${LDFLAGS} ; \
	@echo "Exporting CPPFLAGS..."
	export CPPFLAGS=$${CPPFLAGS} ; \
	@echo "Installing with pip..."
	pip install --verbose --no-build-isolation --editable . ; \
	@echo "Moving back to parent directory..."
	cd .. ; \
	@echo "Installing with poetry..."
	poetry install


install_dev:
	@echo "Install dev (can take a couple of minutes) ..."
	@if ! command -v poetry > /dev/null; then \
		echo "Poetry is not installed. Please install Poetry to continue."; \
		exit 1; \
	fi
	@if [ ! -f .env ]; then \
		echo "The .env file is missing at the project root. Please create it to continue."; \
		exit 1; \
	fi
	@if ! grep -q '^LDFLAGS=.*' .env || ! grep -q '^CPPFLAGS=.*' .env; then \
		echo "LDFLAGS and/or CPPFLAGS are missing in the .env file. Please add them to continue."; \
		exit 1; \
	fi
	@if ! grep -q '^POETRY_ENV_NAME=.*' .env; then \
		echo "POETRY_ENV_NAME is missing in the .env file. Please add it to continue."; \
		exit 1; \
	fi
	@if ! command -v pip > /dev/null; then \
		echo "pip is not installed. Please install pip to continue."; \
		exit 1; \
	fi
	$(MAKE) compile_scikit_tree
	@echo "Dev installed."
	$(MAKE) tests

create_elsa_core_datasets:
	@echo "Creating ELSA Core datasets..."
	poetry run python ./scikit_longitudinal/data_preparation/elsa_handler.py --csv_path "data/elsa/elsa_core_dd.csv" --file_format "csv" --dir_output "data/elsa/core/csv" --elsa_type "core"
	poetry run python ./scikit_longitudinal/data_preparation/elsa_handler.py --csv_path "data/elsa/elsa_core_dd.csv" --file_format "arff" --dir_output "data/elsa/core/arff" --elsa_type "core"
	@echo "Done."

create_elsa_nurse_datasets:
	@echo "Creating ELSA Nurse datasets..."
	poetry run python ./scikit_longitudinal/data_preparation/elsa_handler.py --csv_path "data/elsa/elsa_nurse_dd.csv" --file_format "csv" --dir_output "data/elsa/nurse/csv" --elsa_type "nurse"
	poetry run python ./scikit_longitudinal/data_preparation/elsa_handler.py --csv_path "data/elsa/elsa_nurse_dd.csv" --file_format "arff" --dir_output "data/elsa/nurse/arff" --elsa_type "nurse"
	@echo "Done."

run_cfs_per_group_experiment:
	@echo "Running CFS per group experiment..."
	poetry run python ./scikit_longitudinal/experiments/cfs_experiment.py
	@echo "Done."

run_nested_tree_experiment:
	@echo "Running nested tree experiment..."
	poetry run python ./scikit_longitudinal/experiments/nested_tree_experiment.py
	@echo "Done."

run_lexico_rf_experiment:
	@echo "Running Lexico RF experiment..."
	poetry run python ./scikit_longitudinal/experiments/lexico_rf_experiment.py
	@echo "Done."

run_all_experiments:
	@echo "Running all experiments..."
	$(MAKE) run_nested_tree_experiment
	$(MAKE) run_lexico_rf_experiment
	$(MAKE) run_cfs_per_group_experiment
	@echo "Done."


lint: flake8 pylint

format: isort autopep8 black autoflake docformatter

all: format lint precommit tests
