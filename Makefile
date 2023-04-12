.PHONY: test clean_cache isort black flake8 pylint autopep8 precommit lint format

tests:
	@echo "Running Poetry Pytests..."
	poetry run pytest --cov=./ --cov-report=html --cov-config=.coveragerc --cov-report=html:htmlcov/scikit_longitudinal -s
	@echo "Done."

clean:
	rm -rf htmlcov .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -r {} +

isort:
	@echo "Running isort..."
	poetry run isort --skip=scikit-learn scikit_longitudinal
	@echo "Done."

black:
	@echo "Running black..."
	poetry run black --line-length 120 --exclude=./scikit-learn/* scikit_longitudinal
	@echo "Done."

flake8:
	@echo "Running flake8..."
	poetry run flake8 --exclude=./scikit-learn/* scikit_longitudinal
	@echo "Done."

pylint:
	@echo "Running pylint..."
	poetry run pylint --rcfile=pylintrc --ignore-patterns=./scikit-learn/* scikit_longitudinal/
	@echo "Done."

precommit:
	@echo "Running pre-commit hooks..."
	git ls-files -- 'scikit_longitudinal/*.py' | xargs poetry run pre-commit run --files
	@echo "Done."

autopep8:
	@echo "Running autopep8..."
	poetry run autopep8 --in-place --aggressive --aggressive --recursive --exclude=./scikit-learn/* scikit_longitudinal
	@echo "Done."

compile_scikit_tree:
	source .env && \
	rm -rf poetry.lock ; \
	poetry env remove $${POETRY_ENV_NAME} ; \
	cd scikit-learn && \
	make clean && \
	export LDFLAGS=$${LDFLAGS} && \
	export CPPFLAGS=$${CPPFLAGS} && \
	pip install --verbose --no-build-isolation --editable . && \
	cd .. && \
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




lint: flake8 pylint

format: isort autopep8 black

all: format lint precommit tests
