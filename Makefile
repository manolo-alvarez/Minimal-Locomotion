# Makefile

define HELP_MESSAGE
genesis_playground

# Installing

1. Create a new Conda environment: `conda create --name genesis_playground python=3.12`
2. Activate the environment: `conda activate genesis_playground`
3. Install the package: `make install-dev`

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`
3. Run unit tests: `make test`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#          Build           #
# ------------------------ #

install:
	@pip install --verbose -e .
.PHONY: install

install-dev:
	@pip install --verbose -e '.[dev]'
.PHONY: install

build-ext:
	@python setup.py build_ext --inplace
.PHONY: build-ext

clean:
	rm -rf build dist *.so **/*.so **/*.pyi **/*.pyc **/*.pyd **/*.pyo **/__pycache__ *.egg-info .eggs/ .ruff_cache/
.PHONY: clean

# ------------------------ #
#       Static Checks      #
# ------------------------ #


format:
	@isort --profile black genesis_playground
	@black genesis_playground
	@ruff format genesis_playground
	@isort genesis_playground
.PHONY: format

static-checks:
	@isort --profile black --check --diff genesis_playground
	@black --diff --check genesis_playground
	@ruff check genesis_playground
	@mypy --install-types --non-interactive genesis_playground
.PHONY: lint

mypy-daemon:
	@dmypy run -- $(py-files)
.PHONY: mypy-daemon

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test:
	python -m pytest
.PHONY: test
