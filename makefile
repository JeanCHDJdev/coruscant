.PHONY: install install-env format

ENV_NAME = coruscant
PYTHON_VERSION = 3.10
PACKAGE = coruscant

# installation within a conda environment
install-env:
	@echo "Checking for existing Conda environment: $(ENV_NAME)"
	@if conda info --envs | grep -q "^$(ENV_NAME) "; then \
		echo "Removing existing Conda environment: $(ENV_NAME)"; \
		conda remove -y --name $(ENV_NAME) --all; \
	fi
	@echo "Creating new Conda environment: $(ENV_NAME)"
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)
	@echo "Installing package in editable mode in $(ENV_NAME)"
	conda run -n $(ENV_NAME) pip install -e .[dev]
	@echo "Done. To activate the environment, run: conda activate $(ENV_NAME)"

# installation in the current environment
install:
	@echo "Appending dev dependencies to current Conda env"
	pip install -e .[dev]

# formatter
format:
	black $(PACKAGE)