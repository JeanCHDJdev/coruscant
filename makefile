.PHONY: install format

ENV_NAME = coruscant
PYTHON_VERSION = 3.13
PACKAGE = coruscant
ENV_FILE = environment.yaml

# installation within a conda environment
install:
	@echo "Checking for existing Conda environment: $(ENV_NAME)"
	@if conda info --envs | grep -q "^$(ENV_NAME) "; then \
		echo "Removing existing Conda environment: $(ENV_NAME)"; \
		conda env remove -y --name $(ENV_NAME); \
	fi
	@echo "Creating new Conda environment from $(ENV_FILE)"
	conda env create -f $(ENV_FILE)
	@echo "Done. To activate the environment, run: conda activate $(ENV_NAME)"

# formatter
format:
	conda run -n $(ENV_NAME) black $(PACKAGE) tests