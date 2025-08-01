# Makefile for Notes on DSP

# Default Python binary
PYTHON := poetry run python

# Directories to check
SRC_DIRS := src tests
NOTEBOOK_DIR := notebooks
DOCS_DIR := docs

# Help command
.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# Installation commands
.PHONY: install install-dev install-notebook install-data install-test install-minimal
install: ## Install all dependencies (base + dev + extras)
	poetry install --with dev,notebook,data-science,viz,test

install-dev: ## Install only dev dependencies
	poetry install --with dev

install-notebook: ## Install notebook dependencies
	poetry install --with notebook

install-data: ## Install data-science dependencies and viz dependencies
	poetry install --with data-science,viz

install-test: ## Install testing dependencies
	poetry install --with test

install-minimal: ## Install base (no extras)
	poetry install

# Development commands
.PHONY: run setup notebook
run: ## Run the application (example main)
	$(PYTHON) src/dsp/__main__.py

setup: ## Run full environment setup
	bash scripts/setup_env.sh
	
notebook: ## Launch Jupyter Lab
	poetry run jupyter lab

# Code quality commands
.PHONY: lint format typecheck check
lint: ## Run ruff linter on src and tests
	poetry run ruff check $(SRC_DIRS)

format: ## Format code with black
	poetry run black $(SRC_DIRS)

typecheck: ## Run type checks with mypy
	poetry run mypy $(SRC_DIRS)

check: format lint typecheck ## Run all code quality checks

# Testing commands
.PHONY: test test-watch test-cov test-cov-html
test: ## Run pytest on tests
	poetry run pytest --cov=src

test-watch: ## Auto-rerun tests on file change
	poetry run ptw --runner "pytest --cov=src"

test-cov: ## Run tests with coverage report
	poetry run pytest --cov=src --cov-report=term-missing

test-cov-html: ## Run tests with HTML coverage report
	poetry run pytest --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Documentation commands
.PHONY: docs docs-serve docs-build
docs: ## Build and serve documentation
	poetry run mkdocs serve

docs-serve: ## Serve documentation
	poetry run mkdocs serve

docs-build: ## Build documentation
	poetry run mkdocs build

# Data science commands
.PHONY: dvc-pull dvc-push dvc-status
dvc-pull: ## Pull latest data from DVC remote
	poetry run dvc pull

dvc-push: ## Push data changes to DVC remote
	poetry run dvc push

dvc-status: ## Check DVC data status
	poetry run dvc status

# MLflow commands
.PHONY: mlflow-ui
mlflow-ui: ## Start MLflow tracking server
	poetry run mlflow ui

# Streamlit commands
.PHONY: streamlit
streamlit: ## Run Streamlit dashboard
	poetry run streamlit run src/dsp/visualization/dashboard.py

# Cleanup commands
.PHONY: clean clean-pyc clean-build clean-test clean-docs
clean: clean-pyc clean-build clean-test clean-docs ## Remove all build artifacts

clean-pyc: ## Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

clean-build: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/

clean-test: ## Remove test artifacts
	rm -rf .coverage htmlcov/ .pytest_cache/

clean-docs: ## Remove documentation build artifacts
	rm -rf site/
