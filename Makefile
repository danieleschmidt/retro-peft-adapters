# Retro-PEFT-Adapters Makefile

.PHONY: help install install-dev test lint format type-check clean build docs serve-docs docker-build docker-run setup-env

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install package
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

install-all: ## Install package with all optional dependencies
	pip install -e ".[all-backends,dev,test,docs]"

# Development targets
setup-env: ## Setup development environment
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && make install-dev

format: ## Format code with black and isort
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black --line-length 100

lint: ## Run linting checks
	flake8 src/ tests/ --max-line-length 100 --extend-ignore E203,W503
	mypy src/retro_peft --ignore-missing-imports

type-check: ## Run type checking
	mypy src/retro_peft --ignore-missing-imports --disallow-untyped-defs

security: ## Run security checks
	bandit -r src/ -f json -o bandit-report.json
	safety check

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

# Testing targets
test: ## Run tests
	pytest tests/ -v --cov=src/retro_peft --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	pytest tests/ -v -x

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-unit: ## Run unit tests
	pytest tests/unit/ -v

test-watch: ## Run tests in watch mode
	ptw tests/ -- -v

# Build targets
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

publish-test: build ## Publish to test PyPI
	python -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	python -m twine upload dist/*

# Documentation targets
docs: ## Build documentation
	cd docs && make html

docs-clean: ## Clean documentation
	cd docs && make clean

serve-docs: docs ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8080

docs-watch: ## Watch and rebuild docs
	sphinx-autobuild docs docs/_build/html --host 0.0.0.0 --port 8080

# Docker targets
docker-build: ## Build Docker image
	docker build -t retro-peft-adapters .

docker-build-dev: ## Build development Docker image
	docker build -f .devcontainer/Dockerfile -t retro-peft-adapters:dev .

docker-run: ## Run Docker container
	docker run -it --rm -p 8000:8000 retro-peft-adapters

docker-run-dev: ## Run development Docker container
	docker run -it --rm -v $(PWD):/workspace -p 8000:8000 retro-peft-adapters:dev

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose down

# Database targets
db-init: ## Initialize database
	python -c "from src.retro_peft.database import DatabaseManager; DatabaseManager()"

db-migrate: ## Run database migrations
	python scripts/migrate_db.py

db-reset: ## Reset database (WARNING: deletes all data)
	rm -f ./data/retro_peft.db
	make db-init

# Benchmarking targets
benchmark: ## Run performance benchmarks
	python benchmarks/run_benchmarks.py

benchmark-adapters: ## Benchmark adapter performance
	python benchmarks/adapter_benchmarks.py

benchmark-retrieval: ## Benchmark retrieval performance
	python benchmarks/retrieval_benchmarks.py

# Model targets
download-models: ## Download test models
	python scripts/download_models.py

train-example: ## Train example adapter
	python examples/train_retro_lora.py

serve-example: ## Serve example model
	python examples/serve_adapter.py

# Index targets
build-index: ## Build example retrieval index
	python examples/build_index.py

# Monitoring targets
start-monitoring: ## Start monitoring stack
	docker-compose -f docker-compose.monitoring.yml up -d

stop-monitoring: ## Stop monitoring stack
	docker-compose -f docker-compose.monitoring.yml down

# Development utilities
jupyter: ## Start Jupyter lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

profile: ## Run profiling
	python -m cProfile -o profile.stats examples/profile_example.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

check-deps: ## Check for dependency updates
	pip list --outdated

update-deps: ## Update dependencies
	pip-compile requirements.in
	pip-compile requirements-dev.in

# Release targets
tag-release: ## Tag new release (requires VERSION env var)
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)

# All-in-one targets
check-all: format lint type-check security test ## Run all checks

setup-project: clean install-dev db-init download-models ## Setup entire project

ci: check-all build ## Run CI pipeline locally