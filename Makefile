.PHONY: install lint type-check security test bench clean build help

help:
	@echo "Available targets:"
	@echo "  install     Install package and dev dependencies"
	@echo "  lint        Run all linters (autoflake, isort, black, pylint)"
	@echo "  type-check  Run mypy type checking"
	@echo "  security    Run bandit security scan"
	@echo "  test        Run test suite with coverage"
	@echo "  bench       Run performance benchmarks"
	@echo "  clean       Remove build artifacts"
	@echo "  build       Build wheel and sdist"

install:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt
	pip install -e .

lint:
	autoflake --remove-all-unused-imports --in-place -r pyhdc tests
	isort pyhdc tests
	black pyhdc tests
	pylint --recursive=y --rcfile=pyproject.toml pyhdc tests

type-check:
	mypy pyhdc --ignore-missing-imports

security:
	bandit -r pyhdc -ll

test:
	pytest --cov=pyhdc --cov-report=term-missing --cov-report=xml

bench:
	pytest tests/benchmarks/ --benchmark-only --benchmark-autosave

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .coverage coverage.xml htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

build: clean
	python -m build