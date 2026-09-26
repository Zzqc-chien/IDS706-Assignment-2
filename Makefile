# Heart Disease UCI analysis - common commands (run `make help`)

IMAGE ?= heart-analysis

.PHONY: help install format lint test run docker-build docker-run docker-test clean all

help:
	@echo "make install       Install Python dependencies"
	@echo "make format        Format code with black"
	@echo "make lint          Check formatting (black) and style (flake8)"
	@echo "make test          Run pytest with coverage"
	@echo "make run           Run the analysis (figures -> figures/)"
	@echo "make docker-build  Build the Docker image"
	@echo "make docker-run    Run the analysis in a container (figures -> output/)"
	@echo "make docker-test   Run the test suite inside the container"
	@echo "make all           install + lint + test"

install:
	pip install -r requirements.txt

format:
	black .

lint:
	black --check .
	flake8 .

test:
	python -m pytest -v --cov=analysis --cov-report=term-missing

run:
	python analysis.py

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	mkdir -p output
	docker run --rm -v "$(CURDIR)/output:/app/output" $(IMAGE)

docker-test:
	docker run --rm $(IMAGE) python -m pytest -q

clean:
	rm -rf output .pytest_cache .coverage __pycache__ tests/__pycache__

all: install lint test
