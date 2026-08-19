# Convenience targets. Everything here is a thin wrapper over the CLI.
.PHONY: help install install-train lint compile test test-fast smoke validate plan budget \
        pilot clean docker docker-test

PYTHON ?= python
export PYTHONHASHSEED = 0

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:            ## Install the CPU stack (no torch)
	$(PYTHON) -m pip install -e ".[validate,symbolic,dev]"

install-train:      ## Install everything, including torch/peft/accelerate
	$(PYTHON) -m pip install -e ".[all]"

lint:               ## Ruff
	ruff check src tests

compile:            ## Byte-compile every module
	$(PYTHON) -m compileall -q src

test:               ## Full test suite
	$(PYTHON) -m pytest -q

test-fast:          ## Skip torch-gated tests
	$(PYTHON) -m pytest -q -m "not torch"

smoke:              ## Offline end-to-end smoke suite
	$(PYTHON) -m rineya_lab.cli smoke --all

validate:           ## Strictly validate the dataset (dev + test)
	$(PYTHON) -m rineya_lab.cli validate-data --splits dev test

plan:               ## Show both training view plans
	$(PYTHON) -m rineya_lab.cli plan -c configs/train_full.yaml
	$(PYTHON) -m rineya_lab.cli plan -c configs/train_split_stateful.yaml

budget:             ## Compute matched budgets for the two arms
	$(PYTHON) -m rineya_lab.cli budget -c configs/train_full.yaml \
	  --config2 configs/train_split_stateful.yaml --sample 2000

pilot:              ## Full pilot on one seed (needs a GPU and a base checkpoint)
	scripts/run_pilot.sh 1

check: lint compile test smoke  ## Everything CI runs

clean:              ## Remove build and cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker:             ## Build the CPU image
	docker build -t rineya-lab:cpu --target base .

docker-test:        ## Run the test suite inside the CPU image
	docker compose run --rm tests
