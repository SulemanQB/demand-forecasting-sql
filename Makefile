PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: help download build-features backtest all test

help:
	@echo "Targets: download, build-features, backtest, all, test"

download:
	$(PYTHON) -m demandforecast download

build-features:
	$(PYTHON) -m demandforecast build-features

backtest:
	$(PYTHON) -m demandforecast backtest

all:
	$(PYTHON) -m demandforecast all

test:
	$(PYTHON) -m pytest -q
