PYTHON ?= python3
PIP ?= pip

DEVICE ?= cuda

.PHONY: help install test experiments-quick experiments-full plot-main

help:
	@echo "Targets:"
	@echo "  install           Install Python dependencies"
	@echo "  test              Run quick implementation tests"
	@echo "  experiments-quick  Run the full pipeline in quick mode"
	@echo "  experiments-full   Run the full pipeline (slower)"
	@echo "  plot-main          Regenerate the main report plot"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) test_implementation.py

experiments-quick:
	$(PYTHON) run_all_experiments.py --quick --device $(DEVICE)

experiments-full:
	$(PYTHON) run_all_experiments.py --device $(DEVICE)

plot-main:
	$(PYTHON) scripts/generate_main_plot.py
