.PHONY: setup data labels features baselines train eval export web benchmark test all paper-results clean

PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)
CONFIG ?= configs/default.yaml
SEED ?= 42

setup:
	@echo "Setting up pwdstrength environment..."
	mkdir -p data/raw data/processed models/export results/preds results/train_logs web/model tests/web_parity
	$(PYTHON) -m pip install --break-system-packages -e . || true

data:
	@echo "Preparing synthetic dataset and corpus..."
	$(PYTHON) scripts/make_synthetic.py --config $(CONFIG) --seed $(SEED)
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG) --corpus synthetic_demo --seed $(SEED)

labels:
	@echo "Generating out-of-fold reference attacker labels..."
	$(PYTHON) scripts/make_labels.py --config $(CONFIG) --corpus synthetic_demo --seed $(SEED)

features:
	@echo "Extracting 22-dimensional pattern features..."
	$(PYTHON) scripts/extract_features.py --config $(CONFIG) --corpus synthetic_demo

baselines:
	@echo "Evaluating Baselines B1-B4..."
	$(PYTHON) scripts/run_baselines.py --config $(CONFIG) --corpus synthetic_demo --seed $(SEED)

train:
	@echo "Training proposed HybridStrengthNet and B5 BiLSTM..."
	$(PYTHON) scripts/train.py --config $(CONFIG) --corpus synthetic_demo --seed $(SEED) --fast

eval:
	@echo "Computing evaluation metrics, ablations, tables, and figures..."
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --corpus synthetic_demo

export:
	@echo "Exporting PyTorch model to ONNX FP32 and INT8..."
	$(PYTHON) scripts/export_onnx.py --config $(CONFIG)
	$(PYTHON) scripts/benchmark.py --config $(CONFIG)

web: export
	@echo "Syncing exported ONNX artifacts to web/model..."
	mkdir -p web/model
	cp -f models/export/* web/model/ || true
	@echo "Web demo ready at web/index.html (or root dev server)"

test:
	@echo "Running test suite..."
	$(PYTHON) -m pytest tests/ -v || $(PYTHON) tests/runner.py

paper-results:
	@echo "Synchronizing paper tables and hypotheses..."
	$(PYTHON) scripts/paper_results.py --config $(CONFIG)

all: data labels features baselines train eval export web paper-results
	@echo "Full pipeline completed successfully!"

clean:
	rm -rf data/processed/* results/preds/* results/train_logs/* models/export/*
