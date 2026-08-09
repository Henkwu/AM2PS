.PHONY: install smoke train-cxr train-covid ablation

install:
	pip install -r requirements.txt
	pip install -e .

smoke:
	python tests/smoke_test.py

train-cxr:
	python train.py --config configs/chestxray2017.yaml --device cuda

train-covid:
	python train.py --config configs/chestxray_covid19.yaml --device cuda

ablation:
	python experiments/run_ablation.py --config configs/chestxray2017.yaml --output-root outputs/ablation_chestxray2017 --device cuda
