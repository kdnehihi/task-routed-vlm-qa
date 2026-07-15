.PHONY: test check-accelerator validate-artifacts prefetch-models evaluate-routed release-gate serve-api

test:
	pytest

check-accelerator:
	python scripts/check_accelerator.py

validate-artifacts:
	python scripts/validate_artifacts.py --manifest configs/serving_manifest.json

prefetch-models:
	python scripts/prefetch_models.py --manifest configs/serving_manifest.json

evaluate-routed:
	python scripts/evaluate_routed.py \
		--manifest configs/serving_manifest.json \
		--metadata-path data/processed/multitask/validation.jsonl \
		--predictions-path outputs/predictions/routed_validation.jsonl

release-gate:
	python scripts/run_release_gate.py \
		--manifest configs/serving_manifest.json \
		--predictions outputs/predictions/routed_validation.jsonl \
		--report-out outputs/reports/release_gate_report.json

serve-api:
	ROUTED_VLM_MANIFEST=configs/serving_manifest.json python scripts/serve_api.py
