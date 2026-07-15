"""Run release quality gates for a routed VLM serving manifest."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.model_manifest import ServingManifest
from src.ops.release_gate import (
    evaluate_prediction_records,
    evaluate_release_gate,
    load_prediction_jsonl,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/serving_manifest.json",
        help="Path to the serving manifest that defines quality gates.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSONL with prediction, answers, and task_type fields.",
    )
    parser.add_argument(
        "--report-out",
        default="outputs/reports/release_gate_report.json",
        help="Where to write the full quality-gate report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ServingManifest.load(args.manifest, project_root=PROJECT_ROOT)
    records = load_prediction_jsonl(args.predictions)
    metric_report = evaluate_prediction_records(records)
    gate_report = evaluate_release_gate(manifest, metric_report)

    result = {
        "quality_report": metric_report,
        "gate_report": gate_report,
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(gate_report, indent=2), flush=True)

    if not gate_report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
