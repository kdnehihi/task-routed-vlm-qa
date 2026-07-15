"""Validate that the local routed VLM serving artifacts exist."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.model_manifest import ServingManifest


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/serving_manifest.json",
        help="Path to a JSON serving manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ServingManifest.load(args.manifest, project_root=PROJECT_ROOT)
    missing = manifest.validate_local_artifacts()
    runtime_warnings = collect_runtime_warnings(manifest)

    result = {
        "manifest": manifest.to_metadata(),
        "ok": not missing,
        "missing": missing,
        "runtime_warnings": runtime_warnings,
    }
    print(json.dumps(result, indent=2), flush=True)

    if missing:
        raise SystemExit(1)


def collect_runtime_warnings(manifest: ServingManifest) -> list[str]:
    """Return non-fatal runtime compatibility warnings."""
    warnings = []
    if manifest.scikit_learn_version:
        try:
            import sklearn
        except ImportError:
            warnings.append("scikit-learn is not installed in this environment.")
        else:
            if sklearn.__version__ != manifest.scikit_learn_version:
                warnings.append(
                    "scikit-learn version mismatch: "
                    f"runtime={sklearn.__version__}, "
                    f"manifest={manifest.scikit_learn_version}"
                )
    return warnings


if __name__ == "__main__":
    main()
