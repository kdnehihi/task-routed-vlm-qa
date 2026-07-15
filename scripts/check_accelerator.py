"""Print the local accelerator status for routed VLM serving."""

from __future__ import annotations

import json
import platform


def main() -> None:
    import torch

    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    cuda_available = torch.cuda.is_available()

    recommended_device = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "recommended_device": recommended_device,
        "notes": accelerator_notes(cuda_available, mps_built, mps_available),
    }
    print(json.dumps(report, indent=2), flush=True)


def accelerator_notes(
    cuda_available: bool,
    mps_built: bool,
    mps_available: bool,
) -> list[str]:
    notes = []
    if cuda_available:
        notes.append("CUDA is available; use ROUTED_VLM_DEVICE=cuda.")
        notes.append("For constrained GPUs, set ROUTED_VLM_LOAD_IN_4BIT=1.")
        return notes

    if mps_available:
        notes.append("Apple MPS is available; use ROUTED_VLM_DEVICE=mps.")
        notes.append("Qwen2.5-VL-7B generation can still be slow on local MPS.")
        notes.append("Use --load-only for local smoke checks.")
        return notes

    if mps_built:
        notes.append("PyTorch was built with MPS, but MPS is not available at runtime.")
        notes.append("Recreate the environment with the official macOS PyTorch wheels.")
    else:
        notes.append("PyTorch was not built with MPS support.")
    notes.append("The service will fall back to CPU, which is too slow for 7B generation.")
    return notes


if __name__ == "__main__":
    main()
