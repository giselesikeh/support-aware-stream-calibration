from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_logits_labels(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load logits/labels from .pt or .npz files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".pt", ".pth"}:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        logits = blob.get("logits", blob.get("outputs", blob.get("scores")))
        labels = blob.get("labels", blob.get("targets", blob.get("y")))
        if logits is None or labels is None:
            raise KeyError(f"Could not find logits/labels keys in {path}")
        return torch.as_tensor(logits).float(), torch.as_tensor(labels).long()

    if path.suffix == ".npz":
        blob = np.load(path, allow_pickle=True)
        logits_key = "logits" if "logits" in blob.files else "arr_0"
        labels_key = "labels" if "labels" in blob.files else ("y" if "y" in blob.files else "arr_1")
        return torch.tensor(blob[logits_key], dtype=torch.float32), torch.tensor(blob[labels_key], dtype=torch.long)

    raise ValueError(f"Unsupported logits file type: {path}")
