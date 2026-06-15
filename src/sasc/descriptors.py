from __future__ import annotations

import numpy as np
import torch


def _summary_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    arr = values.detach().cpu().numpy().astype(float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=0)),
        f"{prefix}_q10": float(np.quantile(arr, 0.10)),
        f"{prefix}_q25": float(np.quantile(arr, 0.25)),
        f"{prefix}_q50": float(np.quantile(arr, 0.50)),
        f"{prefix}_q75": float(np.quantile(arr, 0.75)),
        f"{prefix}_q90": float(np.quantile(arr, 0.90)),
    }


def stream_descriptors_from_logits(logits: torch.Tensor) -> dict[str, float]:
    """Compute unlabeled stream descriptors from logits only."""
    logits = logits.detach().cpu().float()
    probs = torch.softmax(logits, dim=1)
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)

    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1]
    margin = top1 - top2
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    free_energy = -torch.logsumexp(logits, dim=1)

    out: dict[str, float] = {}
    out.update(_summary_stats("entropy", entropy))
    out.update(_summary_stats("top1_confidence", top1))
    out.update(_summary_stats("margin", margin))
    out.update(_summary_stats("free_energy", free_energy))
    out["n_stream_samples"] = int(logits.shape[0])
    out["n_classes"] = int(logits.shape[1])
    return out
