from __future__ import annotations

import torch
import torch.nn.functional as F


def brier_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = F.one_hot(labels.long(), num_classes=probs.shape[1]).float()
    return float(((probs - one_hot) ** 2).sum(dim=1).mean().item())


def ece_fixed_bins(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels.long()).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        in_bin = (confidence >= lo) & (confidence <= hi) if i == 0 else (confidence > lo) & (confidence <= hi)
        prop = float(in_bin.float().mean().item())
        if prop > 0:
            acc_bin = float(correct[in_bin].mean().item())
            conf_bin = float(confidence[in_bin].mean().item())
            ece += abs(acc_bin - conf_bin) * prop
    return float(ece)


def adaptive_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels.long()).float()
    order = torch.argsort(confidence)
    confidence = confidence[order]
    correct = correct[order]
    n = int(confidence.numel())
    if n == 0:
        return 0.0

    bin_sizes = [n // n_bins] * n_bins
    for i in range(n % n_bins):
        bin_sizes[i] += 1

    out = 0.0
    start = 0
    for size in bin_sizes:
        if size == 0:
            continue
        end = start + size
        out += abs(float(confidence[start:end].mean()) - float(correct[start:end].mean())) * (size / n)
        start = end
    return float(out)


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> dict[str, float]:
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()
    probs = torch.softmax(logits, dim=1)
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels).float()

    accuracy = float(correct.mean().item())
    mean_confidence = float(confidence.mean().item())
    signed_gap = mean_confidence - accuracy

    return {
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "signed_gap": float(signed_gap),
        "abs_signed_gap": float(abs(signed_gap)),
        "nll": float(F.cross_entropy(logits, labels, reduction="mean").item()),
        "brier": brier_from_probs(probs, labels),
        "ece15": ece_fixed_bins(probs, labels, n_bins=n_bins),
        "aece15": adaptive_ece(probs, labels, n_bins=n_bins),
    }
