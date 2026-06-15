from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TemperatureFitResult:
    temperature: float
    log_temperature: float
    nll_before: float
    nll_after: float
    n_samples: int


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply positive scalar temperature to logits."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return logits.detach().float() / float(temperature)


def fit_temperature_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    init_temperature: float = 1.0,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    max_iter: int = 100,
    lr: float = 0.05,
) -> TemperatureFitResult:
    """Fit scalar temperature by minimizing NLL on labeled logits."""
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()

    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [N, C], got {tuple(logits.shape)}")
    if labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("labels must be one-dimensional and match logits length")

    log_temperature = torch.tensor(
        [math.log(float(init_temperature))],
        dtype=torch.float32,
        requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min_temperature, max_temperature)
        loss = F.cross_entropy(logits / temperature, labels, reduction="mean")
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        temperature = float(torch.exp(log_temperature).item())
        temperature = float(max(min_temperature, min(max_temperature, temperature)))
        nll_before = float(F.cross_entropy(logits, labels, reduction="mean").item())
        nll_after = float(F.cross_entropy(logits / temperature, labels, reduction="mean").item())

    return TemperatureFitResult(
        temperature=temperature,
        log_temperature=float(math.log(temperature)),
        nll_before=nll_before,
        nll_after=nll_after,
        n_samples=int(labels.numel()),
    )
