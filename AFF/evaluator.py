from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from expressions.expression import Expression

from .predictor import AlphaForgeNetP, PredictorFitResult, fit_predictor
from .tokenizer import AlphaForgeTokenizer


@dataclass
class EvaluatorOutput:
    scores: torch.Tensor
    latent: torch.Tensor | None = None


class AlphaForgeExpressionEvaluator:
    """
    AlphaForge-style evaluator adapted to the current AlphaMiner expression system.

    Workflow:
    1. Convert expressions into fixed-length postfix action ids
    2. One-hot encode action ids
    3. Predict expression quality with a CNN regressor
    """

    def __init__(
        self,
        *,
        max_len: int = 20,
        hidden: int = 128,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.tokenizer = AlphaForgeTokenizer(max_len=max_len)
        self.model = AlphaForgeNetP(
            n_chars=self.tokenizer.n_actions,
            seq_len=max_len,
            hidden=hidden,
        ).to(self.device)
        self.model.initialize_parameters()

    @property
    def max_len(self) -> int:
        return self.tokenizer.max_len

    @property
    def n_actions(self) -> int:
        return self.tokenizer.n_actions

    def encode(self, exprs: Sequence[Expression]) -> torch.Tensor:
        return self.tokenizer.expressions_to_onehot(exprs, device=self.device)

    def fit(
        self,
        exprs: Sequence[Expression],
        scores: Sequence[float],
        *,
        sample_weight: Sequence[float] | None = None,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 100,
        patience: int = 10,
    ) -> PredictorFitResult:
        x = self.encode(exprs).detach().cpu()
        y = torch.tensor(scores, dtype=torch.float32).view(-1, 1)
        w = None if sample_weight is None else torch.tensor(sample_weight, dtype=torch.float32).view(-1, 1)
        return fit_predictor(
            self.model,
            x,
            y,
            sample_weight=w,
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            device=self.device,
        )

    def predict(self, exprs: Sequence[Expression], return_latent: bool = False) -> EvaluatorOutput:
        x = self.encode(exprs)
        self.model.eval()
        with torch.no_grad():
            if return_latent:
                scores, latent = self.model(x, return_latent=True)
                return EvaluatorOutput(scores=scores.squeeze(-1), latent=latent)
            scores = self.model(x)
        return EvaluatorOutput(scores=scores.squeeze(-1), latent=None)

