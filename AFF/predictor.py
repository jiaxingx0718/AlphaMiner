from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AlphaForgeNetP(nn.Module):
    """
    AlphaForge-style predictor.

    The original repository uses a small CNN on one-hot token sequences to regress
    factor quality. We keep that idea, but compute the flatten dimension dynamically
    so the module remains usable when max_len changes.
    """

    def __init__(self, n_chars: int, seq_len: int, hidden: int = 128) -> None:
        super().__init__()
        self.n_chars = int(n_chars)
        self.seq_len = int(seq_len)
        self.hidden = int(hidden)

        self.convs = nn.Sequential(
            nn.Conv2d(self.n_chars, 96, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(96, self.hidden, kernel_size=(1, 4)),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.n_chars, 1, self.seq_len)
            conv_out = self.convs(dummy)
            self.flat_dim = conv_out.reshape(1, -1).shape[-1]

        self.fc1 = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        """
        Input shape:
            (batch, seq_len, n_chars)
        """
        x = x.float().permute(0, 2, 1)[:, :, None]
        x = self.convs(x)
        x = x.reshape(x.shape[0], self.flat_dim)
        latent = self.fc1(x)
        out = self.fc2(latent)
        if return_latent:
            return out, latent
        return out

    def initialize_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim > 1:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)


@dataclass
class PredictorFitResult:
    best_valid_loss: float
    epochs_run: int


def fit_predictor(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    *,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_epochs: int = 100,
    patience: int = 10,
    valid_size: float = 0.2,
    device: str | torch.device = "cpu",
) -> PredictorFitResult:
    """
    Train the AlphaForge-style predictor with optional sample weights.
    """
    device = torch.device(device)
    n_samples = x.shape[0]
    valid_count = max(1, int(n_samples * valid_size))
    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_samples, generator=generator)
    valid_idx = perm[:valid_count]
    train_idx = perm[valid_count:]
    if train_idx.numel() == 0:
        train_idx = valid_idx[:1]

    x_train, x_valid = x[train_idx], x[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    if sample_weight is None:
        w_train = torch.ones_like(y_train)
        w_valid = torch.ones_like(y_valid)
    else:
        w_train, w_valid = sample_weight[train_idx], sample_weight[valid_idx]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train, w_train),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(x_valid, y_valid, w_valid),
        batch_size=batch_size,
        shuffle=False,
    )

    def weighted_mse_loss(pred, target, weight):
        return (((pred - target) ** 2) * weight.expand_as(pred)).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_valid_loss = float("inf")
    best_weights = None
    patience_counter = 0
    epochs_run = 0

    for epoch in range(num_epochs):
        epochs_run = epoch + 1
        model.train()
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = weighted_mse_loss(pred, batch_y, batch_w)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_losses = []
            for batch_x, batch_y, batch_w in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)
                pred = model(batch_x)
                valid_losses.append(weighted_mse_loss(pred, batch_y, batch_w).item())
            valid_loss = float(sum(valid_losses) / max(len(valid_losses), 1))

        if valid_loss < best_valid_loss - 1e-6:
            best_valid_loss = valid_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return PredictorFitResult(best_valid_loss=best_valid_loss, epochs_run=epochs_run)
