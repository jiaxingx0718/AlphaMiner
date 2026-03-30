from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from expressions.tokens import END_TOKEN
from expressions.tree import ExpressionBuilder
from RL.wrapper import ACTION_TOKENS

from .tokenizer import AlphaForgeTokenizer


@dataclass
class GeneratorSample:
    action_ids: torch.Tensor
    logits: torch.Tensor


@dataclass
class GeneratorForward:
    action_ids: torch.Tensor
    logits: torch.Tensor
    masked_logits: torch.Tensor
    masks: torch.Tensor


class AlphaForgeGeneratorLSTM(nn.Module):
    """
    A lightweight latent-conditioned sequence generator inspired by AlphaForge.

    This module is intentionally modest:
    - it reuses the current AlphaMiner action space
    - it applies builder-based legality masks during sampling
    - it is meant as a compatible starting point, not a full reproduction
    """

    def __init__(
        self,
        *,
        latent_size: int,
        d_model: int,
        n_layers: int,
        dropout: float,
        max_len: int = 20,
    ) -> None:
        super().__init__()
        self.tokenizer = AlphaForgeTokenizer(max_len=max_len)
        self.max_len = max_len
        self.n_actions = self.tokenizer.n_actions
        self.bos_id = self.n_actions
        self.latent_size = latent_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.fc_h = nn.Sequential(nn.Linear(latent_size, n_layers * d_model), nn.ReLU())
        self.fc_c = nn.Sequential(nn.Linear(latent_size, n_layers * d_model), nn.ReLU())
        self.emb = nn.Embedding(self.n_actions + 1, d_model, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(d_model, self.n_actions)

    def initialize_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim > 1:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def _builder_action_mask(self, builder: ExpressionBuilder) -> torch.Tensor:
        mask = torch.zeros(self.n_actions, dtype=torch.bool)
        for idx, token in enumerate(ACTION_TOKENS):
            try:
                mask[idx] = builder.validate(token)
            except Exception:
                mask[idx] = False
        return mask

    def sample(
        self,
        z: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> GeneratorSample:
        device = z.device
        bs = z.shape[0]

        h = self.fc_h(z).view(bs, self.n_layers, self.d_model).permute(1, 0, 2).contiguous()
        c = self.fc_c(z).view(bs, self.n_layers, self.d_model).permute(1, 0, 2).contiguous()

        input_step = torch.full((bs,), fill_value=self.bos_id, dtype=torch.long, device=device)
        builders = [ExpressionBuilder() for _ in range(bs)]
        done = torch.zeros(bs, dtype=torch.bool, device=device)
        action_ids = torch.full(
            (bs, self.max_len),
            fill_value=self.tokenizer.end_action_id,
            dtype=torch.long,
            device=device,
        )
        logits_out = torch.zeros(bs, self.max_len, self.n_actions, device=device)

        for t in range(self.max_len):
            embedded = self.emb(input_step)[:, None]
            output, (h, c) = self.rnn(embedded, (h, c))
            logits = self.fc(output).squeeze(1)
            logits_out[:, t] = logits

            next_actions = []
            for i in range(bs):
                if done[i]:
                    next_actions.append(self.tokenizer.end_action_id)
                    continue

                mask = self._builder_action_mask(builders[i]).to(device)
                masked_logits = logits[i].clone()
                masked_logits[~mask] = -1e9

                if deterministic:
                    action = int(masked_logits.argmax().item())
                else:
                    action = int(Categorical(logits=masked_logits).sample().item())

                token = ACTION_TOKENS[action]
                builders[i].add_token(token)
                if token is END_TOKEN:
                    done[i] = True
                next_actions.append(action)

            input_step = torch.tensor(next_actions, dtype=torch.long, device=device)
            action_ids[:, t] = input_step

            if bool(done.all()):
                break

        return GeneratorSample(action_ids=action_ids, logits=logits_out)

    def forward_masked_logits(self, z: torch.Tensor) -> GeneratorForward:
        """
        Autoregressively produce raw logits, legality masks, and masked logits.

        The builder path is updated using greedy detached actions, exactly so we can
        reuse legality masks while still keeping the masked logits differentiable
        with respect to generator parameters.
        """
        device = z.device
        bs = z.shape[0]

        h = self.fc_h(z).view(bs, self.n_layers, self.d_model).permute(1, 0, 2).contiguous()
        c = self.fc_c(z).view(bs, self.n_layers, self.d_model).permute(1, 0, 2).contiguous()

        input_step = torch.full((bs,), fill_value=self.bos_id, dtype=torch.long, device=device)
        builders = [ExpressionBuilder() for _ in range(bs)]
        done = torch.zeros(bs, dtype=torch.bool, device=device)
        action_ids = torch.full((bs, self.max_len), fill_value=self.tokenizer.end_action_id, dtype=torch.long, device=device)
        logits_out = torch.zeros(bs, self.max_len, self.n_actions, device=device)
        masked_logits_out = torch.zeros(bs, self.max_len, self.n_actions, device=device)
        masks_out = torch.zeros(bs, self.max_len, self.n_actions, dtype=torch.bool, device=device)

        for t in range(self.max_len):
            embedded = self.emb(input_step)[:, None]
            output, (h, c) = self.rnn(embedded, (h, c))
            logits = self.fc(output).squeeze(1)
            logits_out[:, t] = logits

            next_actions = []
            for i in range(bs):
                if done[i]:
                    mask = torch.zeros(self.n_actions, dtype=torch.bool, device=device)
                    mask[self.tokenizer.end_action_id] = True
                    masks_out[i, t] = mask
                    masked_logits = logits[i].clone()
                    masked_logits[~mask] = -1e9
                    masked_logits_out[i, t] = masked_logits
                    next_actions.append(self.tokenizer.end_action_id)
                    continue

                mask = self._builder_action_mask(builders[i]).to(device)
                masks_out[i, t] = mask
                masked_logits = logits[i].clone()
                masked_logits[~mask] = -1e9
                masked_logits_out[i, t] = masked_logits

                action = int(masked_logits.argmax().item())

                token = ACTION_TOKENS[action]
                builders[i].add_token(token)
                if token is END_TOKEN:
                    done[i] = True
                next_actions.append(action)

            input_step = torch.tensor(next_actions, dtype=torch.long, device=device)
            action_ids[:, t] = input_step

            if bool(done.all()):
                break

        return GeneratorForward(
            action_ids=action_ids,
            logits=logits_out,
            masked_logits=masked_logits_out,
            masks=masks_out,
        )
