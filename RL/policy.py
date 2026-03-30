import gymnasium as gym
import math

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
from torch import nn
from torch import Tensor


"""
详细算法见 desc.md
"""


class HyperparameterError(RuntimeError):
    """
    RL模型超参数不合理时抛出的错误
    """


class PositionalEncoding(nn.Module):
    """
    位置编码
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Args:
        - d_model: $d$
        - max_len: 位置编码最长长度, 不低于 L 即可
        """
        super().__init__()

        if d_model < 0 or d_model % 2:
            raise HyperparameterError(f"d_model 必须是正偶数, 当前为: {d_model}")

        position = torch.arange(max_len).unsqueeze(1)
        trig = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * trig)
        pe[:, 1::2] = torch.cos(position * trig)

        self.register_buffer('_pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x形状: (L, d) 或 (B, L, d)
        """
        if x.dim() == 2:
            seq_len = x.size(0)
        elif x.dim() == 3:
            seq_len = x.size(1)
        else:
            raise ValueError("位置编码输入 tensor 必须为2维或3维")
        if seq_len > self._pe.size(0):
            raise HyperparameterError(f"输入 tensor 的表达式列长度为：{seq_len}, 但是位置编码长度上限为 {self._pe.size(0)}")

        return x + self._pe[:seq_len,]  
    

class LSTMSharedNet(BaseFeaturesExtractor):
    """
    LSTM编码器
    """

    def __init__(
        self,
        observation_space: gym.Space,
        d_model: int,
        n_layers: int,
        dropout: float,
        device: torch.device,
    ):
        """
        Args:
        - observation_space: 包含 {0,1,...,K} 的观测空间
        - d_model: $d$
        - n_layers: $M$

        observation_space 记录了当前观测的 action + 1, 而 0 通过 `padding_idx` 指定为预留空位
        Embedding 时分配 $K+2$ 个位置，在 Embedding 前 $K+1$ 个位置映射规则与 observation_space 相同
        规定 [BEGIN] 对应的 Embedding 离散值为 $K+1$ 
        """
        super().__init__(observation_space, d_model)

        self._device = device
        self._d_model = d_model

        self._n_observations = observation_space.high[0] + 1 # K+1

        self._token_embedding = nn.Embedding(self._n_observations + 1, d_model, padding_idx=0)
        self._position_encoding = PositionalEncoding(d_model)

        self._lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        
    def forward(self, obs: Tensor) -> Tensor:   
        """
        输入的是当前 batch 的所有 $\tilde s_t$ 沿左新轴拼成的 tensor

        obs 形状: (B, L-1)
        """

        batchsize = obs.shape[0]

        begin = torch.full(
            (batchsize, 1),
            fill_value=self._n_observations,
            dtype=torch.long,
            device=obs.device
        )

        obs = torch.cat([begin, obs.long()], dim=1) # (B, L)

        maxseqlen = int((obs != 0).sum(1).max().item())

        source = self._position_encoding(self._token_embedding(obs))[:,:maxseqlen] # (B, L, d)
        lstmtop = self._lstm(source)[0]

        lstmpool = lstmtop.mean(dim=1)

        return lstmpool
    

class TransformerSharedNet(BaseFeaturesExtractor):
    pass

