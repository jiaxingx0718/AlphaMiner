from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from data.datatotensor import FeatureType
from expressions.expression import (
    OPEN, CLOSE, HIGH, LOW, VOLUME, AMOUNT,
    ABS, SIGN, LOG, EXP, CSRANK,
    ADD, SUB, MUL, DIV, POW, BINMAX, BINMIN,
    SUM, STD, SKEW, KURT, MAX, MIN, MED, RANK, MA, WMA, EMA,
    PAST, DELTA
)

from expressions.tokens import (
    BinaryOperatorToken,
    ConstantToken,
    END_TOKEN,
    FeatureToken,
    RollingOperatorToken,
    SequenceIndicatorToken,
    UnaryOperatorToken,
    WindowToken,
    Token,
)

from RL.evalcore import AlphaEnvCore


"""
此模块负责将RL模型的离散动作空间 action (一般是一个整数) 与 token 对应, 以对接RL模型
具体的底层的表达式构建评估系统由 core.py 决定, 此模块将其包装
"""


UNARY_ACTION_TOKENS: list[Token] = [
    UnaryOperatorToken(ABS),
    UnaryOperatorToken(SIGN),
    UnaryOperatorToken(LOG),
    UnaryOperatorToken(EXP),
    UnaryOperatorToken(CSRANK),
]

BINARY_ACTION_TOKENS: list[Token] = [
    BinaryOperatorToken(ADD),
    BinaryOperatorToken(SUB),
    BinaryOperatorToken(MUL),
    BinaryOperatorToken(DIV),
    BinaryOperatorToken(POW),
    BinaryOperatorToken(BINMAX),
    BinaryOperatorToken(BINMIN),
]

ROLLING_ACTION_TOKENS: list[Token] = [
    RollingOperatorToken(SUM),
    RollingOperatorToken(STD),
    RollingOperatorToken(SKEW),
    RollingOperatorToken(KURT),
    RollingOperatorToken(MAX),
    RollingOperatorToken(MIN),
    RollingOperatorToken(MED),
    RollingOperatorToken(RANK),
    RollingOperatorToken(MA),
    RollingOperatorToken(WMA),
    RollingOperatorToken(EMA),
    RollingOperatorToken(PAST),
    RollingOperatorToken(DELTA),
]

FEATURE_ACTION_TOKENS: list[Token] = [
    FeatureToken(FeatureType.OPEN),
    FeatureToken(FeatureType.CLOSE),
    FeatureToken(FeatureType.HIGH),
    FeatureToken(FeatureType.LOW),
    FeatureToken(FeatureType.VOLUME),
    FeatureToken(FeatureType.AMOUNT),
]

CONSTANT_ACTION_TOKENS: list[Token] = [
    ConstantToken(0.0),
    ConstantToken(-1.0),
    ConstantToken(1.0),
    ConstantToken(5.0),
    ConstantToken(-5.0),
]

WINDOW_ACTION_TOKENS: list[Token] = [
    WindowToken(5),
    WindowToken(10),
    WindowToken(20),
]

SEQUENCE_ACTION_TOKENS: list[Token] = [END_TOKEN]

ACTION_TOKENS: list[Token] = [
    *UNARY_ACTION_TOKENS,
    *BINARY_ACTION_TOKENS,
    *ROLLING_ACTION_TOKENS,
    *FEATURE_ACTION_TOKENS,
    *CONSTANT_ACTION_TOKENS,
    *WINDOW_ACTION_TOKENS,
    *SEQUENCE_ACTION_TOKENS,
]


class AlphaEnvWrapper(gym.Wrapper):
    """
    AlphaEnvCore 针对RL模型的封装版本, 主要维护: 
    - env: AlphaEnvCore, 所有评估参数由 env 提供
    - _action_tokens: 所有允许生成的 Token 序列
      - 当模型输出整数 action 时, 相当于生成 Token _action_tokens[action]
    - action_space: 允许生成的整数离散动作范围
    - observation_space: 观测到的整数离散动作范围
      - 这里的映射规则为 observation = action + 1, 0表示空位
    - state: 填充空位的当前 observation 序列
      - 形式类似于 [27, 26, 6, 0, 0, 0, ...]
    - counter: 当前写入 action 的计数器

    主要由 step() 接收 action 并给出反馈
    """

    def __init__(self, env: AlphaEnvCore):

        super().__init__(env)
        
        self.env: AlphaEnvCore = env

        self._action_tokens: list[Token] = list(ACTION_TOKENS)

        self._size_action_space = len(self._action_tokens)
        self.action_space = gym.spaces.Discrete(self._size_action_space)
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=self._size_action_space,
            shape=(self.env.config.max_expr_length - 1,),
            dtype=np.int32,
        )

        self.state = np.zeros(self.env.config.max_expr_length - 1, dtype=np.int32)

        self.counter = 0


    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        重置维护的参数
        """
        self.state = np.zeros(self.env.config.max_expr_length - 1, dtype=np.int32)
        self.counter = 0
        _, info = self.env.reset(seed=seed)

        info = dict(info)
        info["action_mask"] = self.action_mask()
        info["tokens"] = [str(tok) for tok in self.env.tokens]

        return self.state.copy(), info


    def step(self, action: int):
        """
        接收一个 action 并执行:
        - 将其映射为 token
        - 调用 self.env.step() 将此 token 交给 AlphaEnvCore 以评估
        - 将 action + 1 写入 state
    
        返回一个元组 (主要基于 gym 统一格式), 包括: 
        - 更新后的 state
        - 传递 AlphaEnvCore 的 reward, terminated, truncated
        - info, 主要包含
          - 'action_mask': 将 self.env.valid_action_types() 展开为 action 布尔掩码
          - 'tokens': 当前已经生成的 token 序列
        """
        token = self.action_to_token(action)

        _, reward, terminated, truncated, info = self.env.step(token)

        if self.counter < len(self.state):
            self.state[self.counter] = action + 1
            self.counter += 1

        info = dict(info)
        info["action_mask"] = self.action_mask()
        info["tokens"] = [str(tok) for tok in self.env.tokens]

        return self.state.copy(), reward, terminated, truncated, info


    def action_to_token(self, action: int) -> Token:
        if action < 0 or action >= self._size_action_space:
            raise ValueError(f"action 候选空间 {self.action_space}, 当前 action : {action}")
        return self._action_tokens[action]


    def action_mask(self) -> np.ndarray:

        valid = self.env.valid_action_types()
        mask = np.zeros(self._size_action_space, dtype=bool)

        for i, token in enumerate(self._action_tokens):
            if isinstance(token, UnaryOperatorToken):
                mask[i] = bool(valid["op"]["unary"])
            elif isinstance(token, BinaryOperatorToken):
                mask[i] = bool(valid["op"]["binary"])
            elif isinstance(token, RollingOperatorToken):
                mask[i] = bool(valid["op"]["rolling"])
            elif isinstance(token, FeatureToken):
                mask[i] = bool(valid["select"]["feature"])
            elif isinstance(token, ConstantToken):
                mask[i] = bool(valid["select"]["constant"])
            elif isinstance(token, WindowToken):
                mask[i] = bool(valid["select"]["window"])
            elif isinstance(token, SequenceIndicatorToken):
                mask[i] = bool(valid["select"]["stop"])
            else:
                mask[i] = False

        return mask
     

    @property
    def size_action(self) -> int:
        return self._size_action_space
    
    @property
    def action_tokens(self) -> tuple[Token]:
        return tuple(self._action_tokens)


__all__ = [
    "AlphaEnvWrapper",
    "ACTION_TOKENS",
    "UNARY_ACTION_TOKENS",
    "BINARY_ACTION_TOKENS",
    "ROLLING_ACTION_TOKENS",
    "FEATURE_ACTION_TOKENS",
    "CONSTANT_ACTION_TOKENS",
    "WINDOW_ACTION_TOKENS",
    "SEQUENCE_ACTION_TOKENS",
]
