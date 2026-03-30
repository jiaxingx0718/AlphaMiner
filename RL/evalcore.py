from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import gymnasium as gym
import math

from calculator.calculator import InvalidEvaluateError, StockDataCalculator
from expressions.expression import Expression
from expressions.tokens import BEGIN_TOKEN, END_TOKEN, SequenceIndicatorToken, SequenceIndicatorType, Token
from expressions.tree import ExpressionBuilder, InvalidTokenError


"""
RL类模型的表达式构造和反馈的核心模块, 接收一个 token 并给予一系列反馈信息
不关心RL模型具体接口, 由 wrapper.py 负责形式上的映射
"""


class CoreEvaluateError(RuntimeError):
    """
    评估表达式失败时抛出的异常
    """


@dataclass
class RLCoreConfig:
    max_expr_length: int = 20 # 输出表达式去除BEGIN长度上限
    reward_per_step: float = 0.0 # 接收 token 时的分数
    invalid_reward: float = -1.0 # 接收 token 无效，或结束时表达式仍然不合法的分数
    invalid_eval_reward: float = 0.0 # 表达式合法但是评估阶段无效的分数
    print_expr: bool = False # 是否输出表达式
    eval_metric: str = "ic" # 评价标准


class AlphaEnvCore(gym.Env):
    """
    RL类模型生成表达式评估和反馈的核心, 主要维护:
    - calculator: StockDataCalculator
    - _builder: ExpressionBuilder 
    - _tokens: 序列
    - _done: bool
    - eval_cnt: 评估最终表达式数量的计数器

    主要由 step() 接收 token 并给出反馈
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        calculator: StockDataCalculator,
        max_expr_length: int = 20,
        step_reward: float = 0.0,
        invalid_reward: float = -1.0,
        invalid_eval_reward: float = 0.0,
        print_expr: bool = False,
        eval_metric: str = "ic",
    ) -> None:
        
        super().__init__()

        self.calculator = calculator
        self.config = RLCoreConfig(
            max_expr_length=max_expr_length,
            reward_per_step=step_reward,
            invalid_reward=invalid_reward,
            invalid_eval_reward=invalid_eval_reward,
            print_expr=print_expr,
            eval_metric=eval_metric.lower(),
        )

        if self.config.eval_metric not in {"ic", "rankic"}:
            raise ValueError("评估标准仅支持 'ic' 或 'rankic'")

        self.eval_cnt = 0
        self._tokens: list[Token] = []
        self._builder = ExpressionBuilder()
        self._done = False

        self.reset()


    def reset(self,seed: Optional[int] = None) -> tuple[list[Token], dict[str, Any]]:
        """
        重置维护的参数
        """
        super().reset(seed=seed)
        self._tokens = [BEGIN_TOKEN]
        self._builder = ExpressionBuilder()
        self._done = False
        return self.tokens, self.valid_action_types()


    def step(self, action: Token) -> tuple[list[Token], float, bool, bool, dict[str, Any]]:
        """
        接收一个 token 并执行:
        - 检查 _done 确认当前是否结束
        - 如果接收 BEGIN, 给基础 reward (通常是0)
        - 如果接收 END, 则加入 token 序列, 计算 reward, 并标记 _done=True
        - 如果接收一般 token, 则尝试通过 _builder 构造树
          - 可以构造: 将 token 加入 _tokens 并加入 _builder 构造, 给基础 reward, 更新 _tokens
          - 不可以构造: 标记 _done=True, 给惩罚 reward, 将错误信息放进 info
        - 如果当前 _tokens 序列长度超过上限, 调用 ExpressionBuilder.validate() 判断当前是否能直接合法结束
          - 可以结束: 计算 reward
          - 不可以结束: 给 self.config.invalid_reward


        返回一个元组 (主要基于 gym 统一格式), 包括:
        - tokens: 当前轮到目前为止的 token 序列
        - reward: 这一步的奖励分数, 一般中间步骤返回 0.0, 结束时返回 calculator 的评估分数, 非法情况返回 invalid_reward
        - terminated: 当前轮是否结束
        - truncated: False (设置人工断点)
        - self.valid_action_types(): 下一步允许的状态候选空间
        """

        if self._done:
            raise RuntimeError("当前 episode 已结束, 请先 reset()")

        info: dict[str, Any] = {}
        truncated = False

        if isinstance(action, SequenceIndicatorToken) and action.indicator == SequenceIndicatorType.BEGIN:
            reward = self.config.reward_per_step
            return self.tokens, reward, False, truncated, self.valid_action_types()

        if isinstance(action, SequenceIndicatorToken) and action.indicator == SequenceIndicatorType.END:
            self._tokens.append(action)
            reward = self._evaluate_current()
            self._done = True
            if math.isnan(reward):
                reward = 0.0
            return self.tokens, reward, True, truncated, self.valid_action_types()

        try:
            self._builder.add_token(action)
            self._tokens.append(action)
            reward = self.config.reward_per_step
            terminated = False
        except InvalidTokenError as exc:
            self._tokens.append(action)
            self._done = True
            reward = self.config.invalid_reward
            terminated = True
            info["error"] = str(exc)

        if len(self._tokens) > self.config.max_expr_length:
            terminated = True
            self._done = True
            if self._builder.validate(END_TOKEN):
                reward = self._evaluate_current()
            else:
                reward = self.config.invalid_reward

        if math.isnan(reward):
            reward = 0.0

        return self.tokens, reward, terminated, truncated, self.valid_action_types()


    def _evaluate_current(self) -> float:
        """
        评估当前表达式
        """
        if not self._builder.validate(END_TOKEN):
            return self.config.invalid_reward

        try:
            expr = self._builder.get_tree()
            if self.config.print_expr:
                print(expr)
            reward = self.evaluate_expr(expr)
            self.eval_cnt += 1
            return reward
        except (InvalidTokenError, InvalidEvaluateError, CoreEvaluateError):
            return self.config.invalid_eval_reward


    def evaluate_expr(self, expr: Expression) -> float:

        if self.config.eval_metric == "ic":
            return self.calculator.calc_single_IC(expr)
        if self.config.eval_metric == "rankic":
            return self.calculator.calc_single_rankIC(expr)
        raise CoreEvaluateError(f"不支持的评估指标: {self.config.eval_metric}")


    def valid_action_types(self) -> dict[str, Any]:
        """
        以字典形式给出下一步 token 的候选空间
        """
        valid_unary = self._builder.validate_unaryop(tokenless=True)
        valid_binary = self._builder.validate_binaryop(tokenless=True)
        valid_rolling = self._builder.validate_rollingop(tokenless=True)
        valid_feature = self._builder.validate_feature()
        valid_constant = self._builder.validate_constant()
        valid_window = self._builder.validate_window()
        valid_stop = self._builder.validate(END_TOKEN)

        return {
            "select": {
                "op": valid_unary or valid_binary or valid_rolling,
                "feature": valid_feature,
                "constant": valid_constant,
                "window": valid_window,
                "stop": valid_stop,
            },
            "op": {
                "unary": valid_unary,
                "binary": valid_binary,
                "rolling": valid_rolling,
            },
        }


    @property
    def tokens(self) -> list[Token]:
         return list(self._tokens)

    @property
    def done(self) -> bool:
        return self._done


    def render(self) -> None: ...



__all__ = [
    "AlphaEnvCore",
    "RLCoreConfig",
    "CoreEvaluateError",
]
