from __future__ import annotations

from enum import IntEnum
from typing import Type

from data.datatotensor import FeatureType
from expressions.expression import Expression


"""
所有模型输出最终都会翻译为由本文件定义的 Token 类组成的序列, 然后基于 tree.py 还原为 Expression
这一层只定义 token 语义, 不负责 token 与具体模型动作空间的映射
"""


class Token:
    """
    所有表达式构造 token 的公共父类
    """

    def __repr__(self) -> str:
        return str(self)


class FeatureToken(Token):
    """
    基础特征 token, 对应 Feature
    """

    def __init__(self, feature: FeatureType) -> None:
        self.feature = feature

    def __str__(self) -> str:
        return self.feature.name


class ConstantToken(Token):
    """
    数值常量 token, 对应 Constant
    """

    def __init__(self, constant: float) -> None:
        self.constant = float(constant)

    def __str__(self) -> str:
        return str(self.constant)



class WindowToken(Token):
    """
    窗口参数 token, 对应当前表达式系统中的:
    - RollingOperator 的 window
    - PastOperator 的 npast
    - FORWARDRET 的 window

    它和 ConstantToken 在表达式形式上相同, 但是属于不同含义
    """

    def __init__(self, window: int) -> None:
        self.window = int(window)

    def __str__(self) -> str:
        return str(self.window)


class UnaryOperatorToken(Token):
    """
    单目算子 token
    """
    def __init__(self, operator: Type[Expression]) -> None:
        self.operator = operator

    def __str__(self) -> str:
        return self.operator.__name__


class BinaryOperatorToken(Token):
    """
    双目算子 token
    """
    def __init__(self, operator: Type[Expression]) -> None:
        self.operator = operator

    def __str__(self) -> str:
        return self.operator.__name__


class RollingOperatorToken(Token):
    """
    需要一个 Expression 和一个 WindowToken 的算子 token
    用于 RollingOperator/PastOperator/FORWARDRET 这类表达式
    """
    def __init__(self, operator: Type[Expression]) -> None:
        self.operator = operator

    def __str__(self) -> str:
        return self.operator.__name__


class SequenceIndicatorType(IntEnum):
    """
    表达式序列控制标记, 包括:
    - BEGIN: 序列开始标记, 主要给 RL / 序列模型使用
    - END: 表达式结束标记, 相当于 stop action
    """
    BEGIN = 0
    END = 1


class SequenceIndicatorToken(Token):
    """
    序列控制 token
    """

    def __init__(self, indicator: SequenceIndicatorType) -> None:
        self.indicator = indicator

    def __str__(self) -> str:
        return self.indicator.name


BEGIN_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.BEGIN)
END_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.END)


class ExpressionToken(Token):
    """
    对完整 Expression 的包装 token, 通常不会在原始模型输出序列中直接遇到
    主要用于子表达式复用或将现成表达式直接送入构树器
    """

    def __init__(self, expression: Expression) -> None:
        self.expression = expression

    def __str__(self) -> str:
        return str(self.expression)
