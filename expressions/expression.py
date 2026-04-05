from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Union

import torch
from torch import Tensor

from data.datatotensor import FeatureType, StockData

_ExprOrFloat = Union["Expression", float, int]


class InvalidExpressionError(RuntimeError):
    """
    表达式结构或参数不合理时抛出的异常
    """


class Expression(metaclass=ABCMeta):
    """
    所有表达式节点的抽象基类
    子类必须实现 evaluate 并返回形状为 (T, N) 的 tensor
    """

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def evaluate(self, data: StockData) -> Tensor:
        """
        在给定的 StockData 上递归计算表达式
        输入底层数据形状为 (T, F, N)
        输出必须是形状为 (T, N) 的 tensor
        """


def _into_expr(value: _ExprOrFloat) -> Expression:
    """
    将输入统一转换为 Expression
    主要用于二元算子和一元算子的常数自动包装
    """

    if isinstance(value, Expression):
        return value
    try:
        return Constant(float(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"构造 Constant 接受值必须是浮点数, 当前为: {value}") from exc


class Feature(Expression):
    """
    原始特征节点
    包括:
    - OPEN 
    - CLOSE 
    - HIGH 
    - LOW 
    - VOLUME 
    - AMOUNT
    """

    def __init__(self, feature: FeatureType) -> None:
        self._feature = feature

    def __str__(self) -> str:
        return self._feature.name

    def evaluate(self, data: StockData) -> Tensor:
        """
        直接从底层数据中提取对应特征
        """

        value = data.data[:, int(self._feature), :]
        return value

CLOSE = Feature(FeatureType.CLOSE)
OPEN = Feature(FeatureType.OPEN)
HIGH = Feature(FeatureType.HIGH)
LOW = Feature(FeatureType.LOW)
VOLUME = Feature(FeatureType.VOLUME)
AMOUNT = Feature(FeatureType.AMOUNT)


class Constant(Expression):
    """
    常数节点
    包括:
    - X (常数)
    """

    def __init__(self, constant: float):
        self._constant = constant

    def __str__(self) -> str:
        return str(self._constant)

    def evaluate(self, data: StockData) -> Tensor:
        """
        生成值全为常数的 (T, N) tensor
        """

        value = torch.full(
            size=(data.n_days, data.n_stocks),
            fill_value=self._constant,
            dtype=data.data.dtype,
            device=data.data.device,
        )

        return value


class UnaryOperator(Expression):
    """
    单 Expression 算子基类
    包括:
    - ABS
    - SIGN 
    - LOG 
    - EXP 
    - CSRANK (截面排序分位数)

    算法解读
    先递归计算内部 operand
    再对得到的 (T, N) tensor 做逐元素变换
    """

    def __init__(self, operand: _ExprOrFloat) -> None:
        self._operand = _into_expr(operand)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"

    def evaluate(self, data: StockData) -> Tensor:

        operand_value = self._operand.evaluate(data)
        value = self._apply(operand_value)

        return value

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

class ABS(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.abs()

class SIGN(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.sign()
    
class LOG(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.log()
    
class EXP(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.exp()
    
class CSRANK(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        value = torch.full_like(operand, torch.nan)
        for i in range(operand.shape[0]):
            row = operand[i]
            mask = torch.isnan(row)
            valid = row[~mask]
            if valid.numel() == 0:
                continue
            _, inv, counts = valid.unique(sorted=True, return_inverse=True, return_counts=True)
            cs = counts.cumsum(dim=0).to(valid.dtype)
            cs = torch.cat([torch.zeros(1, dtype=valid.dtype, device=valid.device), cs], dim=0)
            ranks = (cs[:-1]+ cs[1:] - 1) / 2
            value[i, ~mask] = ranks[inv] / valid.numel()
        return value


class BinaryOperator(Expression):
    """
    双 Expression 算子基类
    包括:
    - ADD 
    - SUB 
    - MUL 
    - DIV 
    - POW 
    - BINMAX 
    - BINMIN

    先递归计算左右两个 operand
    再对两个 (T, N) tensor 做逐元素组合
    """

    def __init__(self, L_operand: _ExprOrFloat, R_operand: _ExprOrFloat) -> None:
        self._L_operand = _into_expr(L_operand)
        self._R_operand = _into_expr(R_operand)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._L_operand}, {self._R_operand})"

    def evaluate(self, data: StockData) -> Tensor:

        L_operand_value = self._L_operand.evaluate(data)
        R_operand_value = self._R_operand.evaluate(data)
        value = self._apply(L_operand_value, R_operand_value)

        return value
    
    @abstractmethod
    def _apply(self, L_operand: Tensor, R_operand: Tensor) -> Tensor: ...

class ADD(BinaryOperator):
    def _apply(self, L_operand, R_operand) -> Tensor:
        return L_operand + R_operand
    
class SUB(BinaryOperator):
    def _apply(self, L_operand, R_operand) -> Tensor:
        return L_operand - R_operand

class MUL(BinaryOperator):
    def _apply(self, L_operand, R_operand) -> Tensor:
        return L_operand * R_operand

class DIV(BinaryOperator):
    def _apply(self, L_operand, R_operand) -> Tensor:
        return L_operand / R_operand

class POW(BinaryOperator):
    def _apply(self, L_operand, R_operand) -> Tensor:
        return L_operand ** R_operand

class BINMAX(BinaryOperator):
    def _apply(self, L_operand, R_operand) -> Tensor:
        return L_operand.max(R_operand)

class BINMIN(BinaryOperator):
    def _apply(self, L_operand, R_operand) -> Tensor:
        return L_operand.min(R_operand)


class RollingOperator(Expression):
    """
    滚动算子基类
    对包含 t 日在内的过去 window 长度窗口做聚合
    包括:
    - SUM
    - STD 
    - SKEW 
    - KURT
    - MAX 
    - MIN 
    - MED 
    - RANK 
    - MA 
    - WMA 
    - EMA 

    先递归计算内部 operand
    再沿时间轴生成长度为 window 的滑动窗口
    前面不足 window 的位置补 torch.nan
    """

    def __init__(self, operand: _ExprOrFloat, window: int) -> None:
        """
        window: 滚动窗口长度
        """
        if window < 1:
            raise ValueError(f"窗口长度应当为正整数, 当前为: {window}")
        self._operand = _into_expr(operand)
        self._window = int(window)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand}, {self._window})"

    def evaluate(self, data: StockData) -> Tensor:

        operand_value = self._operand.evaluate(data)
        if self._window == 1:
            return self._apply_window_one(operand_value)

        t, n = operand_value.shape

        if t < self._window:
            return torch.full(
                size=(t, n),
                fill_value=torch.nan,
                dtype=operand_value.dtype,
                device=operand_value.device,
            )
        window_values = operand_value.unfold(0, self._window, 1)
        valid_value = self._apply(window_values)
        prefix_value = torch.full(
            size=(self._window - 1, n),
            fill_value=torch.nan,
            dtype=operand_value.dtype,
            device=operand_value.device,
        )
        value = torch.cat([prefix_value, valid_value], dim=0)

        return value

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor:
        """
        对形状为 (T-window+1, N, window) 的 tensor
        沿最后一维做聚合
        """

    @abstractmethod
    def _apply_window_one(self, operand: Tensor) -> Tensor:
        """
        window 等于 1 时的特殊处理
        以下子类返回全 nan
        - STD
        - SKEW
        - KURT
        - RANK
        - WMA
        - EMA
        其余子类直接返回输入本身
        """

class SUM(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.sum(dim=-1)
    def _apply_window_one(self, operand):
        return operand

class STD(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.std(dim=-1)
    def _apply_window_one(self, operand):
        return torch.full(operand.shape, torch.nan, dtype=operand.dtype, device=operand.device)
    
class SKEW(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return (central ** 3).mean(dim=-1) / ((central ** 2).mean(dim=-1)) ** 1.5
    def _apply_window_one(self, operand):
        return torch.full(operand.shape, torch.nan, dtype=operand.dtype, device=operand.device)
    
class KURT(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return (central ** 4).mean(dim=-1) / (operand.var(dim=-1)) ** 2 - 3.0
    def _apply_window_one(self, operand):
        return torch.full(operand.shape, torch.nan, dtype=operand.dtype, device=operand.device)

class MAX(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.max(dim=-1)[0]
    def _apply_window_one(self, operand):
        return operand
    
class MIN(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.min(dim=-1)[0]
    def _apply_window_one(self, operand):
        return operand

class MED(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.median(dim=-1)[0]
    def _apply_window_one(self, operand):
        return operand

class RANK(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        w = operand.shape[-1]
        last = operand[:, :, -1, None].clone()
        left = (last < operand).count_nonzero(dim=-1)
        lefteq = (last <= operand).count_nonzero(dim=-1)
        return (left + lefteq + (lefteq > left)) / 2 / w
    def _apply_window_one(self, operand):
        return torch.full(operand.shape, torch.nan, dtype=operand.dtype, device=operand.device)

class MA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    def _apply_window_one(self, operand):
        return operand

class WMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        w = operand.shape[-1]
        weight = torch.arange(
            w,
            dtype=operand.dtype,
            device=operand.device,
        )
        normalized_weight = weight / weight.sum()
        return (normalized_weight * operand).sum(dim=-1)
    def _apply_window_one(self, operand):
        return torch.full(operand.shape, torch.nan, dtype=operand.dtype, device=operand.device)

class EMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        w = operand.shape[-1]
        alpha = 1-2/(w+1)
        weight = alpha ** torch.arange(
            w, 0, -1,
            dtype=operand.dtype,
            device=operand.device,
        )
        normalized_weight = weight / weight.sum()
        return (normalized_weight * operand).sum(dim=-1)
    def _apply_window_one(self, operand):
        return torch.full(operand.shape, torch.nan, dtype=operand.dtype, device=operand.device)


class PastOperator(Expression):
    """
    过去回看算子基类
    只比较 t 日和 t-window 日
    包括:
    - PAST
    - DELTA

    实现形式和 rolling 很像
    但只关心窗口首尾两个时间点
    """

    def __init__(self, operand: _ExprOrFloat, window: int) -> None:
        if window < 1:
            raise ValueError(f"窗口长度应当为正整数, 当前为: {window}")
        self._operand = _into_expr(operand)
        self._window = int(window)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand}, {self._window})"

    def evaluate(self, data: StockData) -> Tensor:

        operand_value = self._operand.evaluate(data)

        t, n = operand_value.shape

        if t <= self._window:
            return torch.full(
                size=(t, n),
                fill_value=torch.nan,
                dtype=operand_value.dtype,
                device=operand_value.device,
            )
        window_values = operand_value.unfold(0, self._window + 1, 1)
        valid_value = self._apply(window_values)
        prefix_value = torch.full(
            size=(self._window, n),
            fill_value=torch.nan,
            dtype=operand_value.dtype,
            device=operand_value.device,
        )
        value = torch.cat([prefix_value, valid_value], dim=0)

        return value
    
    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

class PAST(PastOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand[:, :, 0]

class DELTA(PastOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand[:, :, -1] - operand[:, :, 0]


class PairRollingOperator(Expression):
    """
    双变量滚动算子基类
    对包含 t 日在内的过去 window 长度窗口做聚合
    包括:
    - COV
    - CORR
    - REG

    """

    pass


class FORWARDRET(Expression):
    """
    未来收益率表达式
    使用 t+window 日与 t+1 日比较

    原则上不纳入模型编码, 只用作 target 构造和评估
    """

    def __init__(self, operand: _ExprOrFloat, window: int) -> None:
        if window < 1:
            raise ValueError(f"窗口长度应当为正整数, 当前为: {window}")
        self._operand = _into_expr(operand)
        self._window = int(window)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand}, {self._window})"

    def evaluate(self, data: StockData) -> Tensor:

        operand_value = self._operand.evaluate(data)

        t, n = operand_value.shape

        if t <= self._window:
            return torch.full(
                size=(t, n),
                fill_value=torch.nan,
                dtype=operand_value.dtype,
                device=operand_value.device,
            )

        valid_value = operand_value[self._window:] / operand_value[1:t-self._window+1] - 1.0
        suffix_value = torch.full(
            size=(self._window, n),
            fill_value=torch.nan,
            dtype=operand_value.dtype,
            device=operand_value.device,
        )
        value = torch.cat([valid_value, suffix_value], dim=0)

        return value

