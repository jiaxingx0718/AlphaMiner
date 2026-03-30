from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union

import pandas as pd
import torch
from torch import Tensor

from data.datatotensor import StockData
from expressions.expression import Expression

_TargetLike = Union[Expression, Tensor, None]


class InvalidEvaluateError(RuntimeError):
    """
    当计算表达式在评估窗口某一天存在过多无效值时抛出的异常
    """


def _mask_invalid(
    x: Tensor,
    y: Tensor,
    fill_with: float=0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    对于相同形状的 x, y (通常是未来统计的因子和未来一期收益率 (D, N) tensor)
    其中 x, y 至少有一个非有限值的位置算作无效位置, 预填充0, 返回:
    - 处理后的 (D, N) tensor x
    - 处理后的 (D, N) tensor y
    - 每一天有效股票数 (D,) tensor n_valid
    - 无效位置 invalid_mask

    目标是在保留 invalid_mask 位置时在后面方便地实现相关系数计算而不必担心无效值
    """
    x = x.clone()
    y = y.clone()
    invalid_mask = (~torch.isfinite(x)) | (~torch.isfinite(y))
    x[invalid_mask] = fill_with
    y[invalid_mask] = fill_with
    n_valid = (~invalid_mask).sum(dim=1)
    return x, y, n_valid, invalid_mask


def masked_mean_std(
    x: Tensor,
    n: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    对因子 (D, N) tensor , 计算每一天的平均值和标准差
    """
    if mask is None:
        mask = ~torch.isfinite(x)
    if n is None:
        n = (~mask).sum(dim=1)
    x = x.clone()
    x[mask] = 0.0

    n_safe = n.clamp_min(1)
    mean = x.sum(dim=1) / n_safe
    std = ((((x - mean[:, None]) * ~mask) ** 2).sum(dim=1) / n_safe).sqrt()

    return mean, std


def _rank_data_1d(x: Tensor) -> Tensor:
    """
    将截面 (N,) tensor 转换为 rank, 用于计算Spearman相关系数
    对于重复值作平均秩处理, 对每个值先计算排名范围再取平均值
    """
    _, inv, counts = x.unique(return_inverse=True, return_counts=True) # .unique已经实现排序
    cs = counts.cumsum(dim=0).to(x.dtype)
    cs = torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device), cs], dim=0)
    ranks = (cs[:-1] + cs[1:] - 1) / 2

    return ranks[inv]


def _rank_data(x: Tensor, invalid_mask: Tensor) -> Tensor:
    """
    对 (D, N,) tensor 逐行计算 rank 并对无效位置归0
    """
    rank = torch.stack([_rank_data_1d(row) for row in x])
    rank[invalid_mask] = 0.0
    return rank


def _batch_pearsonr_given_mask(
    x: Tensor,
    y: Tensor,
    n: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    对于给定的一对 (D, N) tensor, 计算其按天的Pearson相关系数序列 (D,) tensor
    x, y, n 定义同 _mask_invalid, masked_mean_std
    通常情况接收的 x, y 是经过 _mask_invalid 填充的, 因此不会有无效值问题
    """
    x_mean, x_std = masked_mean_std(x, n, mask)
    y_mean, y_std = masked_mean_std(y, n, mask)
    safe_n = n.clamp_min(1)
    cov = (x * y).sum(dim=1) / safe_n - x_mean * y_mean
    stdmul = x_std * y_std
    stdmul[(x_std < 1e-12) | (y_std < 1e-12)] = 1.0
    corrs = cov / stdmul
    corrs[n < 2] = torch.nan

    return corrs


def batch_pearsonr(x: Tensor, y: Tensor) -> Tensor:
    """
    计算其按天的Pearson相关系数序列 (D,) tensor
    """
    return _batch_pearsonr_given_mask(*_mask_invalid(x, y, fill_with=0.0))


def batch_spearmanr(x: Tensor, y: Tensor) -> Tensor:
    """
    计算其按天的Spearman相关系数序列 (D,) tensor
    """
    x, y, n, invalid_mask = _mask_invalid(x, y, fill_with=0.0)
    rx = _rank_data(x, invalid_mask)
    ry = _rank_data(y, invalid_mask)
    return _batch_pearsonr_given_mask(rx, ry, n, invalid_mask)


class AlphaCalculator(metaclass=ABCMeta):
    """
    所有Alpha评估的抽象基类, 要求子类必须实现对Expression计算一系列计算IC的功能
    """
    @abstractmethod
    def calc_single_IC(self, expr: Expression) -> float:
        """计算单个表达式的IC"""

    @abstractmethod
    def calc_single_rankIC(self, expr: Expression) -> float:
        """计算单个表达式的rankIC"""

    @abstractmethod
    def calc_single_IC_daily(self, expr: Expression) -> Tensor:
        """计算单个表达式按天的IC序列 (D,)"""

    @abstractmethod
    def calc_single_rankIC_daily(self, expr: Expression) -> Tensor:
        """计算单个表达式按天的rankIC序列 (D,)"""

    def calc_single_all_IC(self, expr: Expression) -> tuple[float, float]:
        return self.calc_single_IC(expr), self.calc_single_rankIC(expr)

    def calc_single_all_IC_daily(self, expr: Expression) -> tuple[Tensor, Tensor]:
        return self.calc_single_IC_daily(expr), self.calc_single_rankIC_daily(expr)


    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        """计算两个表达式之间的mutualIC"""


    @abstractmethod
    def calc_pool_IC(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        """计算线性组合后的poolIC"""

    @abstractmethod
    def calc_pool_rankIC(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        """计算线性组合后的poolrankIC"""

    def calc_pool_all_IC(self, exprs: Sequence[Expression], weights: Sequence[float]) -> tuple[float, float]:
        return self.calc_pool_IC(exprs, weights), self.calc_pool_rankIC(exprs, weights)


class TensorAlphaCalculator(AlphaCalculator):
    """
    基于 tensor 的Alpha评估抽象基类, 具体实现了 AlphaCalculator 要求的一系列IC计算功能

    要求子类必须实现:
    - 绑定具体的数据 (StockData)
    - evaluate_alpha 将表达式转换为并切成 (D, N) tensor

    如果要求实现单因子IC或poolIC, 则子类还应实现:
    - 根据 StockData 计算 target (未来一期收益率) (D, N) tensor
    """

    def __init__(self, target: Optional[Tensor]) -> None:
        self._target = target

    @property
    def target(self) -> Tensor:
        if self._target is None:
            raise ValueError("计算单因子或组合IC之前必须先设置 target")
        return self._target

    @abstractmethod
    def evaluate_alpha(self, expr: Expression) -> Tensor:
        """计算因子表达式值 (T, N) tensor 并切为 (D, N) 的 tensor"""

    def make_ensemble_alpha(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tensor:
        """
        将多个表达式的 evaluate_alpha 结果按照给定 weights 线性组合
        等价于先按照 weights 线性组合形成总因子再进行评估
        """
        if len(exprs) != len(weights):
            raise ValueError("exprs 和 weights 长度必须一致")
        factors = [self.evaluate_alpha(exprs[i]) * weights[i] for i in range(len(exprs))]
        return torch.sum(torch.stack(factors, dim=0), dim=0)


    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).nanmean().item()

    def _calc_rankIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).nanmean().item()

    def _calc_IC_daily(self, value1: Tensor, value2: Tensor) -> Tensor:
        return batch_pearsonr(value1, value2)

    def _calc_rankIC_daily(self, value1: Tensor, value2: Tensor) -> Tensor:
        return batch_spearmanr(value1, value2)


    def calc_single_IC(self, expr: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr), self.target)

    def calc_single_rankIC(self, expr: Expression) -> float:
        return self._calc_rankIC(self.evaluate_alpha(expr), self.target)

    def calc_single_IC_daily(self, expr: Expression) -> Tensor:
        return self._calc_IC_daily(self.evaluate_alpha(expr), self.target)

    def calc_single_rankIC_daily(self, expr: Expression) -> Tensor:
        return self._calc_rankIC_daily(self.evaluate_alpha(expr), self.target)


    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))


    def calc_pool_IC(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        value = self.make_ensemble_alpha(exprs, weights)
        return self._calc_IC(value, self.target)

    def calc_pool_rankIC(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        value = self.make_ensemble_alpha(exprs, weights)
        return self._calc_rankIC(value, self.target)

    
    @property
    @abstractmethod
    def n_days(self) -> int: ...


def winsorize_by_day(
    value: Tensor,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tensor:
    """
    对因子 (D, N) tensor 作截面去极值
    """
    if not (0.0 <= lower_q <= upper_q <= 1.0):
        raise ValueError("winsorize 分位数必须满足 0 <= lower_q <= upper_q <= 1")

    value = value.clone()

    for i in range(value.shape[0]):
        row = value[i]
        finite_mask = torch.isfinite(row)
        if not finite_mask.any():
            continue
        finite_values = row[finite_mask]
        lower = torch.quantile(finite_values, lower_q)
        upper = torch.quantile(finite_values, upper_q)
        row[finite_mask] = row[finite_mask].clamp(min=lower, max=upper)

    return value


def normalize_by_day(value: Tensor) -> Tensor:
    """
    对因子 (D, N) tensor 作截面标准化
    """
    invalid_mask = ~torch.isfinite(value)
    mean, std = masked_mean_std(value)

    safe_std = std.clone()
    safe_std[safe_std < 1e-12] = 1.0
    value = (value - mean[:, None]) / safe_std[:, None]

    value[invalid_mask] = torch.nan

    return value


class StockDataCalculator(TensorAlphaCalculator):
    """
    AlphaMiner 使用的 TensorAlphaCalculator:
    - _build_eval_slice 指定回测区间切片 D, 它是全长 T 的一部分
    无论对于 evaluate_alpha 接收的因子还是 target (通常是未来一期收益率), 必须经过_prepare_value():
    - 先计算因子 Expression 的 evaluate (T, N) tensor, 然后将其切片为 (D, N) tensor 作为评估的因子值
    - _check_invalid 检测因子值是否有超出限制数量的无效值, 如果有则抛出 InvalidEvaluateError
    - _postprocess 对因子值作截面winsorize去极值和标准化
    """

    def __init__(
        self,
        data: StockData,
        target: _TargetLike = None,
        eval_start: Optional[Union[str, pd.Timestamp]] = None,
        eval_end: Optional[Union[str, pd.Timestamp]] = None,
        max_invalid: Optional[int] = None,
        cs_max_invalid: Optional[int] = None,
        winsorize: bool = True,
        normalize: bool = True,
    ) -> None:
        """
        Args:
        - data: 股票数据面板, 其中 data.data 为 (T, F, N) tensor
        - target: 计算IC的目标, 需要为表达式或者 (T, N) tensor
        - eval_start: 因子测试起始日期
        - eval_end: 因子测试结束日期
        - max_invalid: 允许评估窗口内无效值的最大数量
        - cs_max_invalid: 允许单日横截面中无效值的最大数量
        - winsorize: 是否对因子去极值化
        - normalize: 是否对因子标准化
        """
        self.data = data
        self._max_invalid = max_invalid
        self._cs_max_invalid = cs_max_invalid
        self._winsorize = winsorize
        self._normalize = normalize
        self._eval_slice = self._build_eval_slice(eval_start, eval_end)

        target_tensor: Optional[Tensor]
        if isinstance(target, Expression):
            target_tensor = self._prepare_value(target.evaluate(data))
        elif isinstance(target, Tensor):
            target_tensor = self._prepare_value(target)
        else:
            target_tensor = None

        super().__init__(target_tensor)


    def _build_eval_slice(
        self,
        eval_start: Optional[Union[str, pd.Timestamp]],
        eval_end: Optional[Union[str, pd.Timestamp]],
    ) -> slice:
        dates = self.data.dates
        start_idx = 0
        end_idx = len(dates)

        if eval_start is not None:
            start_ts = pd.Timestamp(eval_start)
            start_idx = int(dates.searchsorted(start_ts, side="left"))

        if eval_end is not None:
            end_ts = pd.Timestamp(eval_end)
            end_idx = int(dates.searchsorted(end_ts, side="right"))

        if start_idx >= end_idx:
            raise ValueError(f"评估窗口为空, 当前起始时间为 {eval_start}, {eval_end}")

        return slice(start_idx, end_idx)

    def _check_invalid(self, value: Tensor) -> None:

        if self._max_invalid is not None:
            invalid_count = (~torch.isfinite(value)).sum()
            if invalid_count >= self._max_invalid:
                raise InvalidEvaluateError(
                    f"表达式在评估窗口内无效值过多, "
                    f"invalid={invalid_count}, limit={self._max_invalid}"
                )

        if self._cs_max_invalid is not None:
            invalid_count = (~torch.isfinite(value)).sum(dim=1)
            bad_days = invalid_count > self._cs_max_invalid
            if bad_days.any():
                first_bad_idx = int(torch.nonzero(bad_days, as_tuple=False)[0].item())
                first_bad_date = self.data.dates[self._eval_slice][first_bad_idx]
                first_bad_count = int(invalid_count[first_bad_idx].item())
                raise InvalidEvaluateError(
                    f"表达式在交易日内无效值过多: {first_bad_date}, "
                    f"invalid={first_bad_count}, limit={self._cs_max_invalid}"
                )

    def _postprocess(self, value: Tensor) -> Tensor:
        value = value.clone()
        if self._winsorize:
            value = winsorize_by_day(value)
        if self._normalize:
            value = normalize_by_day(value)
        else:
            invalid_mask = ~torch.isfinite(value)
            value[invalid_mask] = 0.0
        return value

    def _prepare_value(self, value: Tensor) -> Tensor:
        value = value[self._eval_slice].clone()
        self._check_invalid(value)
        return self._postprocess(value)

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        return self._prepare_value(expr.evaluate(self.data))

    
    @property
    def n_days(self) -> int:
        return len(range(*self._eval_slice.indices(self.data.n_days)))

    @property
    def eval_slice(self) -> slice:
        return self._eval_slice

    @property
    def eval_dates(self):
        return self.data.dates[self._eval_slice]