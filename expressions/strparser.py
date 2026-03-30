from __future__ import annotations

import re

from data.datatotensor import FeatureType

from expressions.expression import Expression
from expressions.expression import (
    ABS, SIGN, LOG, EXP, CSRANK,
    ADD, SUB, MUL, DIV, POW, BINMAX, BINMIN,
    SUM, STD, SKEW, KURT, MAX, MIN, MED, RANK, MA, WMA, EMA,
    PAST, DELTA, FORWARDRET,
)

from expressions.tokens import (
    BinaryOperatorToken,
    ConstantToken,
    END_TOKEN,
    ExpressionToken,
    FeatureToken,
    SequenceIndicatorToken,
    SequenceIndicatorType,
    Token,
    UnaryOperatorToken,
    RollingOperatorToken,
    WindowToken,
)
from expressions.tree import ExpressionBuilder


"""
将字符串类型解析 tokens.py 定义 Token 类序列, 再通过 tree.py 构造实际表达式算子, 主要用于人工输入和LLM输出
"""


_TOKEN_PATTERN = re.compile(r"[A-Z_][A-Z_]*|[+-]?\d+(?:\.\d+)?|[(),]")


NAME_TO_TOKEN_MAP = {
    "OPEN": FeatureToken(FeatureType.OPEN),
    "CLOSE": FeatureToken(FeatureType.CLOSE),
    "HIGH": FeatureToken(FeatureType.HIGH),
    "LOW": FeatureToken(FeatureType.LOW),
    "VOLUME": FeatureToken(FeatureType.VOLUME),
    "AMOUNT": FeatureToken(FeatureType.AMOUNT),

    "ABS": UnaryOperatorToken(ABS),
    "SIGN": UnaryOperatorToken(SIGN),
    "LOG": UnaryOperatorToken(LOG),
    "EXP": UnaryOperatorToken(EXP),
    "CSRANK": UnaryOperatorToken(CSRANK),

    "ADD": BinaryOperatorToken(ADD),
    "SUB": BinaryOperatorToken(SUB),
    "MUL": BinaryOperatorToken(MUL),
    "DIV": BinaryOperatorToken(DIV),
    "POW": BinaryOperatorToken(POW),
    "BINMAX": BinaryOperatorToken(BINMAX),
    "BINMIN": BinaryOperatorToken(BINMIN),

    "SUM": RollingOperatorToken(SUM),
    "STD": RollingOperatorToken(STD),
    "SKEW": RollingOperatorToken(SKEW),
    "KURT": RollingOperatorToken(KURT),
    "MAX": RollingOperatorToken(MAX),
    "MIN": RollingOperatorToken(MIN),
    "MED": RollingOperatorToken(MED),
    "RANK": RollingOperatorToken(RANK),
    "MA": RollingOperatorToken(MA),
    "WMA": RollingOperatorToken(WMA),
    "EMA": RollingOperatorToken(EMA),
    "PAST": RollingOperatorToken(PAST),
    "DELTA": RollingOperatorToken(DELTA),
    "FORWARDRET": RollingOperatorToken(FORWARDRET)
}


class StringParsingError(ValueError):
    """
    字符串解析为 token 序列失败时抛出的异常
    """


def clean_expression(expr: str) -> str:
    """
    对输入字符串做基本清洗:
    - 去除空白
    - 统一转成大写
    """
    return re.sub(r"\s+", "", expr).upper()


def tokensplit(expr: str) -> list[str]:
    """
    将清洗后的字符串按照 _TOKEN_PATTERN 规则切分为语法单元, 语法相对比较严格:
    - [A-Z_][A-Z_]* 对应 CLOSE, MIN 这类名字 token
    - [+-]?\d+(?:\.\d+)? 对应 -3.14, 520 这类数字token
    - [(),] 对应左右括号和逗号
    """
    cleaned = clean_expression(expr)
    tokens = _TOKEN_PATTERN.findall(cleaned)
    if "".join(tokens) != cleaned:
        raise StringParsingError(f"字符串中存在无法识别的片段: {expr}")
    return tokens


class StringTokenParser:
    """
    将字符串翻译为 Token 序列, 使得其最终可以通过 tree.py 构造成最终 Expression

    例如 'MA(CLOSE, 5)' -> [FeatureToken(FeatureType.CLOSE), WindowToken(5), RollingOperatorToken(MA)]
    """

    def __init__(self) -> None:
        self._pieces: list[str] = []
        self._pos = 0

    def parse_to_tokens(self, expr: str) -> list[Token]:
        self._pieces = tokensplit(expr)
        self._pos = 0
        tokens = self._parse_expr()
        if self._pos != len(self._pieces):
            raise StringParsingError(f"存在多余 token: {self._pieces[self._pos:]}")
        return tokens

    def _parse_expr(self) -> list[Token]:

        token = self._peek()

        if token is None:
            raise StringParsingError("表达式提前结束")

        if token in NAME_TO_TOKEN_MAP:
            named = self._clone_named_token(self._pop())
            if isinstance(named, FeatureToken):
                return [named]
            if isinstance(named, (UnaryOperatorToken, BinaryOperatorToken, RollingOperatorToken)):
                return self._parse_call(named)
            raise StringParsingError(f"不支持的命名 token 类型: {type(named).__name__}")

        if self._is_number(token):
            return [ConstantToken(self._to_float(self._pop()))]

        raise StringParsingError(f"无法识别的表达式片段: {token}")


    def _parse_call(
        self,
        op_token: UnaryOperatorToken | BinaryOperatorToken | RollingOperatorToken,
    ) -> list[Token]:
        self._expect("(")

        if isinstance(op_token, UnaryOperatorToken):
            arg = self._parse_expr()
            self._expect(")")
            return arg + [op_token]

        if isinstance(op_token, BinaryOperatorToken):
            lhs = self._parse_expr()
            self._expect(",")
            rhs = self._parse_expr()
            self._expect(")")
            return lhs + rhs + [op_token]

        operand = self._parse_expr()
        self._expect(",")
        window = self._parse_window_token()
        self._expect(")")
        return operand + [window] + [op_token]

    def _parse_window_token(self) -> WindowToken:
        token = self._peek()
        if token is None:
            raise StringParsingError("窗口参数缺失")
        if not self._is_number(token):
            raise StringParsingError(f"窗口参数必须是数字, 当前为: {token}")
        value = self._to_float(self._pop())
        if not float(value).is_integer():
            raise StringParsingError(f"窗口参数必须是整数, 当前为: {value}")
        return WindowToken(int(value))

    def _clone_named_token(self, name: str) -> Token:
        token = NAME_TO_TOKEN_MAP[name]
        if isinstance(token, FeatureToken):
            return FeatureToken(token.feature)
        if isinstance(token, UnaryOperatorToken):
            return UnaryOperatorToken(token.operator)
        if isinstance(token, BinaryOperatorToken):
            return BinaryOperatorToken(token.operator)
        if isinstance(token, RollingOperatorToken):
            return RollingOperatorToken(token.operator)
        raise StringParsingError(f"不支持克隆的 token 类型: {type(token).__name__}")

    def _peek(self) -> str | None:
        if self._pos >= len(self._pieces):
            return None
        return self._pieces[self._pos]

    def _pop(self) -> str:
        token = self._peek()
        if token is None:
            raise StringParsingError("没有更多 token 可读取")
        self._pos += 1
        return token

    def _expect(self, expected: str) -> None:
        token = self._pop()
        if token != expected:
            raise StringParsingError(f"期望 {expected}, 实际读到 {token}")


    def parse_to_expression(self, expr: str) -> Expression:
        builder = ExpressionBuilder()
        for token in self.parse_to_tokens(expr):
            builder.add_token(token)
        return builder.get_tree()


    @staticmethod
    def _is_number(token: str) -> bool:
        return re.fullmatch(r"[+-]?\d+(?:\.\d+)?", token) is not None

    @staticmethod
    def _to_float(token: str) -> float:
        try:
            return float(token)
        except ValueError as exc:
            raise StringParsingError(f"无法将 {token} 转成数值") from exc



