from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from expressions.expression import (
    BinaryOperator,
    Constant,
    Expression,
    Feature,
    FORWARDRET,
    PairRollingOperator,
    PastOperator,
    RollingOperator,
    UnaryOperator,
)
from expressions.tokens import (
    BinaryOperatorToken,
    ConstantToken,
    END_TOKEN,
    FeatureToken,
    RollingOperatorToken,
    SequenceIndicatorToken,
    Token,
    UnaryOperatorToken,
    WindowToken,
)
from expressions.tree import ExpressionBuilder
from RL.wrapper import ACTION_TOKENS


@dataclass(frozen=True)
class TokenSignature:
    """
    Stable token identity used to map expression tokens into RL action ids.

    We avoid relying on `str(token)` because:
    - ConstantToken(5.0) and WindowToken(5) print similarly
    - the same operator name can appear under different token categories
    """

    category: str
    value: object


class UnsupportedExpressionError(RuntimeError):
    """
    Raised when the current AlphaMiner expression system cannot yet be serialized
    into the AlphaForge-style postfix action space.
    """


def _token_signature(token: Token) -> TokenSignature:
    if isinstance(token, FeatureToken):
        return TokenSignature("feature", token.feature)
    if isinstance(token, ConstantToken):
        return TokenSignature("constant", float(token.constant))
    if isinstance(token, WindowToken):
        return TokenSignature("window", int(token.window))
    if isinstance(token, UnaryOperatorToken):
        return TokenSignature("unary", token.operator)
    if isinstance(token, BinaryOperatorToken):
        return TokenSignature("binary", token.operator)
    if isinstance(token, RollingOperatorToken):
        return TokenSignature("rolling", token.operator)
    if isinstance(token, SequenceIndicatorToken):
        return TokenSignature("sequence", token.indicator)
    raise UnsupportedExpressionError(f"Unsupported token type: {type(token).__name__}")


class AlphaForgeTokenizer:
    """
    Convert current AlphaMiner expressions into AlphaForge-style fixed-length
    postfix action sequences.

    Design choices:
    - sequence does not include BEGIN
    - END is appended by default
    - padding is also done with END, matching the original AlphaForge builder code
    """

    def __init__(self, max_len: int = 20) -> None:
        self.max_len = int(max_len)
        self.action_tokens = tuple(ACTION_TOKENS)
        self.end_action_id = self._build_action_index()[_token_signature(END_TOKEN)]

    def _build_action_index(self) -> dict[TokenSignature, int]:
        return {
            _token_signature(token): idx
            for idx, token in enumerate(self.action_tokens)
        }

    @property
    def action_index(self) -> dict[TokenSignature, int]:
        return self._build_action_index()

    @property
    def n_actions(self) -> int:
        return len(self.action_tokens)

    def expression_to_postfix_tokens(self, expr: Expression) -> list[Token]:
        """
        Serialize an AlphaMiner expression tree into postfix token order.

        This exactly matches the order expected by ExpressionBuilder / RL wrapper.
        """
        if isinstance(expr, Feature):
            return [FeatureToken(expr._feature)]

        if isinstance(expr, Constant):
            return [ConstantToken(expr._constant)]

        if isinstance(expr, UnaryOperator):
            return (
                self.expression_to_postfix_tokens(expr._operand)
                + [UnaryOperatorToken(type(expr))]
            )

        if isinstance(expr, BinaryOperator):
            return (
                self.expression_to_postfix_tokens(expr._L_operand)
                + self.expression_to_postfix_tokens(expr._R_operand)
                + [BinaryOperatorToken(type(expr))]
            )

        if isinstance(expr, RollingOperator) or isinstance(expr, PastOperator) or isinstance(expr, FORWARDRET):
            return (
                self.expression_to_postfix_tokens(expr._operand)
                + [WindowToken(expr._window)]
                + [RollingOperatorToken(type(expr))]
            )

        if isinstance(expr, PairRollingOperator):
            raise UnsupportedExpressionError(
                "PairRollingOperator serialization is not implemented yet in AlphaMiner."
            )

        raise UnsupportedExpressionError(
            f"Unsupported expression type for AlphaForge tokenizer: {type(expr).__name__}"
        )

    def expression_to_action_ids(
        self,
        expr: Expression,
        append_end: bool = True,
        pad_to_max: bool = True,
    ) -> list[int]:
        tokens = self.expression_to_postfix_tokens(expr)
        if append_end:
            tokens = [*tokens, END_TOKEN]

        action_index = self.action_index
        action_ids = [action_index[_token_signature(token)] for token in tokens]

        if len(action_ids) > self.max_len:
            raise UnsupportedExpressionError(
                f"Expression length {len(action_ids)} exceeds tokenizer max_len={self.max_len}"
            )

        if pad_to_max and len(action_ids) < self.max_len:
            action_ids = [*action_ids, *([self.end_action_id] * (self.max_len - len(action_ids)))]

        return action_ids

    def expressions_to_action_tensor(
        self,
        exprs: Sequence[Expression],
        append_end: bool = True,
        pad_to_max: bool = True,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        ids = [
            self.expression_to_action_ids(expr, append_end=append_end, pad_to_max=pad_to_max)
            for expr in exprs
        ]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def expressions_to_onehot(
        self,
        exprs: Sequence[Expression],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        action_tensor = self.expressions_to_action_tensor(exprs, device=device)
        return F.one_hot(action_tensor, num_classes=self.n_actions).float()

    def action_ids_to_expression(self, action_ids: Iterable[int]) -> Expression:
        builder = ExpressionBuilder()
        for action_id in action_ids:
            token = self.action_tokens[int(action_id)]
            builder.add_token(token)
            if token is END_TOKEN:
                break
        return builder.get_tree()

