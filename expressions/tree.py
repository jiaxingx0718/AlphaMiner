from __future__ import annotations

from typing import Any

from expressions.expression import (
    BinaryOperator,
    Constant,
    Expression,
    Feature,
    FORWARDRET,
    PastOperator,
    RollingOperator,
    UnaryOperator,
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


class InvalidTokenError(ValueError):
    """
    token 序列无法构造成合法表达式树时抛出的异常
    """


class ExpressionBuilder:
    """
    可以接收 Token 类型数据, 以树结构生成最终表达式构造器

    类内部维护一个栈, 采用栈+算子规约结构
    每次可以通过 add_token 加入 token, 然后执行语法检查, 再变换入栈
    通过 get_tree 获取当前栈唯一的 Expression 作为最终结果
    
    例如 DIV(ADD(CLOSE, OPEN), 2) 的读取顺序应当是 CLOSE - OPEN - ADD - 2 - DIV
    
    
    叶子节点:
    FeatureToken:
    - 检查: 当前未结束, 栈顶不是 WindowToken
    - 变换: FeatureToken(feature) -> Feature(feature) 入栈

    ConstantToken:
    - 检查: 当前未结束, 栈顶不是 WindowToken
    - 变换: ConstantToken(constant) -> Constant(constant) 入栈

    WindowToken:
    - 检查: 当前未结束, 栈不能为空, 栈顶必须是 Expression
    - 变换: WindowToken(window) -> window 入栈

    
    中间节点:
    UnaryOperatorToken:
    - 检查: 当前未结束, 栈不能为空, 栈顶必须是 Expression, operator 必须是 UnaryOperator 子类
    - 变换: 弹出 operand, token.operator(operand) 入栈

    BinaryOperatorToken:
    - 检查: 当前未结束, 栈至少有2个元素, 栈顶必须是两个 Expression, operator 必须是 BinaryOperator 子类
    - 变换: 弹出 R_operand, 弹出 L_operand, token.operator(L_operand, R_operand) 入栈

    RollingOperatorToken:
    - 检查: 当前未结束, 栈至少有2个元素, 栈顶必须是整数, 倒数第二个必须是 Expression, operator 必须是 RollingOperator 子类
    - 变换: 弹出 window, 弹出 operand, token.operator(operand, window) 入栈

    # PairRollingOperatorToken:
    # - 检查: 当前未结束, 栈至少有3个元素, 栈顶必须是整数, 倒数第二个, 第三个必须是 Expression, operator 必须是 PairRollingOperator
    # - 变换: 弹出 window, 弹出 R_operand, 弹出 L_operand, token.operator(L_operand, R_operand, window) 入栈
    

    子树 (相当于叶子节点):
    ExpressionToken:
    - 检查: 当前未结束, 栈顶不是 WindowToken
    - 变换: ExpressionToken(expression) -> expression, 入栈

    
    开始和结束:
    SequenceIndicatorToken(BEGIN):
    - 检查: 当前未结束, 栈必须为空
    - 变换: 无

    SequenceIndicatorToken(END):
    - 检查: 当前未结束, 栈里正好有一个元素, 栈顶必须是 Expression
    - 变换: 标记 _ended=True

    
    对于操作符的validate 有额外的 tokenless 参数控制是否查看具体算子, 用于生成下一个 token 的候选空间
    """

    def __init__(self) -> None:
        self.stack: list[Any] = []
        self._ended = False

    def __repr__(self) -> str:
        return f"构建表达式树, 当前栈: {self.stack}"


    @property
    def ended(self) -> bool:
        return self._ended

    def reset(self) -> None:
        self.stack.clear()
        self._ended = False

    def get_tree(self) -> Expression:
        if len(self.stack) != 1:
            raise InvalidTokenError(f"表达式树栈应当只有一个元素, 当前栈: {self.stack}")
        expr = self.stack.pop()
        if not isinstance(expr, Expression):
            raise InvalidTokenError(f"表达式树栈顶元素不是表达式: {expr}")
        return expr


    def add_token(self, token: Token) -> None:

        if self._ended:
            raise InvalidTokenError("表达式生成已结束, 需要 reset() 重置")
        
        if not self.validate(token):
            raise InvalidTokenError(f"Token {token} 在此位置加入不合法, 当前栈: {self.stack}")

        if isinstance(token, FeatureToken):
            self.stack.append(Feature(token.feature))
            return

        if isinstance(token, ConstantToken):
            self.stack.append(Constant(token.constant))
            return

        if isinstance(token, WindowToken):
            self.stack.append(token.window)
            return

        if isinstance(token, UnaryOperatorToken):
            operand = self.stack.pop()
            self.stack.append(token.operator(operand))
            return

        if isinstance(token, BinaryOperatorToken):
            R_operand = self.stack.pop()
            L_operand = self.stack.pop()
            self.stack.append(token.operator(L_operand, R_operand))
            return

        if isinstance(token, RollingOperatorToken):
            window = self.stack.pop()
            operand = self.stack.pop()
            self.stack.append(token.operator(operand, window))
            return

        if isinstance(token, ExpressionToken):
            self.stack.append(token.expression)
            return

        if isinstance(token, SequenceIndicatorToken):
            if token.indicator == SequenceIndicatorType.BEGIN:
                return
            if token.indicator == SequenceIndicatorType.END:
                self._ended = True
                return
            raise InvalidTokenError(f"表达式树构造器不支持的指示变量类型: {token.indicator}")

        
    def validate(self, token: Token) -> bool:

        if isinstance(token, FeatureToken):
            return self.validate_feature()
        if isinstance(token, ConstantToken):
            return self.validate_constant()
        if isinstance(token, WindowToken):
            return self.validate_window()
        if isinstance(token, UnaryOperatorToken):
            return self.validate_unaryop(token)
        if isinstance(token, BinaryOperatorToken):
            return self.validate_binaryop(token)
        if isinstance(token, RollingOperatorToken):
            return self.validate_rollingop(token)
        if isinstance(token, ExpressionToken):
            return self.validate_expression()
        if isinstance(token, SequenceIndicatorToken):
            return self.validate_sequence(token)
        
        raise InvalidTokenError(f"表达式树构造器不支持的 token 类型: {type(token).__name__}")


    def _top_is_int(self):
        return len(self.stack) >= 1 and isinstance(self.stack[-1], int)

    def validate_feature(self) -> bool:
        return (not self._ended 
                and not self._top_is_int())

    def validate_constant(self) -> bool:
        return (not self._ended 
                and not self._top_is_int())
    
    def validate_window(self) -> bool:
        return (not self._ended 
                and len(self.stack) >= 1 
                and isinstance(self.stack[-1], Expression))

    def validate_expression(self) -> bool:
        return (not self._ended 
                and not self._top_is_int())

    def validate_unaryop(self, token: UnaryOperatorToken | None = None, tokenless: bool=False) -> bool:
        return (not self._ended 
                and len(self.stack) >= 1
                and (tokenless
                     or issubclass(token.operator, UnaryOperator))
                and isinstance(self.stack[-1], Expression))

    def validate_binaryop(self, token: BinaryOperatorToken | None = None, tokenless: bool=False) -> bool:
        return (not self._ended 
                and len(self.stack) >= 2
                and (tokenless
                     or issubclass(token.operator, BinaryOperator))
                and isinstance(self.stack[-2], Expression) 
                and isinstance(self.stack[-1], Expression))

    def validate_rollingop(self, token: RollingOperatorToken | None = None, tokenless: bool=False) -> bool:
        return (not self._ended 
                and len(self.stack) >= 2
                and (tokenless
                     or issubclass(token.operator, RollingOperator) 
                     or issubclass(token.operator, PastOperator)
                     or token.operator is FORWARDRET)
                and isinstance(self.stack[-2], Expression) 
                and isinstance(self.stack[-1], int))

    def validate_sequence(self, token: SequenceIndicatorToken) -> bool:
        if token.indicator == SequenceIndicatorType.BEGIN:
            return (len(self.stack) == 0 
                    and not self._ended)
        if token.indicator == SequenceIndicatorType.END:
            return (len(self.stack) == 1 
                    and isinstance(self.stack[-1], Expression))
        return False




__all__ = [
    "InvalidTokenError",
    "ExpressionBuilder",
    "END_TOKEN",
]
