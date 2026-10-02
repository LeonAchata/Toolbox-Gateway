"""Arithmetic tools."""

from __future__ import annotations

import ast
import math
import operator
from typing import Annotated

from pydantic import Field

from ..registry import registry

Number = Annotated[float, Field(description="A real number")]

_MAX_EXPRESSION_LENGTH = 200
_MAX_EXPONENT = 1000


def tidy(value: float) -> int | float:
    """Return ints for integral results so the model sees 8 instead of 8.0."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Result is not a finite number")
        rounded = round(value, 10)
        if rounded.is_integer() and abs(rounded) < 1e15:
            return int(rounded)
        return rounded
    return value


@registry.register("math")
def add(
    a: Annotated[float, Field(description="First addend")],
    b: Annotated[float, Field(description="Second addend")],
) -> int | float:
    """Add two numbers and return the sum."""
    return tidy(a + b)


@registry.register("math")
def subtract(
    a: Annotated[float, Field(description="Number to subtract from")],
    b: Annotated[float, Field(description="Number to subtract")],
) -> int | float:
    """Subtract b from a and return the difference."""
    return tidy(a - b)


@registry.register("math")
def multiply(
    a: Annotated[float, Field(description="First factor")],
    b: Annotated[float, Field(description="Second factor")],
) -> int | float:
    """Multiply two numbers and return the product."""
    return tidy(a * b)


@registry.register("math")
def divide(
    a: Annotated[float, Field(description="Dividend (numerator)")],
    b: Annotated[float, Field(description="Divisor (denominator), must not be zero")],
) -> int | float:
    """Divide a by b and return the quotient."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return tidy(a / b)


@registry.register("math")
def power(
    base: Annotated[float, Field(description="The base")],
    exponent: Annotated[float, Field(description="The exponent")],
) -> int | float:
    """Raise base to the given exponent."""
    if abs(exponent) > _MAX_EXPONENT:
        raise ValueError(f"Exponent must be between -{_MAX_EXPONENT} and {_MAX_EXPONENT}")
    try:
        result = base**exponent
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(str(exc)) from exc
    if isinstance(result, complex):
        raise ValueError("Result is a complex number")
    return tidy(result)


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("Exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args))
    raise ValueError(f"Unsupported syntax: {ast.dump(node)[:60]}")


@registry.register("math")
def calculate(
    expression: Annotated[
        str,
        Field(
            description=(
                "Arithmetic expression, e.g. '(25 * 8) + 3' or 'sqrt(2) * pi'. "
                "Supports + - * / // % **, parentheses, pi, e and the functions "
                "sqrt, abs, round, floor, ceil, log, log10, sin, cos, tan."
            )
        ),
    ],
) -> int | float:
    """Evaluate an arithmetic expression safely and return the numeric result."""
    expression = expression.strip().replace("^", "**")
    if not expression:
        raise ValueError("Expression is empty")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression is longer than {_MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc.msg}") from exc
    try:
        result = _evaluate(tree)
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero") from exc
    except (OverflowError, TypeError) as exc:
        raise ValueError(str(exc)) from exc
    if isinstance(result, complex):
        raise ValueError("Result is a complex number")
    return tidy(float(result)) if isinstance(result, float) else result
