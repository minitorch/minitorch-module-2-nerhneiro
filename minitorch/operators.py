"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List, Any

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x : float, y : float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return float(-x)


def lt(x: float, y: float) -> float:
    return float(x < y)


def eq(x: float, y: float) -> float:
    return float(x == y)


def max(x: float, y: float) -> float:
    if x >= y:
        return x
    return y


def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-5


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return max(0.0, x)


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    return y / x


def inv(x: float) -> float:
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    return -1.0 / x ** 2 * y


def relu_back(x: float, y: float) -> float:
    if x > 0:
        return y
    return 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

# TODO: Implement for Task 0.3.


def map(f: Callable[[float], float], arr: Iterable[float]) -> List[float]:
    return [f(x) for x in arr]


def zipWith(f: Callable[[float, float], float], arr1: Iterable[float], arr2: Iterable[float]) -> List[float]:
    answer: List[float] = []
    try:
        it1 = iter(arr1)
        it2 = iter(arr2)
        while True:
            next1 = next(it1)
            next2 = next(it2)
            answer.append(f(next1, next2))
    except StopIteration:
        return answer


def reduce(f: Callable[[float, float], float], arr: Iterable[float]) -> float:
    it = iter(arr)
    answer: float = 0.0
    next_val: float = 0.0
    first = True
    try:
        while True:
            if first:
                answer = next(it)
                first = False
            next_val = next(it)
            answer = f(answer, next_val)
    except StopIteration:
        return answer


def negList(arr: List[float]) -> List[float]:
    return map(neg, arr)


def addLists(arr1: List[float], arr2: List[float]) -> List[float]:
    return zipWith(add, arr1, arr2)


def sum(arr: List[float]) -> float:
    return reduce(add, arr)


def prod(arr: List[float]) -> float:
    return reduce(mul, arr)
