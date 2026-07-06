"""一元一次方程步骤检查器（纯规则 + fractions，不依赖 LLM / sympy）。

职责：
- 从学生消息中抽取方程（extract_equation）
- 解析/求解一元一次方程（parse_equation / solve）
- 判断学生输入是否是当前方程的合理下一步（check_step）
- 判断等式变形是否等价（is_equivalent）
- 判断最终答案是否正确（含只输入数字的情况）
- 建议下一步（suggest_next_step / next_question）

只用 stdlib（fractions.Fraction），保证判定精确，避免老师误判学生的正确步骤。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Optional

# (a, b) 表示 a*x + b
Side = tuple[Fraction, Fraction]
# ((a1, b1), (a2, b2)) 表示 a1*x + b1 = a2*x + b2
Equation = tuple[Side, Side]

_FULLWIDTH = str.maketrans({
    "＋": "+", "－": "-", "＝": "=", "×": "*", "÷": "/", "．": ".", "（": "(", "）": ")",
    "Ｘ": "x", "ｘ": "x", "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})

_MATH_CHARS = set("0123456789xX+-*/=.() ")


class StepValidity(str, Enum):
    NOT_A_STEP = "not_a_step"
    CORRECT_STEP = "correct_step"
    CORRECT_FINAL_ANSWER = "correct_final_answer"
    INCORRECT_STEP = "incorrect_step"
    UNPARSEABLE = "unparseable"


@dataclass
class StepCheckResult:
    validity: StepValidity
    student_equation: str = ""
    explanation: str = ""


def _normalize(text: str) -> str:
    return text.translate(_FULLWIDTH)


def fmt_frac(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _expand_brackets(s: str) -> Optional[str]:
    """展开形如 k(ax+b) 的简单括号（分配律）。失败返回 None。"""
    for _ in range(10):
        if "(" not in s and ")" not in s:
            return s
        m = re.search(r"([+-]?\d*(?:/\d+)?)\(([^()]*)\)", s)
        if m is None:
            return None
        inner = _parse_side(m.group(2))
        if inner is None:
            return None
        coef_str = m.group(1)
        if coef_str in ("", "+"):
            k = Fraction(1)
        elif coef_str == "-":
            k = Fraction(-1)
        else:
            try:
                k = Fraction(coef_str)
            except (ValueError, ZeroDivisionError):
                return None
        a, b = k * inner[0], k * inner[1]
        rep = ""
        if a != 0:
            rep += ("+" if a > 0 else "-") + (fmt_frac(abs(a)) if abs(a) != 1 else "") + "x"
        if b != 0:
            rep += ("+" if b > 0 else "-") + fmt_frac(abs(b))
        if not rep:
            rep = "+0"
        s = s[:m.start()] + rep + s[m.end():]
    return None


def _parse_side(side: str) -> Optional[Side]:
    s = side.replace(" ", "").replace("*", "")
    if not s:
        return None
    if "(" in s or ")" in s:
        expanded = _expand_brackets(s)
        if expanded is None:
            return None
        s = expanded
    tokens = re.findall(r"[+-]?[^+-]+", s)
    a, b = Fraction(0), Fraction(0)
    for tok in tokens:
        if "x" in tok.lower():
            coef = tok.lower().replace("x", "")
            if coef in ("", "+"):
                a += 1
            elif coef == "-":
                a -= 1
            elif coef.startswith("/") or coef.startswith("-/"):
                # x/2 -> 1/2；-x/2 由符号前缀处理
                try:
                    a += Fraction("1" + coef) if coef.startswith("/") else -Fraction("1" + coef[1:])
                except (ValueError, ZeroDivisionError):
                    return None
            else:
                try:
                    a += Fraction(coef)
                except (ValueError, ZeroDivisionError):
                    return None
        else:
            try:
                b += Fraction(tok)
            except (ValueError, ZeroDivisionError):
                return None
    return a, b


def parse_equation(eq: str) -> Optional[Equation]:
    """解析形如 '3x+6=10' 的一元一次方程；解析失败返回 None。"""
    t = _normalize(eq)
    if t.count("=") != 1:
        return None
    left, right = t.split("=")
    ls, rs = _parse_side(left), _parse_side(right)
    if ls is None or rs is None:
        return None
    return ls, rs


def format_side(side: Side) -> str:
    a, b = side
    parts: list[str] = []
    if a != 0:
        if a == 1:
            parts.append("x")
        elif a == -1:
            parts.append("-x")
        else:
            parts.append(f"{fmt_frac(a)}x")
    if b != 0:
        if parts:
            parts.append(f"+ {fmt_frac(b)}" if b > 0 else f"- {fmt_frac(-b)}")
        else:
            parts.append(fmt_frac(b))
    return " ".join(parts) if parts else "0"


def format_equation(eq: Equation) -> str:
    return f"{format_side(eq[0])} = {format_side(eq[1])}"


def extract_equation(text: str) -> Optional[str]:
    """从自然语言消息中抽取第一个可解析的方程，返回规范格式（如 '3x + 6 = 10'）。"""
    t = _normalize(text)
    for i, ch in enumerate(t):
        if ch != "=":
            continue
        l = i
        while l > 0 and t[l - 1] in _MATH_CHARS and t[l - 1] != "=":
            l -= 1
        r = i + 1
        while r < len(t) and t[r] in _MATH_CHARS and t[r] != "=":
            r += 1
        candidate = t[l:r].strip()
        parsed = _parse_candidate(candidate)
        if parsed is not None:
            return format_equation(parsed)
    return None


def _parse_candidate(candidate: str) -> Optional[Equation]:
    """解析候选片段；失败时从左侧逐字符裁剪重试（去掉粘连的杂质字符）。"""
    for i in range(len(candidate)):
        sub = candidate[i:].strip()
        if "=" not in sub:
            return None
        parsed = parse_equation(sub)
        if parsed is not None:
            return parsed
    return None


def find_equations(text: str) -> list[str]:
    """抽取消息中所有可解析的方程（规范格式，去重）。"""
    t = _normalize(text)
    found: list[str] = []
    i = 0
    while i < len(t):
        if t[i] == "=":
            l = i
            while l > 0 and t[l - 1] in _MATH_CHARS and t[l - 1] != "=":
                l -= 1
            r = i + 1
            while r < len(t) and t[r] in _MATH_CHARS and t[r] != "=":
                r += 1
            parsed = _parse_candidate(t[l:r].strip())
            if parsed is not None:
                eq = format_equation(parsed)
                if eq not in found:
                    found.append(eq)
            i = r
        else:
            i += 1
    return found


def solve(eq: Equation) -> Optional[Fraction]:
    """唯一解；无唯一解（恒等/矛盾/非一元一次）返回 None。"""
    (a1, b1), (a2, b2) = eq
    a = a1 - a2
    if a == 0:
        return None
    return (b2 - b1) / a


def solve_str(eq_str: str) -> Optional[Fraction]:
    parsed = parse_equation(eq_str)
    return solve(parsed) if parsed else None


def is_equivalent(eq1: str, eq2: str) -> bool:
    """两个方程是否同解（等式变形是否等价）。"""
    s1, s2 = solve_str(eq1), solve_str(eq2)
    return s1 is not None and s1 == s2


def is_final_form(eq: Equation) -> bool:
    """是否是 'x = 常数' 或 '常数 = x' 的最终答案形式。"""
    (a1, b1), (a2, b2) = eq
    return (a1 == 1 and b1 == 0 and a2 == 0) or (a2 == 1 and b2 == 0 and a1 == 0)


def bare_number(text: str) -> Optional[Fraction]:
    """学生只输入数字（如 '4/3'、'-2'、'1.5'）时解析为分数。"""
    t = _normalize(text).strip().rstrip("。.！!")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?", t):
        return None
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def has_math_content(text: str) -> bool:
    return extract_equation(text) is not None or bare_number(text) is not None


def describe_transformation(prev_eq: str, new_eq: str) -> str:
    """描述从 prev_eq 到 new_eq 的同解变形（用于给学生解释为什么对）。"""
    p, n = parse_equation(prev_eq), parse_equation(new_eq)
    if p is None or n is None:
        return "这是对等式两边做同一种运算得到的"
    (pa1, pb1), _ = p
    (na1, nb1), _ = n
    if pb1 != 0 and nb1 == 0 and na1 == pa1:
        op = "减去" if pb1 > 0 else "加上"
        return f"原式 {prev_eq} 两边同时{op} {fmt_frac(abs(pb1))}，就得到 {new_eq}"
    if pb1 == 0 and pa1 not in (0, 1) and na1 == 1 and nb1 == 0:
        return f"原式 {prev_eq} 两边同时除以 {fmt_frac(pa1)}，就得到 {new_eq}"
    return f"{new_eq} 与 {prev_eq} 同解（等式两边做了同一种运算）"


def check_step(target_eq: str, student_text: str, prev_step: str | None = None) -> StepCheckResult:
    """判断学生输入相对原题 target_eq 是否是正确步骤/最终答案。

    prev_step：学生上一次确认到的方程状态，用于生成变形解释。
    """
    t_parsed = parse_equation(target_eq)
    if t_parsed is None:
        return StepCheckResult(StepValidity.NOT_A_STEP, explanation="原题不是可解析的方程")
    sol_t = solve(t_parsed)
    if sol_t is None:
        return StepCheckResult(StepValidity.NOT_A_STEP, explanation="原题没有唯一解")

    s_eq_str = extract_equation(student_text)
    if s_eq_str is None:
        n = bare_number(student_text)
        if n is None:
            return StepCheckResult(StepValidity.NOT_A_STEP)
        if n == sol_t:
            return StepCheckResult(
                StepValidity.CORRECT_FINAL_ANSWER,
                student_equation=f"x = {fmt_frac(n)}",
                explanation=f"x = {fmt_frac(n)} 正是原方程 {target_eq} 的解",
            )
        return StepCheckResult(
            StepValidity.INCORRECT_STEP,
            student_equation=f"x = {fmt_frac(n)}",
            explanation=f"x = {fmt_frac(n)} 不是原方程 {target_eq} 的解，把它代回去两边并不相等",
        )

    s_parsed = parse_equation(s_eq_str)
    assert s_parsed is not None
    sol_s = solve(s_parsed)
    if sol_s is None:
        return StepCheckResult(
            StepValidity.UNPARSEABLE,
            student_equation=s_eq_str,
            explanation=f"{s_eq_str} 没有唯一解，无法判断",
        )
    if sol_s == sol_t:
        if is_final_form(s_parsed):
            return StepCheckResult(
                StepValidity.CORRECT_FINAL_ANSWER,
                student_equation=s_eq_str,
                explanation=f"{s_eq_str} 就是原方程 {target_eq} 的正确解",
            )
        return StepCheckResult(
            StepValidity.CORRECT_STEP,
            student_equation=s_eq_str,
            explanation=describe_transformation(prev_step or target_eq, s_eq_str),
        )
    return StepCheckResult(
        StepValidity.INCORRECT_STEP,
        student_equation=s_eq_str,
        explanation=(
            f"{s_eq_str} 不是当前原题 {target_eq} 的正确变形（它与原方程不同解）"
        ),
    )


def suggest_next_step(eq_str: str) -> Optional[tuple[str, str]]:
    """给当前方程建议下一步，返回 (操作描述, 变形后的方程)；已解出返回 None。"""
    parsed = parse_equation(eq_str)
    if parsed is None:
        return None
    (a1, b1), (a2, b2) = parsed
    if a1 - a2 == 0:
        return None
    if a2 != 0:
        new: Equation = ((a1 - a2, b1), (Fraction(0), b2))
        op = "减去" if a2 > 0 else "加上"
        return (f"两边同时{op} {fmt_frac(abs(a2))}x，把含 x 的项移到同一边", format_equation(new))
    if b1 != 0:
        new = ((a1, Fraction(0)), (Fraction(0), b2 - b1))
        op = "减去" if b1 > 0 else "加上"
        return (f"两边同时{op} {fmt_frac(abs(b1))}", format_equation(new))
    if a1 != 1:
        sol = solve(parsed)
        assert sol is not None
        new = ((Fraction(1), Fraction(0)), (Fraction(0), sol))
        return (f"两边同时除以 {fmt_frac(a1)}", format_equation(new))
    return None


def next_question(eq_str: str) -> str:
    """针对当前方程状态，生成一个（且只有一个）引导性小问题。"""
    parsed = parse_equation(eq_str)
    if parsed is None:
        return "你觉得下一步应该做什么？"
    (a1, b1), (a2, _) = parsed
    if a2 != 0:
        return "下一步：怎样把含 x 的项移到等号同一边？"
    if b1 != 0:
        op = "减去" if b1 > 0 else "加上"
        return f"下一步：两边同时{op}几，才能只留下含 x 的项？"
    if a1 != 1:
        return "现在要把 x 单独留下来，下一步应该两边同时除以几？"
    return "你已经得到 x 的值了，把它代回原方程检验一下，两边相等吗？"


def uses_new_example(response: str, target_eq: str) -> bool:
    """回复中是否出现了与原题不同解的新方程（新例题）。"""
    sol_t = solve_str(target_eq)
    if sol_t is None:
        return False
    for eq in find_equations(response):
        sol = solve_str(eq)
        if sol is not None and sol != sol_t:
            return True
    return False


def mentions_equation(text: str, eq_str: str) -> bool:
    """回复中是否提到了该方程（忽略空格差异）。"""
    if not eq_str:
        return False
    return eq_str.replace(" ", "") in _normalize(text).replace(" ", "")
