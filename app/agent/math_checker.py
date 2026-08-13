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
    "＋": "+", "－": "-", "＝": "=", "×": "*", "·": "*", "⋅": "*", "∙": "*", "÷": "/", "．": ".", "（": "(", "）": ")",
    "Ｘ": "x", "ｘ": "x", "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})

_MATH_CHARS = set("0123456789xX+-*/=.() ")
_UNSUPPORTED_EQUATION_SYNTAX = re.compile(
    r"(?:"
    r"\*\*|\^|[²³⁴⁵⁶⁷⁸⁹]"
    r"|[xX]\s*\*\s*[xX]"
    r"|[xX]\s*[xX]"
    r"|/\s*[+-]?[xX]"
    r"|/\s*\([^)]*[xX][^)]*\)"
    r"|[xX]\s*\*?\s*\("
    r"|\)\s*\*?\s*[xX]"
    r"|\)\s*\*?\s*\("
    r")"
)


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
    return text.translate(_FULLWIDTH).replace("等于", "=")


def has_unsupported_linear_syntax(text: str) -> bool:
    """Return whether raw text is outside the linear-equation support domain."""

    return bool(_UNSUPPORTED_EQUATION_SYNTAX.search(_normalize(text)))


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
    if has_unsupported_linear_syntax(t):
        return None
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
    if has_unsupported_linear_syntax(t):
        return None
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
    """Parse one complete candidate without changing its mathematical domain."""

    return parse_equation(candidate.strip())


def find_equations(text: str) -> list[str]:
    """抽取消息中所有可解析的方程（规范格式，去重）。"""
    t = _normalize(text)
    if has_unsupported_linear_syntax(t):
        return []
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


def classify_misconception(prev_eq: str, student_eq: str) -> Optional[str]:
    """识别典型误因（当前支持：移项不变号）。识别不出返回 None。"""
    p, s = parse_equation(prev_eq), parse_equation(student_eq)
    if p is None or s is None:
        return None
    (a1, b1), (a2, b2) = p
    denom = a1 - a2
    if denom == 0:
        return None
    s_sol = solve(s)
    if s_sol is None:
        return None
    # 常数项移项忘记变号：ax+b=c 被写成 ax=c+b
    if b1 != 0 and s_sol == (b2 + b1) / denom:
        return "移项时忘记变号。"
    # 含 x 项移项忘记变号：a1x+b1=a2x+b2 被按 a1+a2 处理
    if a2 != 0 and (a1 + a2) != 0 and s_sol == (b2 - b1) / (a1 + a2):
        return "移项时忘记变号。"
    return None


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


# ---------------------------------------------------------------- 灵活回答：操作解析与多步计划

_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_NUM_PAT = r"(?:\d+(?:\.\d+)?(?:/\d+)?|[零一二两三四五六七八九十])"

_OP_VERBS = {"sub": "减去", "add": "加上", "div": "除以", "mul": "乘以"}

_OP_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("div", re.compile(rf"除以\s*({_NUM_PAT})")),
    ("mul", re.compile(rf"乘以?\s*({_NUM_PAT})")),
    ("sub", re.compile(rf"减去?\s*({_NUM_PAT})")),
    ("add", re.compile(rf"加上?\s*({_NUM_PAT})")),
    ("move", re.compile(rf"把?\s*([+-]?{_NUM_PAT})\s*移")),
]


@dataclass
class MathOperation:
    kind: str  # sub / add / div / mul / move
    value: Fraction
    pos: int


@dataclass
class PlanCheckResult:
    valid: bool
    reaches_final: bool = False
    final_value: Optional[Fraction] = None
    steps: list[str] = field(default_factory=list)
    end_equation: str = ""
    explanation: str = ""


def _parse_num_token(tok: str) -> Optional[Fraction]:
    tok = tok.strip()
    neg = tok.startswith("-")
    core = tok.lstrip("+-")
    if core in _CN_NUM:
        val = Fraction(_CN_NUM[core])
    else:
        try:
            val = Fraction(core)
        except (ValueError, ZeroDivisionError):
            return None
    return -val if neg else val


def extract_operations(text: str) -> list[MathOperation]:
    """从中文消息中按出现顺序抽取代数操作（减4/都减四/同时除以3/把4移过去…）。"""
    t = _normalize(text)
    found: list[MathOperation] = []
    taken: list[tuple[int, int]] = []
    for kind, pat in _OP_PATTERNS:
        for m in pat.finditer(t):
            span = (m.start(), m.end())
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue  # 与已匹配片段重叠（如“乘以”与“除以”不会，防御性处理）
            value = _parse_num_token(m.group(1))
            if value is None:
                continue
            found.append(MathOperation(kind=kind, value=value, pos=m.start()))
            taken.append(span)
    return sorted(found, key=lambda o: o.pos)


def apply_operation(eq: Equation, op: MathOperation) -> Optional[Equation]:
    """对方程两边同时施加操作；不合法（除以0等）返回 None。"""
    (a1, b1), (a2, b2) = eq
    n = op.value
    if op.kind == "sub":
        return (a1, b1 - n), (a2, b2 - n)
    if op.kind == "add":
        return (a1, b1 + n), (a2, b2 + n)
    if op.kind == "div":
        if n == 0:
            return None
        return (a1 / n, b1 / n), (a2 / n, b2 / n)
    if op.kind == "mul":
        if n == 0:
            return None
        return (a1 * n, b1 * n), (a2 * n, b2 * n)
    if op.kind == "move":
        # “把 n 移过去”：找到方程里等于 ±n 的常数项，等价于两边同减/同加
        if b1 == n or b2 == -n:
            return (a1, b1 - n), (a2, b2 - n)
        if b1 == -n or b2 == n:
            return (a1, b1 + n), (a2, b2 + n)
        return None
    return None


def _op_description(op: MathOperation) -> str:
    if op.kind == "move":
        return f"把 {fmt_frac(op.value)} 移到另一边（两边同时{'减去' if op.value > 0 else '加上'} {fmt_frac(abs(op.value))}）"
    return f"两边同时{_OP_VERBS[op.kind]} {fmt_frac(op.value)}"


def _find_equations_with_pos(text: str) -> list[tuple[int, str]]:
    t = _normalize(text)
    found: list[tuple[int, str]] = []
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
                found.append((l, format_equation(parsed)))
            i = r
        else:
            i += 1
    return found


def final_value_mention(text: str) -> Optional[Fraction]:
    """识别“最后等于1”这类直接给出的最终值（左侧不是可解析表达式）。"""
    t = _normalize(text)
    for m in re.finditer(r"=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)", t):
        l = m.start()
        while l > 0 and t[l - 1] in _MATH_CHARS and t[l - 1] != "=":
            l -= 1
        left = t[l:m.start()].strip()
        if left == "" or _parse_side(left) is None:
            try:
                return Fraction(m.group(1))
            except (ValueError, ZeroDivisionError):
                continue
    return None


def classify_student_math_response(message: str) -> str:
    """把学生消息分类为：why_question / multi_step_plan / single_step /
    intermediate_equation / final_answer / unknown。"""
    if "为什么" in message:
        return "why_question"
    ops = extract_operations(message)
    eqs = find_equations(message)
    if len(ops) + len(eqs) >= 2:
        return "multi_step_plan"
    if len(ops) == 1 and not eqs:
        return "single_step"
    if len(eqs) == 1 and not ops:
        parsed = parse_equation(eqs[0])
        if parsed is not None and is_final_form(parsed):
            return "final_answer"
        return "intermediate_equation"
    if bare_number(message) is not None or final_value_mention(message) is not None:
        return "final_answer"
    return "unknown"


def _simulate_plan(start_eq: str, message: str) -> PlanCheckResult:
    start = parse_equation(start_eq)
    if start is None:
        return PlanCheckResult(valid=False, explanation="当前方程无法解析")
    target_sol = solve(start)
    if target_sol is None:
        return PlanCheckResult(valid=False, explanation="当前方程没有唯一解")

    items: list[tuple[int, str, object]] = []
    for op in extract_operations(message):
        items.append((op.pos, "op", op))
    for pos, eq in _find_equations_with_pos(message):
        items.append((pos, "eq", eq))
    fv = None
    if not any(kind == "eq" for _, kind, _ in items):
        fv = final_value_mention(message)
    if fv is not None:
        items.append((len(message) + 1, "eq", f"x = {fmt_frac(fv)}"))
    items.sort(key=lambda x: x[0])
    if len(items) < 2:
        return PlanCheckResult(valid=False, explanation="不构成多步计划")

    state = start
    steps: list[str] = []
    for _, kind, payload in items:
        if kind == "op":
            new = apply_operation(state, payload)  # type: ignore[arg-type]
            if new is None:
                return PlanCheckResult(
                    valid=False, steps=steps,
                    explanation=f"“{_op_description(payload)}”在 {format_equation(state)} 上不可行",  # type: ignore[arg-type]
                )
            steps.append(f"{_op_description(payload)}，得到 {format_equation(new)}")  # type: ignore[arg-type]
            state = new
        else:
            eq_str = payload  # type: ignore[assignment]
            sol = solve_str(eq_str)
            if sol is None or sol != target_sol:
                return PlanCheckResult(
                    valid=False, steps=steps, end_equation=eq_str,
                    explanation=f"{eq_str} 与原方程 {start_eq} 不同解",
                )
            if format_equation(state) != eq_str:
                steps.append(describe_transformation(format_equation(state), eq_str))
            parsed = parse_equation(eq_str)
            assert parsed is not None
            state = parsed

    reaches_final = is_final_form(state)
    return PlanCheckResult(
        valid=True,
        reaches_final=reaches_final,
        final_value=solve(state) if reaches_final else None,
        steps=steps,
        end_equation=format_equation(state),
        explanation="整体思路正确",
    )


def check_multi_step_plan(
    start_eq: str, student_message: str, fallback_start: str | None = None
) -> PlanCheckResult:
    """整体检查学生的多步计划。

    先从当前推进到的方程模拟；若备选起点（原题）能把计划走到最终解，
    则优先采用（学生描述完整思路时通常是从原题出发的）。
    """
    primary = _simulate_plan(start_eq, student_message)
    if fallback_start and fallback_start != start_eq:
        alt = _simulate_plan(fallback_start, student_message)
        if alt.valid and alt.reaches_final and not (primary.valid and primary.reaches_final):
            return alt
        if alt.valid and not primary.valid:
            return alt
    return primary


def check_single_operation(current_eq: str, student_message: str) -> StepCheckResult:
    """检查“先减4”“两边除以3”这类单个操作描述。"""
    ops = extract_operations(student_message)
    if not ops:
        return StepCheckResult(StepValidity.NOT_A_STEP)
    parsed = parse_equation(current_eq)
    if parsed is None:
        return StepCheckResult(StepValidity.NOT_A_STEP, explanation="当前方程无法解析")
    new = apply_operation(parsed, ops[0])
    if new is None:
        return StepCheckResult(
            StepValidity.INCORRECT_STEP,
            explanation=f"“{_op_description(ops[0])}”在 {current_eq} 上不可行",
        )
    new_str = format_equation(new)
    validity = StepValidity.CORRECT_FINAL_ANSWER if is_final_form(new) else StepValidity.CORRECT_STEP
    return StepCheckResult(
        validity,
        student_equation=new_str,
        explanation=f"{_op_description(ops[0])}，得到 {new_str}",
    )


def check_final_answer(original_eq: str, student_message: str) -> StepCheckResult:
    """检查最终答案：支持 x=1、纯数字 1、“最后等于1”。"""
    fv = final_value_mention(student_message)
    if fv is not None and extract_equation(student_message) is None \
            and bare_number(student_message) is None:
        return check_step(original_eq, f"x = {fmt_frac(fv)}")
    return check_step(original_eq, student_message)
