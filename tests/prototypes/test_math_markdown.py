from prototypes.llm_led_teacher.math_markdown import normalize_math_markdown


def test_normalizes_model_bracket_latex_to_display_math():
    raw = "公式：\n\n[ x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a} ]\n\n其中 a 不为 0。"

    normalized, changed = normalize_math_markdown(raw)

    assert normalized == "公式：\n\n$$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$\n\n其中 a 不为 0。"
    assert changed is True


def test_does_not_change_ordinary_brackets_or_code_fences():
    raw = "取值范围是 [0, 1]。\n```python\nprint(r'\\frac{1}{2}')\n```"

    assert normalize_math_markdown(raw) == (raw, False)


def test_does_not_change_tilde_fenced_code_or_close_shorter_fence():
    raw = (
        "~~~python\n"
        "print(r'\\(x\\)')\n"
        "print(r'\\frac{1}{2}')\n"
        "~~~\n\n"
        "~~~~python\n"
        "print(r'\\(y\\)')\n"
        "~~~\n"
        "print(r'\\(z\\)')\n"
        "~~~~"
    )

    assert normalize_math_markdown(raw) == (raw, False)


def test_normalizes_standard_latex_delimiters():
    normalized, _ = normalize_math_markdown(r"行内 \\(a^2+b^2\\)，展示 \\[x=\\frac12\\]")

    assert normalized == r"行内 $a^2+b^2$，展示 $$x=\\frac12$$"
