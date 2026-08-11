from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("deliverables/slide_assets")
OUT.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def ui_mock() -> None:
    img = Image.new("RGB", (1600, 900), "#eef2f7")
    d = ImageDraw.Draw(img)
    rounded(d, (70, 60, 1530, 840), 34, "#ffffff", "#d8dee9", 3)
    d.rectangle((70, 60, 1530, 130), fill="#111827")
    d.ellipse((105, 84, 125, 104), fill="#ef4444")
    d.ellipse((138, 84, 158, 104), fill="#f59e0b")
    d.ellipse((171, 84, 191, 104), fill="#22c55e")
    d.text((235, 85), "LLM-led Teacher Prototype", font=font(28, True), fill="#f8fafc")

    rounded(d, (105, 165, 410, 805), 20, "#f8fafc", "#e2e8f0", 2)
    d.text((135, 195), "RAG status", font=font(24, True), fill="#0f172a")
    for y, label, color in [
        (250, "Local KB: ready", "#10b981"),
        (310, "Online: auto", "#2563eb"),
        (370, "Math audit: on", "#7c3aed"),
    ]:
        d.ellipse((135, y + 8, 153, y + 26), fill=color)
        d.text((170, y), label, font=font(22), fill="#334155")
    rounded(d, (135, 470, 380, 735), 16, "#ffffff", "#dbe3ee", 2)
    d.text((160, 500), "Retrieved", font=font(22, True), fill="#0f172a")
    for i in range(4):
        d.rounded_rectangle((160, 555 + i * 40, 350, 575 + i * 40), radius=8, fill="#dbeafe")

    rounded(d, (455, 165, 1485, 805), 20, "#ffffff", "#e2e8f0", 2)
    d.text((500, 200), "Student asks", font=font(22, True), fill="#64748b")
    rounded(d, (500, 245, 985, 315), 20, "#e0f2fe", None)
    d.text((535, 266), "3x + 4 = 9, how do I solve it?", font=font(24), fill="#075985")
    d.text((500, 375), "Teacher response", font=font(22, True), fill="#64748b")
    rounded(d, (500, 420, 1390, 590), 24, "#ecfdf5", None)
    d.text((535, 450), "Let's first isolate 3x. What can we do", font=font(27, True), fill="#064e3b")
    d.text((535, 490), "to both sides to remove +4?", font=font(27, True), fill="#064e3b")
    rounded(d, (500, 645, 1390, 735), 18, "#f8fafc", "#dbe3ee", 2)
    d.text((535, 675), "Claim audit: equation step verified by math_checker", font=font(23), fill="#475569")
    img.save(OUT / "ui_mock.png")


def architecture() -> None:
    img = Image.new("RGB", (1600, 900), "#f8fafc")
    d = ImageDraw.Draw(img)
    d.text((90, 70), "Teaching Agent Flow", font=font(46, True), fill="#0f172a")
    boxes = [
        (90, 240, "Student\nmessage", "#dbeafe", "#1d4ed8"),
        (320, 240, "LLM\nObserver", "#ede9fe", "#6d28d9"),
        (550, 240, "RAG\nretrieve", "#ffedd5", "#c2410c"),
        (780, 240, "Math\nchecker", "#dcfce7", "#15803d"),
        (1010, 240, "DeepSeek\nteacher", "#e0f2fe", "#0369a1"),
        (1240, 240, "Guard\nrepair", "#fee2e2", "#b91c1c"),
    ]
    for x, y, text, fill, stroke in boxes:
        rounded(d, (x, y, x + 190, y + 145), 24, fill, stroke, 3)
        lines = text.split("\n")
        for j, line in enumerate(lines):
            d.text((x + 34, y + 36 + j * 42), line, font=font(29, True), fill="#111827")
        if x < 1240:
            d.line((x + 198, y + 72, x + 222, y + 72), fill="#64748b", width=6)
            d.polygon([(x + 222, y + 72), (x + 205, y + 62), (x + 205, y + 82)], fill="#64748b")
    rounded(d, (430, 530, 760, 695), 26, "#ffffff", "#cbd5e1", 3)
    d.text((480, 575), "Student memory", font=font(30, True), fill="#0f172a")
    d.text((480, 620), "profile + episode", font=font(24), fill="#475569")
    d.line((595, 530, 595, 410), fill="#64748b", width=5)
    d.polygon([(595, 410), (585, 430), (605, 430)], fill="#64748b")
    rounded(d, (860, 530, 1190, 695), 26, "#ffffff", "#cbd5e1", 3)
    d.text((925, 575), "Turn record", font=font(30, True), fill="#0f172a")
    d.text((925, 620), "logs + references", font=font(24), fill="#475569")
    d.line((1025, 530, 1025, 410), fill="#64748b", width=5)
    d.polygon([(1025, 410), (1015, 430), (1035, 430)], fill="#64748b")
    img.save(OUT / "architecture.png")


def results() -> None:
    img = Image.new("RGB", (1600, 900), "#0f172a")
    d = ImageDraw.Draw(img)
    d.text((90, 75), "Ablation Result", font=font(48, True), fill="#f8fafc")
    d.text((90, 135), "2026 Henan chemistry exam, max score 50", font=font(25), fill="#cbd5e1")
    labels = ["Base LLM", "Local RAG", "Local + Web"]
    vals = [43, 48, 50]
    colors = ["#38bdf8", "#34d399", "#fbbf24"]
    for i, (label, val, color) in enumerate(zip(labels, vals, colors)):
        x = 210 + i * 440
        h = int(val / 50 * 420)
        d.rounded_rectangle((x, 690 - h, x + 230, 690), radius=24, fill=color)
        d.text((x + 63, 690 - h - 70), f"{val}/50", font=font(38, True), fill="#f8fafc")
        d.text((x + 22, 735), label, font=font(27, True), fill="#e2e8f0")
    d.line((130, 690, 1460, 690), fill="#334155", width=4)
    d.text((1110, 255), "+16% vs base", font=font(42, True), fill="#fbbf24")
    d.text((1110, 315), "Local knowledge base is the main lift.", font=font(26), fill="#cbd5e1")
    img.save(OUT / "results.png")


if __name__ == "__main__":
    ui_mock()
    architecture()
    results()
    print(OUT)
