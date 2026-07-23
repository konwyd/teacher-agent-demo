from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


OUT = Path("deliverables/teacher_agent_project_overview.pptx")

SLIDES = [
    {
        "title": "Teacher Agent Demo",
        "subtitle": "K9 教育 RAG 知识增强 + LLM 主导教学 Agent",
        "bullets": [
            "目标：让大模型像老师一样循序渐进讲题，而不是只给答案",
            "对象：中文初中数学为主，扩展到多学科中考知识库",
            "当前状态：Prototype UI + 本地/联网 RAG 已接通",
        ],
        "accent": "项目概览",
    },
    {
        "title": "我们解决什么问题",
        "bullets": [
            "单纯 LLM 会做题，但不一定会教学",
            "学生需要诊断卡点、分步提示、纠错和追问",
            "教育场景需要可控：不乱判步骤、不提前泄答案、不偏离原题",
        ],
        "accent": "从答题到教学",
    },
    {
        "title": "系统做了什么",
        "bullets": [
            "Agent workflow：观察学生输入，判断新题/追问，控制教学节奏",
            "math_checker：用规则验证数学步骤，LLM 不负责裁决正误",
            "Student memory：记录学生掌握度、对话状态和学习事件",
            "Response guard：审计生成内容，发现风险时修复",
        ],
        "accent": "教学闭环",
    },
    {
        "title": "RAG 知识库能力",
        "bullets": [
            "知识规模：约 28.8 万条，覆盖教材、题库、课标和教资资料",
            "混合检索：BM25 关键词 + bge-m3 语义向量 + RRF 融合排序",
            "多级回退：教材原文 → 相似题库 → Tavily 联网补充",
            "服务形态：ChromaDB + HNSW，支持本地毫秒级检索",
        ],
        "accent": "开卷教学",
    },
    {
        "title": "Prototype 教学流程",
        "bullets": [
            "用户消息 → LLM Observer → RAG 检索 → 数学验证",
            "DeepSeek 生成学生可读回复，默认一轮只推进一个关键步骤",
            "抽取可验证方程并审计，必要时触发 repair",
            "Streamlit 展示教师回复、RAG 状态和来源引用",
        ],
        "accent": "可演示链路",
    },
    {
        "title": "效果与当前状态",
        "bullets": [
            "化学中考消融：纯模型 43/50，本地 RAG 48/50，本地+联网 50/50",
            "测试覆盖：项目测试约 237 个通过，核心教学约束有回归用例",
            "当前待办：重启 Streamlit 让 rag 字段重新加载",
            "下一步：调优 RAG prompt、逐题 RAG、扩展更多学科教学策略",
        ],
        "accent": "结果与下一步",
    },
]


def content_types(n: int) -> str:
    overrides = [
        "/ppt/presentation.xml",
        "/ppt/slideMasters/slideMaster1.xml",
        "/ppt/slideLayouts/slideLayout1.xml",
        "/ppt/theme/theme1.xml",
        "/docProps/core.xml",
        "/docProps/app.xml",
    ]
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, n + 1)
    )
    base = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
            '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
            '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
            '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
            slide_overrides,
            "</Types>",
        ]
    )
    return base


def root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def presentation_xml(n: int) -> str:
    slide_ids = "\n".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, n + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{n + 1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def presentation_rels(n: int) -> str:
    rels = [
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, n + 1)
    ]
    rels.append(
        f'<Relationship Id="rId{n + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "\n".join(rels)
        + "</Relationships>"
    )


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <PresentationFormat>宽屏</PresentationFormat>
  <Slides>6</Slides>
</Properties>"""


def core_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Teacher Agent Demo 项目讲解</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-07-20T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-07-20T00:00:00Z</dcterms:modified>
</cp:coreProperties>"""


def theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="TeacherAgent">
  <a:themeElements>
    <a:clrScheme name="TeacherAgent">
      <a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="10B981"/></a:accent2>
      <a:accent3><a:srgbClr val="F59E0B"/></a:accent3><a:accent4><a:srgbClr val="EF4444"/></a:accent4>
      <a:accent5><a:srgbClr val="8B5CF6"/></a:accent5><a:accent6><a:srgbClr val="0F766E"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="8B5CF6"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="TeacherAgent"><a:majorFont><a:latin typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="TeacherAgent"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def slide_master_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def slide_master_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def slide_layout_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
</p:sldLayout>"""


def slide_layout_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


def shape_text(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, text: str, size: int, color: str, bold: bool = False) -> str:
    b = "<a:b/>" if bold else ""
    return f"""<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/><a:p><a:r><a:rPr lang="zh-CN" sz="{size}" dirty="0">{b}<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr><a:t>{escape(text)}</a:t></a:r><a:endParaRPr lang="zh-CN" sz="{size}"/></a:p></p:txBody>
</p:sp>"""


def bullet_box(shape_id: int, bullets: list[str], y: int) -> str:
    paragraphs = []
    for bullet in bullets:
        paragraphs.append(
            f'<a:p><a:pPr marL="342900" indent="-228600"><a:buChar char="•"/></a:pPr>'
            f'<a:r><a:rPr lang="zh-CN" sz="2600" dirty="0"><a:solidFill><a:srgbClr val="111827"/></a:solidFill></a:rPr>'
            f'<a:t>{escape(bullet)}</a:t></a:r><a:endParaRPr lang="zh-CN" sz="2600"/></a:p>'
        )
    return f"""<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="Bullets"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="1180000" y="{y}"/><a:ext cx="9500000" cy="3700000"/></a:xfrm><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody><a:bodyPr wrap="square" anchor="t"/><a:lstStyle/>{''.join(paragraphs)}</p:txBody>
</p:sp>"""


def rect(shape_id: int, x: int, y: int, cx: int, cy: int, color: str) -> str:
    return f"""<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="Accent"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr>
</p:sp>"""


def slide_xml(i: int, data: dict[str, object]) -> str:
    accent_colors = ["2563EB", "0F766E", "7C3AED", "EA580C", "0891B2", "16A34A"]
    color = accent_colors[(i - 1) % len(accent_colors)]
    subtitle = data.get("subtitle")
    parts = [
        rect(2, 0, 0, 220000, 6858000, color),
        shape_text(3, "Section", 870000, 520000, 3000000, 350000, str(data["accent"]), 1600, color, True),
        shape_text(4, "Title", 870000, 930000, 9800000, 900000, str(data["title"]), 3900, "111827", True),
    ]
    bullet_y = 2500000
    if subtitle:
        parts.append(shape_text(5, "Subtitle", 900000, 1830000, 9800000, 540000, str(subtitle), 2100, "475569"))
        bullet_y = 2850000
    parts.append(bullet_box(6, data["bullets"], bullet_y))
    parts.append(shape_text(7, "Footer", 10000000, 6300000, 1300000, 260000, f"{i:02d}", 1400, "94A3B8", True))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
  <p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{''.join(parts)}</p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def slide_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(SLIDES)))
        zf.writestr("_rels/.rels", root_rels())
        zf.writestr("docProps/app.xml", app_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("ppt/presentation.xml", presentation_xml(len(SLIDES)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(SLIDES)))
        zf.writestr("ppt/theme/theme1.xml", theme_xml())
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels())
        for i, slide in enumerate(SLIDES, 1):
            zf.writestr(f"ppt/slides/slide{i}.xml", slide_xml(i, slide))
            zf.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels())
    print(OUT)


if __name__ == "__main__":
    main()
